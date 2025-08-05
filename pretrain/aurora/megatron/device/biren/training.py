# Copyright © 2024 Shanghai Biren Technology Co., Ltd. All rights reserved.
import dataclasses
from datetime import datetime
import math
import time
import sys
import gc
import os
import torch
import torch.distributed
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron.core import mpu
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.distributed import DistributedDataParallel as LocalDDP
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer import OptimizerConfig
from megatron.core.num_microbatches_calculator import get_num_microbatches, update_num_microbatches
from megatron.core.utils import get_model_config, check_param_hashes_across_dp_replicas
import megatron.training
from megatron.training import (
    get_args, get_timers, is_last_rank, get_tensorboard_writer)
from megatron.training.utils import (
    unwrap_model, report_memory, append_to_progress_log,
    calc_params_l2_norm, check_adlr_autoresume_termination)
from megatron.training.training import (
    evaluate, save_checkpoint_and_time, print_datetime,
    _TRAIN_START_TIME, num_floating_point_operations,
    build_train_valid_test_data_iterators,
    get_optimizer_param_scheduler, get_model)
from megatron.training.global_vars import get_one_logger, get_signal_handler, get_wandb_writer
from megatron.training.theoretical_memory_usage import report_theoretical_memory
from megatron.training.initialize import initialize_megatron, write_args_to_tensorboard
from megatron.training.checkpointing import save_checkpoint, load_checkpoint
from megatron.training.async_utils import maybe_finalize_async_save
from megatron.training import one_logger_utils
from megatron.legacy.model import Float16Module
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import get_megatron_optimizer

from megatron_br.training.utils import throughput_calculator
from megatron_br.training.kineto_profiler import global_profiler_handle
from megatron_br.training.precision_debugger_tool import precision_debugger_tool
from megatron_br.training.utils import print_rank_last, print_rank_0, br_log
from megatron_br.core.offload.common import reset_runtime_infos_micro_batch_id
from megatron_br.core.offload.async_offload import reset_offloaders
from megatron_br.training.initialize import wrapper_initialize_megatron, set_jit_fusion_options


from torch_br.supa.profiler_kineto import record_function
from megatron_br.training.real_time_profiler import RealTimeProfiler, record_kernel, record_log
from megatron_br.core.sbp_utils import use_sbp

from megatron_br.core.pipeline_parallel.weight_grad_store import reset_weight_grad_stores
from megatron_br.training.initialize import update_parallel_group_hardware_timeout

# add support for aurora
from aurora.megatron.utils.global_vars import (
    get_data_client,
    set_global_data_client_variables
)
from aurora.common.metrics.megatron_training_performance import MegatronPretrainPerformanceMetrics
from aurora.common.data_store.data_client import DataTopic

_is_first_step = True

def train_step(forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config, next_is_eval=False):
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    """Single training step."""
    args = get_args()
    args.stage_of_forward = 'do_training'
    timers = get_timers()

    # BIRENTECH ----
    if args.save and args.save_init_checkpoint:
        save_checkpoint_and_time(args.iteration, model, optimizer, opt_param_scheduler)
        torch.distributed.barrier()
        print_datetime('exiting program after saving initial checkpoint')
        sys.exit()
    # ---- BIRENTECH

    # BIRENTECH ----
    if precision_debugger_tool._is_enable_dump():
        precision_debugger_tool.set_modules_dump(model[0])

    reset_runtime_infos_micro_batch_id()
    # ---- BIRENTECH

    # Set grad to zero.
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    if use_sbp():
        optimizer.zero_grad(set_to_none=(not args.cpu_optimizer))
    else:
        optimizer.zero_grad()

    # Forward pass.
    def run_forward_backward_func():
        forward_backward_func = get_forward_backward_func()
        return forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False,
        )
    losses_reduced = run_forward_backward_func()

    # BIRENTECH ----
    reset_weight_grad_stores()
    # ---- BIRENTECH

    # in case model backward didn't call lazy init, call lazy init finally here.
    if not use_sbp():
        optimizer.lazy_init(custom_str="finally")

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()
        
    # Reduce gradients.
    if not use_sbp():
        with record_function("_exec_reduce_grads"):
            with record_kernel("_exec_reduce_grads"):
                optimizer.reduce_model_grads(args, timers)

    # Vision gradients.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0], (torchDDP, LocalDDP, Float16Module))
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    if args.enable_zero_bubble and args.enable_optimizer_post_validation:
        from megatron_br.core.pipeline_parallel.zb_schedules import get_zb_scheduler_instance
        zb_scheduler = get_zb_scheduler_instance()
        if optimizer.post_validation_enabled and not next_is_eval:
            optimizer.pre_step(args, timers)
            zb_scheduler.optimizer = optimizer
            assert not zb_scheduler.is_first_run and zb_scheduler.do_post_validation
            update_successful, grad_norm, num_zeros_in_grad = run_forward_backward_func()
            # Here num_zeros_in_grad is a fake name, representing for optimizer_rollback
        else:
            update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)
            zb_scheduler.is_first_run = True
        optimizer.record_grad_norm(grad_norm)
    else:
        with record_function("_exec_optimizer_step"):
            with record_kernel("_exec_optimizer_step"):
                if not use_sbp():
                    update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)
                else:
                    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers('optimizer').stop()

    # Gather params.
    if not use_sbp():
        if update_successful:
            with record_function("_exec_gather_model_params"):
                optimizer.gather_model_params(args, timers)

    # Vision momentum.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0], (torchDDP, LocalDDP, Float16Module))
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    # After the first training step, reduce the timeout value.
    if args.sync_timeout_enable:
        global _is_first_step
        if _is_first_step:
            from pybrml import brmlDeviceGetHandleByIndex, brmlDeviceSetHealthCheckCpTimeout, brmlDeviceSetSoftwareSyncTimeout
            brml_device = brmlDeviceGetHandleByIndex(args.local_rank)
            timeout_after_first_step = args.sync_timeout_after_first_step if args.sync_timeout_after_first_step > 60 else 60
            br_log.info(f'Setting sync timeout after the first step to {timeout_after_first_step}s for rank {args.local_rank}')
            brmlDeviceSetHealthCheckCpTimeout(brml_device, timeout_after_first_step+60)
            brmlDeviceSetSoftwareSyncTimeout(brml_device, timeout_after_first_step)  
            update_parallel_group_hardware_timeout()
            _is_first_step = False

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0].keys():
            numerator = 0
            denominator = 0
            for x in losses_reduced:
                val = x[key]
                # there is one dict per microbatch. in new reporting, we average
                # over the total number of tokens across the global batch.
                if isinstance(val, tuple) or isinstance(val, list):
                    numerator += val[0]
                    denominator += val[1]
                else:
                    # legacy behavior. we average over the number of microbatches,
                    # and so the denominator is 1.
                    numerator += val
                    denominator += 1
            loss_reduced[key] = numerator / denominator
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad

    reset_offloaders()
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    # BIRENTECH ----
    args.actual_seq_length = args.seq_length
    if not hasattr(args, 'consumed_train_tokens'):
        args.consumed_train_tokens = args.consumed_train_samples * args.actual_seq_length
        # new samples has been updated, do not need add more
    else:
        new_samples = mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
        args.consumed_train_tokens += new_samples * args.actual_seq_length
    seq_len = args.actual_seq_length
    # ---- BIRENTECH
    timers = get_timers()
    writer = get_tensorboard_writer()
    wandb_writer = get_wandb_writer()
    data_client = get_data_client()


    # BIRENTECH ----
    if precision_debugger_tool._is_enable_dump():
        precision_debugger_tool.step()
    RealTimeProfiler().step(iteration)
    # ---- BIRENTECH
    one_logger = get_one_logger()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.tensor([0.0], dtype=torch.float, device='cuda')) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        'forward-backward',
        'forward-compute',
        'backward-compute',
        'batch-generator',
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-forward-recv',
        'forward-send-backward-recv',
        'backward-send-forward-recv',
        'backward-send-backward-recv',
        'forward-backward-send-forward-backward-recv',
        'layernorm-grads-all-reduce',
        'embedding-grads-all-reduce',
        'all-grads-sync',
        'params-all-gather',
        'optimizer-copy-to-main-grad',
        'optimizer-unscale-and-check-inf',
        'optimizer-clip-main-grad',
        'optimizer-count-zeros',
        'optimizer-inner-step',
        'optimizer-copy-main-to-model-params',
        'optimizer']

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \
        get_num_microbatches()

    # Track app tag & app tag ID
    one_logger_utils.track_app_tag(batch_size, args.world_size, args.seq_length)

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)

    if args.num_experts is not None:
        moe_loss_scale = 1 / get_num_microbatches()
        track_moe_metrics(moe_loss_scale, iteration, writer, wandb_writer, total_loss_dict, args.moe_per_layer_logging)

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations

        seq_len = args.seq_length
        if hasattr(args, 'actual_seq_length'):
            seq_len = args.actual_seq_length

        # ---- BIRENTECH
        samples_per_sec, tflops, tgs = throughput_calculator(args, elapsed_time, total_iterations)
        # ---- BIRENTECH
        samples_per_sec_per_replica = samples_per_sec / args.data_parallel_size
        tokens_per_sec = samples_per_sec * seq_len
        tokens_per_sec_per_replica = tokens_per_sec / args.data_parallel_size
        tokens_per_gpu_per_second = tokens_per_sec / args.world_size
        tokens_per_gpu_per_second_per_replica = tokens_per_gpu_per_second / args.data_parallel_size
        
        throughput = num_floating_point_operations(args, batch_size) / (
            elapsed_time_per_iteration * 10**12 * args.world_size)
        total_throughput = throughput * args.world_size

        one_logger_utils.track_e2e_metrics(args.log_throughput, throughput)

        # INFINI-AI ----
        if wandb_writer:
            wandb_metrics = {
                'throughput/iteration-time': elapsed_time_per_iteration,  # 1000 ms / s
                'throughput/samples_per_sec': samples_per_sec,
                'throughput/samples_per_sec_per_replica': samples_per_sec_per_replica,
                'throughput/tokens_per_sec': tokens_per_sec,
                'throughput/tokens_per_sec_per_replica': tokens_per_sec_per_replica,
                'throughput/tokens_per_gpu_per_sec': tokens_per_gpu_per_second,
                'throughput/tokens_per_gpu_per_sec_per_replica': tokens_per_gpu_per_second_per_replica,
                'throughput/tflops': tflops,
                'throughput/elapsed_ms_per_iteration': elapsed_time_per_iteration,
                'throughput/iteration': iteration,
            }
            if loss_dict is not None:
                wandb_metrics.update({
                    f'loss/{k}': v for k, v in loss_dict.items()
                })
                wandb_metrics.update({'loss/iteration': iteration})
        
        if data_client:
            performance_metrics = MegatronPretrainPerformanceMetrics(
                iteration=iteration,
                elapsed_time_per_iteration=elapsed_time_per_iteration,
                samples_per_sec=samples_per_sec,
                tokens_per_sec=tokens_per_sec,
                tokens_per_sec_per_replica=tokens_per_sec_per_replica,
                tokens_per_gpu_per_second=tokens_per_gpu_per_second,
                tokens_per_gpu_per_second_per_replica=tokens_per_gpu_per_second_per_replica,
                total_tflops=total_throughput,
                tflops_per_gpu=throughput,
                loss=loss_dict['lm loss'].item(),
            )
            data_client.send_data(DataTopic.TRAINING_MEGATRON_PRETRAIN_PERFORMANCE, performance_metrics)
        # ---- INFINI-AI

        if RealTimeProfiler().is_running():
            record_dict = {
                "iteration": iteration,
                "samples": args.consumed_train_samples,
                "tokens": args.consumed_train_tokens,
                "iteration_time": elapsed_time_per_iteration*1000,
                "tgs": tgs,
                "tflops": tflops,
                "loss_scale": loss_scale,
                "grad_norm": grad_norm,
                "num_zeros": num_zeros_in_grad if num_zeros_in_grad != None else 0,
                "params_norm": params_norm if params_norm != None else 0,
                "samples_per_sec": samples_per_sec,
            }
            
            for key in total_loss_dict:
                if key not in [advanced_iters_key, skipped_iters_key, nan_iters_key]:
                    avg = total_loss_dict[key].item() / float(max(1, total_loss_dict[advanced_iters_key]))
                    if avg > 0.0:
                        record_dict[key.replace(" ", "_")] = round(avg, 3)
            record_log(
                "training_state",
                record_dict
            )
        # ---- BIRENTECHS
        log_string = ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        if args.log_throughput:
            log_string += f' throughput per GPU (TFLOP/s/GPU): {throughput:.1f} |'
            if args.log_timers_to_tensorboard:
                if writer:
                    writer.add_scalar('throughput', throughput, iteration)
                if wandb_writer:
                    wandb_writer.log({'throughput': throughput}, iteration)
        assert learning_rate is not None
        # Decoupled_learning_rate should be not None only on first and last pipeline stage.
        log_string += ' learning rate: {:.6E} |'.format(learning_rate)
        if args.decoupled_lr is not None and (mpu.is_pipeline_first_stage(ignore_virtual=True) or
                                              mpu.is_pipeline_last_stage(ignore_virtual=True)):
            assert decoupled_learning_rate is not None
            log_string += ' decoupled learning rate: {:.6E} |'.format(decoupled_learning_rate)
        else:
            assert decoupled_learning_rate is None
        log_string += ' global batch size: {:5d} |'.format(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device='cuda')
        log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)
        # BIRENTECH ----
        log_string += ' actual seqlen: {:5d} |'.format(seq_len)
        # ---- BIRENTECH
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
	    # BIRENTECH ----
        log_string += ' samples per second: {:.3f} |'.format(samples_per_sec)
        log_string += ' TFLOPs: {:.2f} |'.format(tflops)
        log_string += ' TGS: {:.2f} |'.format(tgs)
	    # ---- BIRENTECH
        # INFINI-AI ----
        if wandb_writer:
            wandb_metrics.update({
                'training/iteration': iteration,
                'training/iteration_time': elapsed_time_per_iteration,
                'training/iteration_time_vs_samples': (
                    (elapsed_time_per_iteration
                        / args.consumed_train_samples),
                ),
                'training/consumed_samples': args.consumed_train_samples,
            })

        if wandb_writer:
            wandb_writer.log(wandb_metrics)

        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag and learning_rate > 0.:
            # Report memory after optimizer state has been initialized.
            if torch.distributed.get_rank() == 0:
                num_microbatches = get_num_microbatches()
                report_theoretical_memory(args, num_microbatches=num_microbatches, verbose=True)
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def evaluate(forward_step_func, data_iterator, model, process_non_loss_data_func, config, verbose=False):
    """Evaluation."""
    args = get_args()

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        compute_feature_bank(model)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    total_loss_dict = {}

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration, args.eval_iters))

            forward_backward_func = get_forward_backward_func()
            loss_dicts = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
            )

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        total_loss_dict[key] = total_loss_dict.get(key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]

            args.consumed_valid_samples += (
                mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
            )
        collected_non_loss_data = None
        if process_non_loss_data_func is not None and is_last_rank():
            collected_non_loss_data = forward_backward_func(
                forward_step_func,
                data_iterator,
                model,
                optimizer=None,
                timers=None,
                forward_only=True,
                collect_non_loss_data=True,
            )

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * get_num_microbatches()

    return total_loss_dict, collected_non_loss_data


def evaluate_and_print_results(
    prefix, forward_step_func, data_iterator, model, iteration, process_non_loss_data_func, config, verbose=False
):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    args.stage_of_forward = 'do_valid'
    writer = get_tensorboard_writer()
    # BIRENTECH ----
    wandb_writer = get_wandb_writer()
    # ---- BIRENTECH

    total_loss_dict, collected_non_loss_data = evaluate(
        forward_step_func, data_iterator, model, process_non_loss_data_func, config, verbose
    )
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer:
            writer.add_scalar('{} validation'.format(key), total_loss_dict[key].item(), iteration)
            writer.add_scalar(
                '{} validation vs samples'.format(key), total_loss_dict[key].item(), args.consumed_train_samples
            )
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar('{} validation ppl'.format(key), ppl, iteration)
                writer.add_scalar('{} validation ppl vs samples'.format(key), ppl, args.consumed_train_samples)
            # BIRENTECH ----
            if wandb_writer and is_last_rank():
                wandb_writer.log({'{} validation'.format(key): total_loss_dict[key].item()}, iteration)
    # ---- BIRENTECH
    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, iteration, writer)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)


def train(forward_step_func, model, optimizer, opt_param_scheduler,
          train_data_iterator, valid_data_iterator,
          process_non_loss_data_func, config, checkpointing_context):
    """Train the model function."""
    args = get_args()
    timers = get_timers()
    one_logger = get_one_logger()

    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration

    # Track E2E metrics at the start of training
    one_logger_utils.on_train_start(iteration=iteration, consumed_train_samples=args.consumed_train_samples,
                                    train_samples=args.train_samples, seq_length=args.seq_length,
                                    train_iters=args.train_iters, save=args.save, async_save=args.async_save,
                                    log_throughput=args.log_throughput,
                                    num_floating_point_operations_so_far=args.num_floating_point_operations_so_far)

    num_floating_point_operations_so_far = args.num_floating_point_operations_so_far

    # Setup some training config params
    config.grad_scale_func = optimizer.scale_loss
    config.timers = timers
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, \
            ('When overlap_grad_reduce is True, config.no_sync_func must be None; '
             'a custom no_sync_func is not supported when overlapping grad-reduce')
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.delay_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.delay_param_gather:
        config.param_sync_func = [lambda x: optimizer.finish_param_sync(model_index, x)
                                  for model_index in range(len(model))]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]

    if use_sbp():
        config.finalize_model_grads_func = finalize_model_grads
    else:
        config.finalize_model_grads_func = None

    timers('interval-time', log_level=0).start(barrier=True)
    print_datetime('before the start of training step')
    report_memory_flag = True
    exit = False

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, \
            'Manual garbage collection interval should be laerger than or equal to 0.'
        gc.disable()
        gc.collect()

    # Singleton Initialization
    if args.log_straggler:
        global stimer
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        mmcnt = args.straggler_minmax_count
        stimer.configure(world, rank,
                mmcnt = mmcnt,
                enabled = not args.disable_straggler_on_startup,
                port = args.straggler_ctrlr_port)
    total_flops = 0.0

    num_microbatches = get_num_microbatches()
    eval_duration = 0.0
    eval_iterations = 0

    def get_e2e_base_metrics():
        """Get base metrics values for one-logger to calculate E2E tracking metrics.
        """
        return {
            'iteration': iteration,
            'train_duration': timers('interval-time').active_time(),
            'eval_duration': eval_duration,
            'eval_iterations': eval_iterations,
            'total_flops': total_flops,
            'num_floating_point_operations_so_far': num_floating_point_operations_so_far,
            'consumed_train_samples': args.consumed_train_samples,
            'world_size': args.world_size,
            'seq_length': args.seq_length
        }
    # Cache into one-logger for callback
    if one_logger:
        with one_logger.get_context_manager():
            one_logger.store_set('get_e2e_base_metrics', get_e2e_base_metrics)

    while iteration < args.train_iters:
        if args.profile and \
           iteration == args.profile_step_start and \
           torch.distributed.get_rank() in args.profile_ranks:
            torch.cuda.cudart().cudaProfilerStart()
            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        maybe_finalize_async_save(False)

        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        update_num_microbatches(args.consumed_train_samples, consistency_check=False)
        if get_num_microbatches() != num_microbatches and iteration != 0:
            assert get_num_microbatches() > num_microbatches, \
                "number of microbatches should be increasing due to batch size rampup"
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far,
                                     checkpointing_context)
        num_microbatches = get_num_microbatches()
        update_num_microbatches(args.consumed_train_samples, consistency_check=True)

        args.curr_iteration = iteration
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
            train_step(forward_step_func,
                       train_data_iterator,
                       model,
                       optimizer,
                       opt_param_scheduler,
                       config)
        iteration += 1
        batch_size = mpu.get_data_parallel_world_size() * \
                     args.micro_batch_size * \
                     get_num_microbatches()
        args.consumed_train_samples += batch_size
        num_fp_ops = num_floating_point_operations(args, batch_size)
        num_floating_point_operations_so_far += num_fp_ops
        total_flops += num_fp_ops

        # Logging.
        with record_function("_exec_logging"):
            loss_scale = optimizer.get_loss_scale().item()
            params_norm = None
            if args.log_params_norm:
                params_norm = calc_params_l2_norm(model)


            learning_rate = None
            decoupled_learning_rate = None
            for param_group in optimizer.param_groups:
                if param_group['is_decoupled_lr']:
                    decoupled_learning_rate = param_group['lr']
                else:
                    learning_rate = param_group['lr']
            report_memory_flag = training_log(loss_dict, total_loss_dict,
                                            learning_rate,
                                            decoupled_learning_rate,
                                            iteration, loss_scale,
                                            report_memory_flag, skipped_iter,
                                            grad_norm, params_norm, num_zeros_in_grad)
        # StragglerDetector
        if iteration % args.log_interval == 0 and args.log_straggler:
            stimer.report(total_flops, args.log_interval)
            total_flops = 0.0

        if args.check_weight_hash_across_dp_replicas_interval is not None and \
                iteration % args.check_weight_hash_across_dp_replicas_interval == 0:
            if args.use_distributed_optimizer and args.overlap_param_gather:
                optimizer.disable_pre_hook()
            assert check_param_hashes_across_dp_replicas(model), \
                "Parameter hashes not matching across DP replicas"
            torch.distributed.barrier()
            print_rank_0(f">>> Weight hashes match after {iteration} iterations...")
            if args.use_distributed_optimizer and args.overlap_param_gather:
                optimizer.enable_pre_hook()

        # Autoresume
        if args.adlr_autoresume and \
           (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              opt_param_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \
           args.do_valid:
            timers('interval-time').stop()
            if args.use_distributed_optimizer and args.overlap_param_gather:
                optimizer.disable_pre_hook()
            if args.manual_gc and args.manual_gc_eval:
                # Collect all objects.
                gc.collect()
            prefix = 'iteration {}'.format(iteration)
            timers('eval-time', log_level=0).start(barrier=True)
            evaluate_and_print_results(prefix, forward_step_func,
                                       valid_data_iterator, model,
                                       iteration, process_non_loss_data_func,
                                       config, False)
            eval_duration += timers('eval-time').elapsed()
            eval_iterations += args.eval_iters
            timers('eval-time').stop()
            one_logger_utils.track_e2e_metrics()

            if args.manual_gc and args.manual_gc_eval:
                # Collect only the objects created and used in evaluation.
                gc.collect(generation=0)
            if args.use_distributed_optimizer and args.overlap_param_gather:
                optimizer.enable_pre_hook()
            timers('interval-time', log_level=0).start(barrier=True)

        # Checkpointing
        saved_checkpoint = False
        if args.exit_signal_handler:
            signal_handler = get_signal_handler()
            if any(signal_handler.signals_received()):
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler,
                                         num_floating_point_operations_so_far,
                                         checkpointing_context)
                print_datetime('exiting program after receiving SIGTERM.')
                exit = True
                break

        if args.save and args.save_interval and \
           iteration % args.save_interval == 0:
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far,
                                     checkpointing_context)
            saved_checkpoint = True

        # Exiting based on duration
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.tensor(
                [train_time > args.exit_duration_in_mins],
                dtype=torch.int, device='cuda')
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(iteration, model, optimizer,
                                             opt_param_scheduler,
                                             num_floating_point_operations_so_far,
                                             checkpointing_context)
                print_datetime('exiting program after {} minutes'.format(train_time))
                exit = True
                break

        # Exiting based on iterations
        if args.exit_interval and iteration % args.exit_interval == 0:
            if args.save and not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler,
                                         num_floating_point_operations_so_far,
                                         checkpointing_context)
            torch.distributed.barrier()
            print_datetime('exiting program at iteration {}'.format(iteration))
            exit = True
            break

        if args.profile and \
           iteration == args.profile_step_end and \
           torch.distributed.get_rank() in args.profile_ranks:
            torch.cuda.cudart().cudaProfilerStop()
        
        # BIRENTECH ----
        global_profiler_handle.step()
        # ---- BIRENTECH

        if args.manual_gc:
            if args.manual_gc_interval != 0 and iteration % args.manual_gc_interval == 0:
                gc.collect()

    one_logger_utils.track_e2e_metrics()

    # Flush TensorBoard, WandB writers and one-logger
    writer = get_tensorboard_writer()
    if writer:
        writer.flush()
    wandb_writer = get_wandb_writer()
    if wandb_writer:
        wandb_writer.finish()

    data_client = get_data_client()
    if data_client:
        data_client.finish()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if args.use_distributed_optimizer and args.overlap_param_gather:
        optimizer.disable_pre_hook()

    maybe_finalize_async_save(True)

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if exit:
        sys.exit()

    return iteration, num_floating_point_operations_so_far


def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0):
    """Setup model and optimizer."""
    args = get_args()
    timers = get_timers()
    one_logger = get_one_logger()

    model = get_model(model_provider_func, model_type)
    unwrapped_model = unwrap_model(model)

    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = timers
    optimizer = get_megatron_optimizer(config, model, no_wd_decay_cond,
                                       scale_lr_cond, lr_mult)
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    if args.load is not None or args.pretrained_checkpoint is not None:
        one_logger and one_logger.log_metrics({
            'load_checkpoint_start_time': one_logger_utils.get_timestamp_in_ms()
        })
        timers('load-checkpoint', log_level=0).start(barrier=True)
        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
            model, optimizer, opt_param_scheduler)
        timers('load-checkpoint').stop(barrier=True)
        timers.log(['load-checkpoint'])
        
        # BIRENTECH ----
        # update train_iters
        original_train_iters = args.train_iters
        
        if args.train_samples:
            remaining_iterations = (args.train_samples - args.consumed_train_samples) // args.global_batch_size
            args.train_iters = args.iteration + remaining_iterations
        if args.train_iters != original_train_iters:
            print_rank_0(f"[LOAD] update train_iters from {original_train_iters} to {args.train_iters}, remaining:{remaining_iterations}")
        # ---- BIRENTECH
        one_logger and one_logger.log_metrics({
            'load_checkpoint_finish_time': one_logger_utils.get_timestamp_in_ms(),
            'load_checkpoint_time': timers('load-checkpoint').active_time()
        })
    else:
        args.iteration = 0
        args.num_floating_point_operations_so_far = 0

    # get model without FP16 and/or DDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    return model, optimizer, opt_param_scheduler

def pretrain(train_valid_test_dataset_provider,
             model_provider,
             model_type,
             forward_step_func,
             process_non_loss_data_func=None,
             extra_args_provider=None,
             args_defaults={}):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Args:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    args = get_args()
    timers = get_timers()

    if args.log_progress:
        append_to_progress_log("Starting job")

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    # BIRENTECH ----
    # br166不支持double
    start_time_tensor = torch.tensor([_TRAIN_START_TIME],
                                     dtype=torch.float32,
                                     device='cuda')
    # ---- BIRENTECH
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()

    app_metrics = {}
    app_metrics['app_start_time'] = round(_TRAIN_START_TIME * 1000.0)
    app_metrics['app_model_init_start_time'] = round(_TRAIN_START_TIME * 1000.0)

    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')
    app_metrics['app_model_init_finish_time'] = one_logger_utils.get_timestamp_in_ms()

    args = get_args()
    timers = get_timers()

    # Track E2E metrics on pretrain start
    one_logger_utils.on_pretrain_start()
    set_global_data_client_variables(args)  # aurora: setup data client

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    app_metrics['app_build_optimizer_start_time'] = one_logger_utils.get_timestamp_in_ms()
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type)

    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')
    app_metrics['app_build_optimizer_finish_time'] = one_logger_utils.get_timestamp_in_ms()
    config = get_model_config(model[0])

    # Data stuff.
    app_metrics['app_build_dataiters_start_time'] = one_logger_utils.get_timestamp_in_ms()
    timers('train/valid/test-data-iterators-setup', log_level=0).start(
        barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            iterators = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')
    app_metrics['app_build_dataiters_finish_time'] = one_logger_utils.get_timestamp_in_ms()

    # Track if training is enabled. Can only be done once args.do_train is assigned after dataloader is built.
    one_logger_utils.track_config_flags(args.train_iters, args.skip_train, args.do_train,
                                        args.do_valid, args.do_test, args.dataloader_type,
                                        args.retro_project_dir, args.retro_cyclic_train_iters)

    # Context used for persisting some state between checkpoint saves.
    checkpointing_context = {}

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup',
                'train/valid/test-data-iterators-setup'], barrier=True)

    one_logger = get_one_logger()
    one_logger and one_logger.log_metrics(app_metrics)

    if not args.skip_train:
        print_rank_0('training ...')

        if args.dataloader_type == 'cyclic' and args.retro_project_dir:
            assert args.retro_cyclic_train_iters is not None
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0
        if args.do_train and args.train_iters > 0:
            iteration, num_floating_point_operations_so_far = train(
                forward_step_func,
                model, optimizer, opt_param_scheduler,
                train_data_iterator, valid_data_iterator,
                process_non_loss_data_func, config, checkpointing_context)

        print_datetime('after training is done')

        if args.save and iteration != 0 and iteration % args.save_interval != 0:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                            num_floating_point_operations_so_far, checkpointing_context)

        one_logger and one_logger.log_metrics({
            'app_train_loop_finish_time': one_logger_utils.get_timestamp_in_ms()
        })

    else:
        print_rank_0('skipping training (--skip-train is on) ...')

        iteration = args.iteration

    if args.do_valid:
        prefix = f'iteration {iteration} on validation set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)

    if args.do_test:
        prefix = f'iteration {iteration} on test set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)

    maybe_finalize_async_save(blocking=True)

    one_logger and one_logger.log_metrics({
        'app_finish_time': one_logger_utils.get_timestamp_in_ms()
    })
    one_logger_utils.finish()