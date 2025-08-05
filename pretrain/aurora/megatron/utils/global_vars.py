import os
import torch.distributed as dist

_GLOBAL_DATA_CLIENT = None


def get_data_client():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_DATA_CLIENT


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)


def _set_data_client(args):
    global _GLOBAL_DATA_CLIENT
    _ensure_var_is_not_initialized(_GLOBAL_DATA_CLIENT,
                                   'data client')
    
    if args.aurora_test_type and dist.get_rank() == (dist.get_world_size() - 1):
        from aurora.common.data_store.data_client import DataClient
        if not args.aurora_save_dir:
            # Defaults to the save dir.
            args.aurora_save_dir = 'data_client'
        os.makedirs(args.aurora_save_dir, exist_ok=True)
        data_client = DataClient(args.aurora_save_dir, args)
        _GLOBAL_DATA_CLIENT = data_client


def set_global_data_client_variables(args):
    assert args is not None

    _set_data_client(args)

