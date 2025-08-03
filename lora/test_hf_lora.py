

"""
测试方案: https://infinigence.feishu.cn/wiki/G0dfwi7gIi4I8ykMUzdc2ByEnGe
代码逻辑: 
    1. 加载对应lora模型测试集
    2. 使用hf transformers加载lora模型, 推理上述测试集, 生成base数据(包含logproobs)
    3. 请求加载了同样lora模型的maas服务, 推理上述数据集，生产对比数据(包含logproobs)
    4. 对比base与maas的生成文本、数值精度
额外逻辑:
    1. 支持请求多并发, 模拟不同batch下, 命中不同算子测试
    2. 支持请求在多个lora之间切换, 模拟生产多lora应用场景(需要多训一个lora)
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import torch
from peft import PeftModel
import os
import json
import jsonlines
import random

USER_HOME = os.path.expanduser("~")
PERF_DATA_FOLDER = os.path.join(USER_HOME, '.cache/infini_eval/engine_accuracy_eval/')
os.makedirs(PERF_DATA_FOLDER, exist_ok=True)

model_path = '/share/datasets/public_models/Qwen_Qwen2.5-14B-Instruct/'

def run_hf():
    # 从环境中读取CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    lora_path = '/share/datasets/tmp_share/chenyonghua/models/loras/qwen2.5-14b-instruct_genshin_lines_lora'
    input_file = "datasets/genshin_lora_train/test/test_genshin_dataset_qwen25_14b_instruct_1000.jsonl"

    # lora_path = '/share/datasets/tmp_share/chenyonghua/models/loras/qwen2.5-32b-instruct_huanhuan_lines_lora'
    # input_file = "/share/datasets/tmp_share/chenyonghua/datasets/loras/test_huanhuan_dataset_qwen25_32b_instruct_1000.jsonl"
 
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

    # 加载lora权重
    model = PeftModel.from_pretrained(model, model_id=lora_path)


    # from transformers.generation.configuration_utils import GenerationConfig
    
    # 贪心采样
    count = 0
    items = []
    output_path = f"output/hf_lora_greedy_genshin_1000.jsonl"
    # output_path = f"{PERF_DATA_FOLDER}/hf_lora_greedy_huanhuan.jsonl"
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            data = json.loads(line)
            prompt = data.get("prompt", "")
            answer = data.get("answer", "")
            messages = [
                # {
                #     "role": "system",
                #     "content": "现在你要扮演皇帝身边的女人--甄嬛"
                # },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True
                ).to('cuda')
            # print(f"inputs {tokenizer.decode(inputs['input_ids'][0])}")
            
            gen_kwargs = {
                "max_length": 500,
                "do_sample": False,
                "temperature": 0,
                "output_scores": True,
                "return_dict_in_generate": True,
                "repetition_penalty": 1
            }
            with torch.no_grad():
                output = model.generate(**inputs, **gen_kwargs)

                # 获取生成的序列和scores
                generated_ids = output.sequences
                scores = output.scores  # 每一步生成的分数（logits）

                # 解码生成的序列
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                text = generated_text.split('assistant\n')[-1]
                if text == answer:
                    count += 1
                    print(f"Y request {i} {text}")
                else:
                    print(f"N request {i} {text}")

                # 计算每一步的 top-k 概率
                logprobs = []
                for step, logits in enumerate(scores):
                    # 使用 softmax 将 logits 转为概率分布
                    # print(f"logits {logits.shape} {logits}")
                    probs = torch.nn.functional.log_softmax(logits[0], dim=-1)

                    # 获取 top-k token 和其概率
                    top_probs, top_indices = torch.topk(probs, k=5)
                    top_tokens = [tokenizer.decode([idx]) for idx in top_indices]

                    top_logprobs = []
                    for token, prob in zip(top_tokens, top_probs.tolist()):
                        # print(f"  Token: {repr(token):<10} Probability: {prob:.10f}")
                        top_logprobs.append({
                            "token": token,
                            "logprob": f"{prob:.10f}"
                        })
                    
                    logprobs.append({
                        "step": step,
                        "top_logprobs": top_logprobs,
                    })

                # 本次请求的结果
                items.append({
                    "index": i,
                    "text": text,
                    "logprobs": logprobs
                })

    with jsonlines.open(output_path, 'w') as f:
        for item in items:
            f.write(item)

    print(f"贪心采样结束")        
    print(f"正确数 {count}, 成功率 {100*round(count / len(lines), 2)}%")

if __name__ == "__main__":
    run_hf()
