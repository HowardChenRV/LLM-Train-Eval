import json
import os
import jsonlines
from typing import List, Tuple, Optional
from itertools import chain
from dataclasses import dataclass
import aiohttp
import asyncio
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer


USER_HOME = os.path.expanduser("~")
PERF_DATA_FOLDER = os.path.join(USER_HOME, '.cache/infini_eval/engine_accuracy_eval/')

# AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
TCP_TIMEOUT = 30 * 60

@dataclass
class RequestFuncInput:
    model: str
    prompt: str
    system_prompt: str


def request_chat_completion():
    # model_path = '/mnt/public/Qwen2.5-14B-Instruct'
    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model_id = 'uas-dalrzsfcik7lkqth'
    input_file = "datasets/genshin_lora_train/test/test_genshin_dataset_qwen25_14b_instruct_1000.jsonl"
    api_url = "http://if-dalr6oybd6me5mr3-service:80/v1/chat/completions"
    output_path = f"output/alioth_lora_greedy_genshin.jsonl"
    # model_id = 'lora_1'
    # input_file = "/share/datasets/tmp_share/chenyonghua/datasets/loras/test_huanhuan_dataset_qwen25_32b_instruct_1000.jsonl"
    # api_url = "http://127.0.0.1:8000/v1/chat/completions"
    # output_path = f"{PERF_DATA_FOLDER}/alioth_lora_greedy_huanhuan_1217.jsonl"
    
    session = requests.Session()
    headers = {
        "Content-Type": "application/json",
    }

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    items = []
    count = 0
    for i, line in enumerate(lines):
        data = json.loads(line)
        prompt = data.get("prompt", "")
        answer = data.get("answer", "")
        payload = {
            "model": model_id,
            "messages": [
                # {
                #     "role": "system",
                #     "content": "现在你要扮演皇帝身边的女人--甄嬛"
                # },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": 500,
            "temperature": 0,
            "stream": False,
            "logprobs": True,
            "top_logprobs":5
        }
        with session.post(url=api_url, json=payload, headers=headers, timeout=TCP_TIMEOUT) as response:
            content = response.content.decode('utf-8')
            data = json.loads(content)
            generated_text = data["choices"][0]["message"]["content"]
            
            if generated_text == answer:
                count += 1
                print(f"Y request {i} {generated_text}")
            else:
                print(f"N request {i} {generated_text}")

            logprobs = []
            for step, logprob in enumerate(data["choices"][0]["logprobs"]["content"]):
                top_logprobs = []
                for top_logprob in logprob["top_logprobs"]:
                    top_logprobs.append({
                        "token": top_logprob["token"],
                        "logprob": top_logprob["logprob"]
                    })
                logprobs.append({
                    "step": step,
                    "top_logprobs": top_logprobs
                })

            items.append({
                "index": i,
                "text": generated_text,
                "logprobs": logprobs,
            })

    with jsonlines.open(output_path, 'w') as f:
        for item in items:
            f.write(item)

    print(f"贪心采样结束")
    print(f"正确数 {count}, 成功率 {100*round(count / len(lines), 2)}%")
    return


async def aiohttp_chat_completion():
    # 修改并发和输出路径
    concurrency = 4

    api_url = "http://127.0.0.1:8000/v1/chat/completions"
    output_path_a = f"{PERF_DATA_FOLDER}/alioth_con_{concurrency}_greedy_genshin_1217.jsonl"
    output_path_b = f"{PERF_DATA_FOLDER}/alioth_con_{concurrency}_greedy_huanhuan_1217.jsonl"


    # 创建session
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    tcp_limit = concurrency if concurrency > 100 else 100
    conn = aiohttp.TCPConnector(limit=tcp_limit)
    buf_size = 256 * 1024 
    session = aiohttp.ClientSession(connector=conn, timeout=timeout, read_bufsize=buf_size)

    tasks = [handle_one_parallel(i, session, api_url) for i in range(concurrency)]
    outputs = await asyncio.gather(*tasks)
    print(f"outputs len {len(outputs)}")

    await session.close()

    output_a = outputs[0]
    with jsonlines.open(output_path_a, 'w') as f:
        for idx, text in enumerate(output_a):
            item = {"index": idx, "text": text}
            f.write(item)

    output_b = outputs[1]
    with jsonlines.open(output_path_b, 'w') as f:
        for idx, text in enumerate(output_b):
            item = {"index": idx, "text": text}
            f.write(item)
    print(f"outputs写入完成")


async def handle_one_parallel(i, session, api_url):
    # 一半请求lora_0, 一半请求lora_1
    if i % 2 == 0:
        model_id = "lora_0"
        input_file = "/share/datasets/tmp_share/chenyonghua/datasets/loras/test_genshin_dataset_qwen25_32b_instruct_921.jsonl"

    else:
        model_id = "lora_1"
        input_file = "/share/datasets/tmp_share/chenyonghua/datasets/loras/test_huanhuan_dataset_qwen25_32b_instruct_1000.jsonl"

    # Load the dataset from path
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    filtered_dataset: List[Tuple[str, str]] = []
    for data in lines:
        data = json.loads(data)
        filtered_dataset.append(
            (data.get("system_prompt", ""), data["prompt"])
        )

    # 串行化，一个返回再发下一个
    results = []
    datas = iter(filtered_dataset)
    for item in datas:
        system_prompt, prompt = item
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            system_prompt=system_prompt,
        )
        results.append(await async_request_session_openai_chat_completions(i, session, api_url, request_func_input=request_func_input))
    return results


async def async_request_session_openai_chat_completions(
        i: int,
        session: aiohttp.ClientSession,
        api_url: str,
        request_func_input: RequestFuncInput
):
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "model": request_func_input.model,
        "messages": [
            {
                "role": "system",
                "content": request_func_input.system_prompt,
            },
            {
                "role": "user",
                "content": request_func_input.prompt,
            },
        ],
        "max_tokens": 500,
        "temperature": 0,
        "stream": False,
        "logprobs": True,
        "top_logprobs":5
    }
    async with session.post(url=api_url, json=payload, headers=headers, timeout=TCP_TIMEOUT) as response:
        async for chunk in response.content:
            chunk = chunk.decode('utf-8')
            data = json.loads(chunk)
            if data.get("choices", "") == "":
                generated_text = ""
                print(f"con {i} chunk {data}")
            else:
                generated_text = data["choices"][0]["message"]["content"]
                print(f"con {i} {generated_text}")
    return generated_text
    

if __name__ == "__main__":
    request_chat_completion()
    # asyncio.run(aiohttp_chat_completion())
