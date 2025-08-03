import json
import jsonlines
import statistics

input_a = "output/alioth_lora_greedy_genshin.jsonl"
input_b = "output/hf_lora_greedy_genshin_1000.jsonl"

res_path = "output/compare_alioth_hf_lora_greedy_genshin_1000.jsonl"

def compare_text_and_prob():
    with open(input_a, 'r', encoding='utf-8') as f:
        lines_a = f.readlines()
    
    with open(input_b, 'r', encoding='utf-8') as f:
        lines_b = f.readlines()

    assert len(lines_a) == len(lines_b)

    res = []
    accuracy_deviation_list = []
    total = len(lines_a)
    text_equal_num = 0
    first_diff_right_num = 0
    for i in range(len(lines_a)):
        line_a =json.loads(lines_a[i])
        line_b =json.loads(lines_b[i])
        idx = line_a["index"]
        text_equal = line_a["text"] == line_b["text"]
        first_diff_right = True
        msg = ""

        if text_equal:
            text_equal_num += 1
            continue
        else:
            logprobs_a = line_a["logprobs"]
            logprobs_b = line_b["logprobs"]
            for j in range(len(logprobs_a)):
                if j > len(logprobs_b) - 1:
                    first_diff_right = False
                    break
                step_a = logprobs_a[j]["top_logprobs"]
                step_b = logprobs_b[j]["top_logprobs"]
                assert len(step_a) == len(step_b) == 5

                # 第一个发生变化的token, token在对方的top5中且 logprob <= 0.3 即可            
                token_a = step_a[0]["token"]
                token_b = step_b[0]["token"]
                if token_a == token_b:
                    continue
                else:
                    token_prob_dic_a = {x["token"]: float(x["logprob"]) for x in step_a}
                    token_prob_dic_b = {x["token"]: float(x["logprob"]) for x in step_b}
                    if token_a in token_prob_dic_b.keys() and token_b in token_prob_dic_a.keys():
                        msg += f"step {j} token_a {token_a} and token_b {token_b} in other top5. "
                        accuracy_deviation_list.append(abs(token_prob_dic_b[token_a] - token_prob_dic_a[token_a]))
                        accuracy_deviation_list.append(abs(token_prob_dic_b[token_b] - token_prob_dic_a[token_b]))
                        # if abs(token_prob_dic_b[token_a] - token_prob_dic_a[token_a]) <= 0.3 and abs(token_prob_dic_b[token_b] - token_prob_dic_a[token_b]) <= 0.3:
                        #     msg += f"and logprob also right."                   
                        # else:
                        #     first_diff_right = False
                    else:
                        first_diff_right = False
                    break
        
            if first_diff_right:
                first_diff_right_num += 1

            res.append({
                "index": idx,
                "text_equal": text_equal,
                "first_diff_right": first_diff_right,
                "msg": msg
            })
    
    with jsonlines.open(res_path, 'w') as f:
        for item in res:
            f.write(item)

    print(f"比较文本和logprobs结束")
    average_accuracy_deviation = round(statistics.mean(accuracy_deviation_list), 2)
    print(f"端到端文本匹配数 {text_equal_num}/{total}, 未匹配文本数 {total-text_equal_num}/{total}。未匹配文本中, 首偏移token满足top5-logprobs范围的数量 {first_diff_right_num}/{total-text_equal_num}, 其中偏移的首token的平均偏移精度 {average_accuracy_deviation}")
    return


def compare_text_only():
    with open(input_a, 'r', encoding='utf-8') as f:
        lines_a = f.readlines()
    
    with open(input_b, 'r', encoding='utf-8') as f:
        lines_b = f.readlines()

    assert len(lines_a) == len(lines_b)

    res = []
    total = len(lines_a)
    text_equal_num = 0
    for i in range(len(lines_a)):
        line_a =json.loads(lines_a[i])
        line_b =json.loads(lines_b[i])
        idx = line_a["index"]
        text_equal = line_a["text"] == line_b["text"]
        msg = ""

        if text_equal:
            text_equal_num += 1
        else:
            msg = f'req {i} text not equal, A is {line_a["text"]}, B is {line_b["text"]}.'

        if not text_equal:
            res.append({
                "index": idx,
                "text_equal": text_equal,
                "msg": msg
            })
    
    with jsonlines.open(res_path, 'w') as f:
        for item in res:
            f.write(item)

    print(f"比较text结束")
    print(f"text_equal_num {text_equal_num}/{total}")
    return


if __name__ == "__main__":
    compare_text_and_prob()
    # compare_text_only()
