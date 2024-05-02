from huggingface_hub import login
import argparse
import json
import re
import jsonlines
from vllm import LLM, SamplingParams
import sys
from tqdm.auto import tqdm

def generate(model, data_path, tensor_parallel_size=1, temp=0.7, id=0, task="GSM8K"):
    inputs = []
    answers = []

    # Assume you are using 4 GPUs. for the parts that take longer, assign little less than others.

    if task == "GSM8K":
        if id == 0:
            start, end = 0, 1870 
        elif id == 1:
            start, end = 1870, 3740
        elif id == 2:
            start, end = 3740, 5610
        elif id == 3:
            start, end = 5610, 7473

    elif task == "MATH":
        if id == 0:
            start, end = 0, 1750 
        elif id == 1:
            start, end = 1750, 3750
        elif id == 2:
            start, end = 3750, 5750
        elif id == 3:
            start, end = 5750, 7500

    with open(data_path,"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            
            if not (start <= idx < end):
                continue
            
            inputs.append(item["query"].strip() + "\n")
                
            temp_ans = item['response'] # just add the response. We will worry about it later.
            answers.append(temp_ans)
    
    result_file = args.result_file.replace(".jsonl", f"_{id}.jsonl")

    try:
        already_done = [json.loads(x) for x in open(result_file)]
    except:
        already_done = []
    
    if len(already_done) != 0:
        inputs = inputs[len(already_done):]
        answers = answers[len(already_done):]
    
    if len(inputs) == 0 and len(answers) == 0:
        print("Already completed. Exiting.")
        return 

    stop_tokens = ["Problem:"]
    print("[GPU ID]", id, "Length of inputs", len(inputs))

    if temp == 0.7:
        n = 100 # needs modification
    else:
        n = 1
    
    if task == "GSM8K":
        max_token_num = 512
    else:
        max_token_num = 1024

    sampling_params = SamplingParams(temperature=temp, top_p=1, max_tokens=max_token_num, stop=stop_tokens, n=n)
    print('sampling =====', sampling_params)
    llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size, enforce_eager=True, swap_space=32)
    result = []
    res_completions = []

    completions = llm.generate(inputs, sampling_params)

    for num, output in enumerate(completions):
        prompt = output.prompt
        all_texts = [out.text for out in output.outputs]
        res_completions.append(all_texts)

        answer = answers[num]
        dict_ = {"prompt": prompt, "preds": all_texts, "answer": answer}

        with jsonlines.open(result_file, 'a') as writer:
            writer.write(dict_)            

    print('start===', start, ', end====', end)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)  # model path
    parser.add_argument("--data_file", type=str, default='')  # data path
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    parser.add_argument("--result_file", type=str, default="./log_file.jsonl")  # tensor_parallel_size
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--task", type=str, default="GSM8K")

    return parser.parse_args()

if __name__ == "__main__":
    # Login First.
    login(token="huggingface_token_here")

    args = parse_args()
    generate(model=args.model, data_path=args.data_file, tensor_parallel_size=args.tensor_parallel_size, temp=args.temp, id=args.id, task=args.task)

    # if all 4 exists, now unite file:
    import os
    if all([os.path.exists(args.result_file.replace(".jsonl", f"_{i}.jsonl")) for i in range(4)]):
        from utils_others import unite_file
        unite_file(args.result_file, 4, "_gen")
    
        # Then get rft and dpo data.
        gen_name = args.result_file.replace(".jsonl", "_gen.jsonl")

        # This will generate rft and dpo data.
        from get_rft_data import run_rft_and_dpo
        run_rft_and_dpo(gen_name, args.task)