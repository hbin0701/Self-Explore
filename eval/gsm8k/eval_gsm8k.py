import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

from huggingface_hub import login
import argparse
import json
import re
import jsonlines
from fraction import Fraction
from vllm import LLM, SamplingParams
import sys
from tqdm.auto import tqdm  
from utils_ans import extract_answer

MAX_INT = sys.maxsize

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def gsm8k_test(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1, temp=0.7):
    gsm8k_ins = []
    gsm8k_answers = []
 
    # Check if it already exists.
    try:
        already_done = [json.loads(x) for x in open(args.result_file)]
    except:
        already_done = [] 

    with open(data_path,"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            
            if idx < len(already_done):
                continue

            gsm8k_ins.append(item["query"])
            temp_ans = int(item['response'].split('#### ')[1].replace(',', ''))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('length ====', len(gsm8k_ins))
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    # stop_tokens = ["\n\n", "Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    stop_tokens = []
        
    if temp == 0.7:
        n = 100
    else:
        n = 1
    
    sampling_params = SamplingParams(temperature=temp, top_p=1, max_tokens=512, stop=stop_tokens, n=n)
    print('sampling =====', sampling_params)
    llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size, enforce_eager=True)
    result = []
    res_completions = []
    
    for idx, (prompt, prompt_answer) in tqdm(enumerate(zip(batch_gsm8k_ins, gsm8k_answers))):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(prompt, sampling_params)

        for num, output in enumerate(completions):
            prompt = output.prompt
            all_texts = [out.text for out in output.outputs]
            res_completions.append(all_texts)

            answer = gsm8k_answers[idx*batch_size + num]
            dict_ = {"prompt": prompt, "preds": all_texts, "answer": answer}

            with jsonlines.open(args.result_file, 'a') as writer:
                writer.write(dict_)            
    
        li = [json.loads(x) for x in open(args.result_file)]

        sa = [] # singgle acc

        for x in li:
            if 'answers' in x:
                lbl = str(x['answers'])
            else:
                lbl = str(x['answer'])

            answers = [str(extract_answer(pred)) for pred in x['preds']]
            eq_answers = [ans == lbl for ans in answers]
            sa.append(eq_answers.count(True) / len(eq_answers))

    print("Final Acc:", sum(sa) / len(sa))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)  # model path
    parser.add_argument("--data_file", type=str, default='')  # data path
    parser.add_argument("--start", type=int, default=0) # start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=1000)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    parser.add_argument("--result_file", type=str, default="./log_file.jsonl")  # tensor_parallel_size
    parser.add_argument("--temp", type=float, default=0.7) 

    return parser.parse_args()

if __name__ == "__main__":
    # Login First.
    # login(token="your_huggingface_token")

    args = parse_args()
    gsm8k_test(model=args.model, data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size, temp=args.temp)