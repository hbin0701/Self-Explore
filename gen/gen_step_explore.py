import optparse
import sys
from math_utils.eval_script import eval_math, is_correct
from math_utils.answer_extraction import extract_answer, extract_math_few_shot_cot_answer
        
from huggingface_hub import login
import argparse
import json
import jsonlines
from vllm import LLM, SamplingParams
import sys
from utils_others import get_final_steps
from tqdm.auto import tqdm

MAX_INT = sys.maxsize

def extract_from_pred(x, y, z):
    try:
        return extract_math_few_shot_cot_answer(x,y,z)[0]
    except:
        return "[Invalid]"

def test(model, data_path, tensor_parallel_size=1, temp=0.7, id=0, k=4, task="GSM8K", result_file=""):
    stop_tokens = []

    if temp == 0.7:
        n = k # Change this if you want larger 'k' for exploration.
    else:
        n = 1

    li = [json.loads(x) for x in open(data_path)]

    # Get the rejected sample divided in steps.
    if 'final_steps' not in li[0].keys():
        li = get_final_steps(li, task)

    li = [x for x in li if x['rejected'].strip("\n").strip() != ""] # filter out here.

    # This is based on the assumption there are 4 GPUs.
    portion = (len(li) + 4) // 4
    li = li[int(portion * id): int(portion * (id + 1))]

    all_elems = []

    result_file = args.result_file.replace(".jsonl", f"_{id}.jsonl")
    ## Exclude already processed.
    try:
        already_processed = [json.loads(x) for x in open(result_file)]

        if len(already_processed) >= len(li) - 1:
            return
        else:
            li = li[len(already_processed):]
    except:
        already_processed = []

    max_token_num = 512 if task == "GSM8K" else 1024 
    sampling_params = SamplingParams(temperature=temp, top_p=1, max_tokens=max_token_num, stop=stop_tokens, n=n)

    print('sampling =====', sampling_params)
    print("Length Remaining ...", len(li))
    llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size, enforce_eager=True, swap_space=64)

    ## PREPARE SAMPLES
    for idx, elem in enumerate(li):
        elem['step_pred'] = {}
        elem['all_prompt'] =  elem['final_steps']
        elem['completed'] = False
        elem['num_steps'] = len(elem['all_prompt'])
        elem['sample_idx'] = idx

        # Get answer.
        if task == "GSM8K":
            elem['answer'] = elem['chosen'][elem['chosen'].index("The answer is") + len("The answer is"):].strip()
        else:
            elem['answer'] = extract_math_few_shot_cot_answer(elem['prompt'], elem['chosen'], "")[0]

        all_elems.append(elem)

    print(f"***** [Length Remaining] {len(all_elems)} *****")

    # Samples to process: 100
    SAMPLES_NUM = 1000

    for i in tqdm(range(0, len(all_elems) + SAMPLES_NUM, SAMPLES_NUM)):
        current_samples = all_elems[i:i+SAMPLES_NUM]
        max_step = min(max([len(x['all_prompt']) for x in current_samples]), 20) # set 20 as hard limit.
        
        for step_idx in range(max_step):
            # each samples' step_idx-th step.
            print("step_idx", step_idx)
            curr_step_to_proc = [x['all_prompt'][step_idx] for x in current_samples if x['completed'] == False]
            sample_idxs = [x['sample_idx'] for x in current_samples if x['completed'] == False]

            completions = llm.generate(curr_step_to_proc, sampling_params)

            # check if reached answer, if not discard.
            for sample_idx, output in zip(sample_idxs, completions):
                all_texts = [out.text for out in output.outputs]

                if task == "GSM8K":
                    def get_answer(text):
                        try:
                            ans = text[text.index("The answer is") + len("The answer is"):].strip()
                        except:
                            ans = "[invalid]"
                        return ans

                    all_answers = [get_answer(text) for text in all_texts]
                else:
                    all_answers = [extract_from_pred(all_elems[sample_idx]['prompt'], text, "") for text in all_texts]

                # If all false ... no need to continue. (i.e. we found the first pit!)
                if task == "GSM8K":
                    if True not in [str(pred) == str(all_elems[sample_idx]['answer']) for pred in all_answers]:
                        all_elems[sample_idx]['completed'] = True
                else:
                    if True not in [eval_math({"prediction": pred, "answer": all_elems[sample_idx]['answer']}) for pred in all_answers]:
                        all_elems[sample_idx]['completed'] = True

                # If processed all steps, mark as completed.
                if step_idx + 1 >= current_samples[sample_idx]['num_steps']:
                    all_elems[sample_idx]['completed'] = True

                all_elems[sample_idx]["step_pred"][step_idx + 1] =  {"prompt": all_elems[sample_idx]['all_prompt'][step_idx], "preds": all_texts, "answers": all_answers}

        with jsonlines.open(result_file, "a") as writer:
            writer.write_all(current_samples)

        # return to prevent server crash.
        return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)  # model path
    parser.add_argument("--data_file", type=str, default='')  # data path
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    parser.add_argument("--result_file", type=str, default="./log_file.jsonl")  # tensor_parallel_size
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--k", type=int, default=4) # how many completions to generate per step.
    parser.add_argument("--task", type=str, default="GSM8K") # how many completions to generate per step.

    return parser.parse_args()

if __name__ == "__main__":
    # Login First.
    # login(token="your_hugging_face_token_here")
    args = parse_args()
    test(model=args.model, data_path=args.data_file, tensor_parallel_size=args.tensor_parallel_size, temp=args.temp, id=args.id, k=args.k, task=args.task, result_file=args.result_file)

    # See if completed.
    import os
    file_lists = [args.result_file.replace(".jsonl", f"_{i}.jsonl") for i in range(4)]

    # IF completed:
    if all([os.path.exists(x) for x in file_lists]) and sum([len([json.loads(x) for x in open(file_name)]) for file_name in file_lists]) == len([json.loads(x) for x in open(args.data_file)]):
        from utils_others import unite_file
        unite_file(args.result_file, 4)

        # Then run form_gpair
        from utils_others import form_gpair
        form_gpair(args.result_file, args.result_file.replace(".jsonl", "_gpair_4.jsonl"), args.task)