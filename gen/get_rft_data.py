# GET RFT DATA
import re
import editdistance
from tqdm.auto import tqdm
import json
import jsonlines
import argparse
from utils_others import extract_answer
from get_dpo_data import collect_dpo_gsm8k, collect_dpo_math
import random
import editdistance

def read_jsonl(fname):  
    return [json.loads(line) for line in open(fname)]

def check_equation(eq):
    if eq.find('=') == -1:
        return False
    
    lhs = eq.split('=')[0]
    rhs = eq.split('=')[1]
    
    try:
        lhs_result = eval(str(lhs))
        if abs(float(lhs_result) - float(rhs)) < 1e-3:
            return True
    except BaseException:
        return False
    return False

def is_eq_correct(equations):
    for eq in equations:
        if not check_equation(eq):
            return False
    return True

def collect_rft_data_gsm8k(fname):

    eq_pattern = r'<<([^>]*)>>'  # Match everything inside << >>
    # eq_pattern = r'\b\d+\.?\d*\b' # This for mistral-metamath.

    all_sets = [] # list of dicts of 'query' and 'response'.
    old_li = read_jsonl(fname)

    for sample in tqdm(old_li):
        sample['preds'] = list(set(sample['preds']))
        sample['output_answers'] = [extract_answer(completion) for completion in sample['preds']]
        sample['outputs'] = [x.strip() for x in sample['preds']]
        sample['ground_truth'] = extract_answer(sample['answer'].replace("####", "The answer is"))
        sample['input'] = sample['prompt'].rstrip() + "\n"

        exist_match = []
        correct_preds = [reasoning for reasoning, answer in zip(sample['outputs'], sample['output_answers']) if str(answer) == str(sample['ground_truth'])]

        matches = [{"reasoning": r, "equations": re.findall(eq_pattern, r)} for r in correct_preds] # Find all matches
        
        # remove this line in case of mistral-metamath.
        matches = [m for m in matches if is_eq_correct(m['equations'])]
                    
        final_preds = {}

        for elem in matches:
            match_string = '|'.join(elem['equations']).replace(' ', '')
            if match_string in final_preds:
                other_solutions = [final_preds[k] for k in final_preds if k != match_string]
                now_score = sum([editdistance.eval(elem['reasoning'], ref) for ref in other_solutions])
                original_score = sum([editdistance.eval(final_preds[match_string], ref) for ref in other_solutions])
                if now_score > original_score:
                    final_preds[match_string] = elem['reasoning']
            else:
                final_preds[match_string] = elem['reasoning']

        sample['rft_outputs'] = list(final_preds.values())
        
        MAX_NUM = 8
        for rft_sample in sample['rft_outputs'][:MAX_NUM]:
            all_sets.append({'query': sample['input'].rstrip("\n"), 'response': rft_sample})

    return old_li, all_sets

def collect_rft_data_math(fname):
    from math_utils.eval_script import eval_math, is_correct
    from math_utils.answer_extraction import extract_answer, extract_math_few_shot_cot_answer
        
    random.seed(42)
    MAX_SAMPLES = 8

    def remove_dup(corr, N=100):
        corr = list(set(corr))
        random.shuffle(corr)
        final_results = []
        
        for elem in corr:
            if all([editdistance.eval(elem, x) / (len(elem) + len(x) // 2) > 0.2  for x in final_results]):
                final_results.append(elem)
            if len(final_results) >= N:
                break
        return final_results

    # file should be already aggregated.
    li = [json.loads(x) for x in open(fname)]

    # Make sure you check it by your own.
    # if len(li) != 7500 or len(set([x['prompt'] for x in li])) != 7500:
    #     raise ValueError("RFT is missing some file. Make sure all questions are properly handled.")
    
    # [To-do] Change data_file dir.
    all_q = [json.loads(x) for x in open("../data/MATH_train.jsonl")] 

    for orig_elem, elem in zip(all_q, li):
        elem['answer'] = extract_answer(elem['answer'])
        elem['question'] = orig_elem['query']

    new_li = []

    for idx in tqdm(range(len(li))):

        li[idx]['incorr'] = []
        li[idx]['corr'] = []
        q = li[idx]['question']

        for pred in li[idx]['preds']:
            # Only select those that "Final Answer:" is present, to prevent repetition being used as negative.
            # We want a good quality negative. :)
            tgt_str = "Final Answer:"
            tgt_str = "The answer is"

            if tgt_str not in pred.strip().split("\n")[-1]:
                if tgt_str in pred:
                    pred_idx = pred.index(tgt_str)
                    end_of_line_idx = pred.find('\n', pred_idx + len(tgt_str))
                    if end_of_line_idx != -1:
                        pred = pred[:end_of_line_idx]
                    else:
                        print("Final Answer Error.")
                        continue # just skip.
                else:
                    continue # If not found, likely to be repetition, or incomplete solution.

            try:
                new_x = {"prediction": extract_math_few_shot_cot_answer(q, pred, "")[0], "answer": li[idx]['answer']}
                out = eval_math(new_x)
            except:
                out = False

            if out:
                li[idx]['corr'].append(pred)
            else:
                li[idx]['incorr'].append(pred)
        
        filtered_corr = remove_dup(li[idx]['corr'], N=MAX_SAMPLES)
        filtered_corr = [x.replace("I hope it is correct.", "" ).strip() for x in filtered_corr]
        li[idx]['filtered_corr'] = filtered_corr

        if len(filtered_corr) == 0:
            li[idx]['filtered_incorr'] = []
            continue

        filtered_incorr = remove_dup(li[idx]['incorr'], N=MAX_SAMPLES * 3) # for simplicity.
        filtered_incorr = [x.replace("I hope it is correct.", "" ).strip() for x in filtered_incorr]
        li[idx]['filtered_incorr'] = filtered_incorr

        for filtered_corr_sample in filtered_corr:
            new_li.append({"query": q.rstrip() + "\n", "response": filtered_corr_sample})

    return li, new_li

def run_rft_and_dpo(fname, task):

    fname1 = fname
    fname2 = fname1.replace("gen", "rft")
    fname3 = fname1.replace("gen", "dpo")

    collect_rft_fn = collect_rft_data_gsm8k if task == "GSM8K" else collect_rft_data_math
    collect_dpo_fn = collect_dpo_gsm8k if task == "GSM8K" else collect_dpo_math

    old_set, new_set = collect_rft_fn(fname1)

    # Generate RFT file
    with jsonlines.open(fname2, mode='w') as writer:
        writer.write_all(new_set)

    if task == "GSM8K":
        new_set = collect_dpo_fn(fname2, fname1)
    else:
        new_set = collect_dpo_fn(old_set, fname1)
    
    # Generate DPO file
    with jsonlines.open(fname3, mode='w') as writer:
        writer.write_all(new_set)