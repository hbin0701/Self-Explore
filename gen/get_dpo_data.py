    # COLLECT DPO DATA
import json
import editdistance 
from tqdm.auto import tqdm
import jsonlines
import argparse
from utils_others import extract_answer

def collect_dpo_gsm8k(wo_dup_data_path, w_dup_data_path):
    wo_dup = [json.loads(x) for x in open(wo_dup_data_path)]
    w_dup = [json.loads(x) for x in open(w_dup_data_path)]

    query_preds_dict = {}
    dpo_data = []

    # Set Wrong Outputs
    for elem in w_dup:

        elem['output_answers'] = [extract_answer(completion) for completion in elem['preds']]
        elem['outputs'] = elem['preds']
        elem['ground_truth'] = extract_answer(elem['answer'].replace("####", "The answer is"))
        elem['input'] = elem['prompt']

        query = elem['input'].rstrip("\n")
        wrong_outputs = [out for out, ans in zip(elem['outputs'], elem['output_answers']) if str(ans) != str(elem['ground_truth'])]
        query_preds_dict[query] = list(set(wrong_outputs))

    # Find by Query:
    for elem in tqdm(wo_dup):
        query = elem['query'].rstrip("\n")
        if query not in query_preds_dict:
            raise AssertionError
        else:
            new_elem = {}
            new_elem['prompt'] = query.rstrip() + "\n"
            wrong_outputs = query_preds_dict[query]
            
            if len(wrong_outputs) == 0:
                continue # Skip

            new_elem['chosen'] = elem['response'].strip()
            new_elem['rejected'] = sorted([(ref, editdistance.eval(new_elem['chosen'], ref)) for ref in wrong_outputs], key=lambda x: x[1])[-1][0] # Choose the one with the largest edit distance.
            
            # Remove the rejected from the pool
            query_preds_dict[query].remove(new_elem['rejected'])
            dpo_data.append(new_elem)
    
    return dpo_data

def collect_dpo_math(old_li, fname=""):

    dpo_set = []

    for elem in tqdm(old_li):
        corr = elem['filtered_corr']

        incorr = list(set(elem['filtered_incorr']))  # For incorrect, one could use string-level set, or edit distance based ... i.e., list(set(elem['incorr]))

        # for every correct sample, find the one with the maximum edit distance.
        MIN_NUM = min(len(corr), len(incorr))
        
        for sample in corr[:MIN_NUM]:            
            all_rej = sorted([(ref, editdistance.eval(sample, ref) / (len(sample) + len(ref))) for ref in incorr], key=lambda x: x[1])
            
            # Try to find shortest one among maximum edit distance - this is to prevent from selecting one that has degenerated sample - i.e. repetition.
            filtered = sorted([x for x in all_rej if x[1] > 0.5], key=lambda x: len(x[0]))
            if len(filtered) == 0:
                filtered = sorted([x for x in all_rej if x[1] > 0.4], key=lambda x: len(x[0]))
                if len(filtered) == 0:
                    filtered = sorted([x for x in all_rej if x[1] > 0.3], key=lambda x: len(x[0]))
                    if len(filtered) == 0:
                        print("Not passed any :( Selecting shortest ...")
                        filtered = sorted(all_rej, key=lambda x: len(x[0]))
            
            rej = filtered[0][0]
            dpo_set.append({'prompt': elem['question'], 'chosen': sample, 'rejected': rej})
            incorr.remove(rej)

    return dpo_set