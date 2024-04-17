### From OVM/utils/gsm8k/decoding.py

from contextlib import contextmanager
import signal
import json
import os
import re
import random
import jsonlines
from tqdm.auto import tqdm

# ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
ANS_RE = re.compile(r"The answer is:?\s*(\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        st_str = standardize_value_str(match_str)   
        try: eval(st_str); return st_str
        except: ...
    return INVALID_ANS

def extract_answers(completions):
    return [extract_answer(completion) for completion in completions]

def standardize_value_str(x):
    """Standardize numerical values"""
    y = x.replace(",", "")
    if '.' in y:
        y = y.rstrip('0')
        if y[-1] == '.':
            y = y[:-1]
    if not len(y):
        return INVALID_ANS
    if y[0] == '.':
        y = '0' + y
    if y[-1] == '%':
        y = str(eval(y[:-1]) / 100)
    return y.rstrip('.')

def get_answer_label(response_answer, gt):
    if response_answer == INVALID_ANS:
        return INVALID_ANS
    return response_answer == gt

# taken from
# https://stackoverflow.com/questions/492519/timeout-on-a-function-call
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            return round(eval(formula), ndigits=4)
    except Exception as e:
        signal.alarm(0)
        print(f"Warning: Failed to eval {formula}, exception: {e}")
        return None

# refer to https://github.com/openai/grade-school-math/blob/master/grade_school_math/calculator.py
def use_calculator(sample):
    if "<<" not in sample:
        return None

    parts = sample.split("<<")
    remaining = parts[-1]
    if ">>" in remaining:
        return None
    if "=" not in remaining:
        return None
    lhs = remaining.split("=")[0]
    lhs = lhs.replace(",", "")
    if any([x not in "0123456789*+-/.()" for x in lhs]):
        return None
    ans = eval_with_timeout(lhs)
    if remaining[-1] == '-' and ans is not None and ans < 0:
        ans = -ans
    return ans

# FINAL STEPS

import re

def divide_string_optimally(s):
    # Define the points where we want to split
    split_points = [",$", ", which", " so", " = ", " and", "Thus", "="]
    positions = []

    # Find positions of each split point in the string
    for point in split_points:
        pos = s.find(point)
        while pos != -1:
            positions.append(pos + len(point))  # Add position to split AFTER the point
            pos = s.find(point, pos + 1)

    # Sort the positions to process them in order
    positions.sort()

    # Find the optimal division point
    min_diff = float('inf')
    optimal_pos = None
    optim_first_pos = -1
    optim_second_pos = -1

    for pos in positions:
        first_part_length = pos
        second_part_length = len(s) - pos
        diff = abs(first_part_length - second_part_length)

        if diff < min_diff:
            min_diff = diff
            optimal_pos = pos
            optim_first_pos = first_part_length
            optim_second_pos = second_part_length
    
    # Divide the string at the optimal position
    if optim_first_pos > 20 and optim_second_pos > 20 and optimal_pos is not None:
        return [s[:optimal_pos], s[optimal_pos:]]
    else:
        # Return the original string and an empty string if no division point was found
        return [s]

def process_further(x):
    final = []
    for elem in x:
        if len(elem) < 20:
            final.append(elem)
        else:
            final.extend(divide_string_optimally(elem))
    return final


import re

# Define the modified function as per the user's instructions
def split_string(input_string):
    # Split the input string by "\n" or "\n\n", keeping the delimiter
    split_pattern = r'(?:\n\n|\n)'
    parts_with_delimiters = [e for e in re.split(f'({split_pattern})', input_string) if e]

    # Apply the given regex pattern to each part
    pattern = r'(?:.*?(?:\n+|\.[$]\s+|[$]\.\s+))'
    resulting_parts = []

    for part in parts_with_delimiters:
        # If the part is a delimiter, add it directly to the results
        if re.match(split_pattern, part):
            resulting_parts.append(part)
        else:
            # Apply the regex to non-delimiter parts
            matched_parts = re.findall(pattern, part, re.DOTALL)
            
            # Check if the last part of the string is captured; if not, add it
            last_part = part[len(''.join(matched_parts)):]
            if last_part:
                matched_parts.append(last_part)
            
            # Extend the resulting_parts with matched_parts
            resulting_parts.extend(matched_parts)

    return resulting_parts


def merge_max(parts):
    while len(parts) > 9:
        shortest_index = min(range(len(parts)), key=lambda i: len(parts[i]))
        if shortest_index > 0 and (shortest_index == len(parts) - 1 or len(parts[shortest_index - 1]) < len(parts[shortest_index + 1])):
            parts[shortest_index - 1] += parts.pop(shortest_index)
        else:
            parts[shortest_index] += parts.pop(shortest_index + 1)
            
    return parts

def merge_short_elements(parts):
    
    final_parts = []
    for part in parts:
        if len(final_parts) != 0 and (part.strip() == "" or len(part) < 20):
            final_parts[-1] += part
        else: 
            final_parts.append(part)

    parts = final_parts

    average_length = sum(len(part) for part in parts) / len(parts)
    short_threshold = 0.3 * average_length
    i = 0
    while i < len(parts):
        if len(parts[i]) < short_threshold:
            if i > 0 and (i == len(parts) - 1 or len(parts[i - 1]) < len(parts[i + 1])):
                parts[i - 1] += parts.pop(i)
                i -= 1  # Adjust index since the current element is merged with the previous
            else:
                parts[i] += parts.pop(i + 1)
        else:
            i += 1  # Move to the next element if the current one isn't short or after merging
    
    return parts

# Apply the function to initially split and then merge short elements first
diff = []

def get_final_steps(li, task):
    if task == "MATH":
        for elem in tqdm(li):
            rej = elem['rejected']
            steps = split_string(rej)
            
            elem['steps'] = steps
            merged_steps = merge_short_elements(steps)
            final_steps = merge_max(merged_steps)

            last_elem = final_steps[-1]
            if "The answer is" in last_elem  and last_elem.index("The answer is") != 0:
                final_steps[-1] = last_elem[:last_elem.index("The answer is")]

                try:
                    if len(final_steps[-1]) < 20:
                        final_steps[-2] += final_steps[-1]
                        final_steps[-1] = last_elem[last_elem.index("The answer is"):]
                    else:            
                        final_steps.append(last_elem[last_elem.index("The answer is"):])
                except:
                    final_steps.append(last_elem[last_elem.index("The answer is"):])

            elem['pre_merged_steps'] = final_steps
            
            if len(final_steps) <= 3:
                final_steps = process_further(final_steps)

            elem['merged_steps'] = final_steps
            elem['final_steps'] = [elem['prompt'] + "\n" + ''.join(elem['merged_steps'][:i+1]) for i in range(len(elem['merged_steps']))]
    
    elif task == "GSM8K":
        for elem in li:
            elem['merged_steps'] = [x.strip() + "\n" for x in elem['rejected'].split("\n") if x.strip()]
            elem['final_steps'] = [elem['prompt'].strip() + "\n" + ''.join(elem['merged_steps'][:i+1]) for i in range(len(elem['merged_steps']))]
    
    return li

def unite_file(file_name, k, suffix=None):
    new_fnames = [file_name.replace(".jsonl", f"_{i}.jsonl") for i in range(k)]
    new_files = []
    
    for x in new_fnames:
        new_files.extend([json.loads(line) for line in open(x)])

    if suffix is not None:
        file_name = file_name.replace(".jsonl", f"{suffix}.jsonl")
    
    with jsonlines.open(file_name, 'w') as f:
        f.write_all(new_files)

random.seed(42)

from math_utils.eval_script import eval_math, is_correct
from math_utils.answer_extraction import extract_answer as extract_answer_math 

def form_gpair(data_file, result_file, task="GSM8K"):

    li = [json.loads(x) for x in open(data_file)]
    N = 4

    final_list = []
    query_dict = {}

    for idx, elem in tqdm(enumerate(li)):
        if task == "GSM8K":
            gt_ans = extract_answer(elem['chosen'])
        else:
            gt_ans = extract_answer_math(elem['chosen'])
            
        rejected = elem['rejected']
        prompt = elem['prompt'].rstrip("\n") + "\n"

        # Evaluate each step
        steps = elem['step_pred']
        step_corr = []

        first_wrong_step = -1
        for step_idx in steps:

            corr_answers = [x for x in steps[step_idx]['answers'][:N] if str(x) == str(gt_ans)]

            if len(corr_answers) >= 1:
                continue
            else:
                first_wrong_step = step_idx
                break

        if int(first_wrong_step) == -1:
            # if all correct ... continue - because rejected is not 'actually' rejected from the explorer's viewpoint.
            continue
        elif int(first_wrong_step) == 1:
            orig_problem = elem['prompt']

            if orig_problem[-1] == '\n':
                
                # This will eventually be taken care of in data processing when running DPO, however.
                if task == "GSM8K":
                    orig_problem = orig_problem.rstrip("\n") + "\n"
                elif task == "MATH":
                    orig_problem = orig_problem.rstrip("\n") + "\n\n"

            new_rej = elem['step_pred']['1']['prompt'][len(orig_problem):].lstrip("\n")
            final_list.append({"prompt": orig_problem, "rejected": new_rej, "chosen": elem['chosen']})
        else:
            prompt = elem['prompt']

            if prompt not in query_dict:
                query_dict[prompt] = []

            chosen_step = steps[str(int(first_wrong_step)-1)]
            this_step_prompt = chosen_step['prompt']
            rejected = elem['step_pred'][first_wrong_step]['prompt'][len(this_step_prompt):]

            # Chosen might overlap.
            chosen = sorted([x for x, y in zip(chosen_step['preds'][:N], chosen_step['answers'][:N]) if str(y) == str(gt_ans)], key=lambda x: len(x)) # choose the short one.
            chosen = [x for x in chosen if x not in query_dict[elem['prompt']]]
            chosen = [x for x in chosen if rejected.strip() not in chosen]

            if chosen == []:
                continue
            else:
                chosen = chosen[0]
                
            query_dict[elem['prompt']].append(chosen)

            if rejected[0] == '\n' and chosen[0] == '\n':
                rejected = rejected.lstrip("\n")
                chosen = chosen.lstrip("\n")
                this_step_prompt += "\n"

            dict_ = {"prompt": this_step_prompt, "rejected": rejected, "chosen": chosen}
            final_list.append(dict_)

    query_dict = {}

    for x in final_list:
        prompt = x['prompt'].strip().split("\n")[0]
        if prompt not in query_dict:
            query_dict[prompt] = []

        query_dict[prompt].append(x)

    unique_list = [] 

    for k, v in tqdm(query_dict.items()):
        unique_chosen = set()
        unique_rejected = set()

        for c in v:
            ch = c['chosen']
            rj = c['rejected']
            if ch not in unique_chosen and rj not in unique_rejected:
                unique_chosen.add(ch)
                unique_rejected.add(rj)
                unique_list.append(c)


    with jsonlines.open(result_file, mode="w") as writer:
        writer.write_all(unique_list)
