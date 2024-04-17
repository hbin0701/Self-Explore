#!/bin/bash
model_path="put_your_FT_model_path" # Put the model path.
result_file="put_your_result_file" # Put the result file path you wish to save the result.
data_file="put_your_data_file" # Put the train jsonl file.
task="GSM8K"

# First set of processes with deepseek-ai/deepseek-math-7b-base
for i in $(seq 0 3) # Loop from 0 to 6, stepping by 2
   do
      CUDA_VISIBLE_DEVICES=$i python gen_rft_data.py --task $task --model $model_path --data_file $data_file --result_file $result_file --temp 0.7 --id $i &
   done
wait