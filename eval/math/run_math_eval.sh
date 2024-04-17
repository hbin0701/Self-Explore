#!/bin/bash

output_dir="outputs/DeepSeek_MATH_Self_Explore"
model_path=""
tokenizer_path=""
model_size="7b"
use_vllm="--use-vllm"
no_markup_question="--no-markup-question"
test_conf="configs/few_shot_test_configs.json" # While we use this config, our code doesn't evaluate in few-shot setting.
prompt_format="few_shot"
n_repeats="1"
temperature="0"
ngpus="4"
rank="0"

python run_subset_parallel.py --output-dir $output_dir \
--model-path $model_path \
--tokenizer-path $tokenizer_path \
--model-size $model_size \
$use_vllm \
$no_markup_question \
--test-conf $test_conf \
--prompt_format $prompt_format \
--n-repeats $n_repeats \
--temperature $temperature \
--ngpus $ngpus \
--rank $rank
