export WANDB_API_KEY=wandb_key # put your wandb key here
export WANDB_PROJECT=project_name # put your project name here
export WANDB_ENTITY=my_id # put your wandb id here.

model_name_or_path=/home/user/models/deepseek-math-7b-base # model path to train on.
save_generator_id=deepseek_GSM8K_FT # model name to be saved.

save_dir=/home/user/models/${save_generator_id}/
export WANDB_NAME=${save_generator_id}    

# lr: 1e-6 for Mistral and 1e-5 for Others.

accelerate launch \
  --config_file scripts/gsm8k/sft/config.yaml \
  --main_process_port=40999 \
  sft/train_generator.py \
  --model_name_or_path ${model_name_or_path} \
  --data_dir put_your_data_file_here \
  --target_set train \
  --save_dir ${save_dir} \
  --num_train_epoches 5 \
  --save_strategy epoch \
  --max_length 384 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing True \
  --learning_rate 1e-5 \
  --weight_decay 0 \
  --lr_scheduler_type "linear" \
  --warmup_steps 0 \
  --save_best False \
  --save_total_limit 5 \
  --logging_dir ./wandb \
  --logging_steps 8 \
  --seed 42 \
  --save_model_only True \
  --mode "ft_GSM8K" # mode is one of "ft_GSM8K", "ft_MATH", "rft_GSM8K", "rft_MATH"