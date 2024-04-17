# Assuming you hvae 4 gpus, run as following to speed up the process.
model_path="put_your_model_path" # Put the model path.
result_file="put_your_result_file" # Put the name of file to save exploration result.
data_file="put_the_dpo_data_here" # Here, put the DPO file generated.
temp=0.7
task="GSM8K" # or MATH

# Change the number of iterations accordingly to the data_file.
for j in $(seq 1 10); do
      CUDA_VISIBLED_DEVICES=0 python gen_step_explore.py --task $task --model $model_path --temp $temp --result_file $result_file --data_file $data_file --id 0 &
      CUDA_VISIBLED_DEVICES=1 python gen_step_explore.py --task $task --model $model_path --temp $temp --result_file $result_file --data_file $data_file --id 1 &
      CUDA_VISIBLED_DEVICES=2 python gen_step_explore.py --task $task --model $model_path --temp $temp --result_file $result_file --data_file $data_file --id 2 &
      CUDA_VISIBLED_DEVICES=3 python gen_step_explore.py --task $task --model $model_path --temp $temp --result_file $result_file --data_file $data_file --id 3 &
   wait
done