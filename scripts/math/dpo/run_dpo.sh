# Run form main directory.
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file scripts/math/dpo/deepspeed_zero3.yaml dpo/run_dpo.py scripts/math/dpo/config.yaml