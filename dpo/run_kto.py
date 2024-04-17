import logging
import sys
import wandb
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed, LlamaTokenizer, AutoTokenizer
from accelerate import Accelerator

from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)

from peft import PeftConfig, PeftModel
# from custom_trainer import DPOTrainer, KTOTrainer, KTOConfig
from prepare_dataset import get_dataset
# from trls.trl.trainer import KTOTrainer, KTOConfig
from custom_trainer import KTOTrainer, KTOConfig
# from step_prepare_dataset import get_dataset
import os

os.environ["WANDB_API_KEY"] = "" # PUR YOUR WANDB KEY
os.environ["WANDB_ENTITY"] = "" # PUT YOUR WANDB ID
os.environ["WANDB_PROJECT"] = "" # PUT YOUR WANDB PROJECT

logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, KTOConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Increase distributed timeout to 3h to enable push to Hub to complete
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # there should be "train" and "test", with column anme "text_prompt", "text_chosen", "text_rejected".
    # raw_datasets = get_dataset(data_args.train_data_file, data_args.test_data_file)
    import json
    from datasets import Dataset

    data = json.load(open(data_args.train_data_file))

    for idx in range(len(data['label'])):
        if data['label'][idx] == False:
            rej = data['completion'][idx]
            tgt_string = "The answer is"
            if tgt_string in rej and len(rej.split("\n")) > 1:
                rej = rej[:rej.index(tgt_string)].strip()
                data['completion'][idx] = rej

    # For format of KTO Dataset, see: https://huggingface.co/docs/trl/main/en/kto_trainer
    train_set = Dataset.from_dict({k: v for k, v in data.items()}) 
    train_set = train_set.shuffle(seed=42)
    test_set = Dataset.from_dict({k: v[:100] for k, v in data.items()})
    
    model_args.torch_dtype = "bfloat16"
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    print("torch_dtype", torch_dtype)

    model_kwargs = dict(
       revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=True,
        torch_dtype=torch_dtype,
        use_cache=False,
        device_map=get_kbit_device_map(),
        quantization_config=get_quantization_config(model_args),
    )

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    ref_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # if model_args.use_peft is True:
    #     ref_model = None
    #     ref_model_kwargs = None

    #########################
    # Instantiate DPO trainer
    #########################
    # training_args.model_init_kwargs = model_kwargs
    # training_args.ref_model_init_kwargs = ref_model_kwargs

    kto_trainer = KTOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset= train_set,
        eval_dataset=test_set,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    train_result = kto_trainer.train()
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_set)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_set))
    kto_trainer.log_metrics("train", metrics)
    kto_trainer.save_metrics("train", metrics)
    kto_trainer.save_state()

    logger.info("*** Training complete ***")

    # Evaluate
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = kto_trainer.evaluate()
    #     max_eval_samples = (
    #         data_args.max_eval_samples if data_args.max_eval_samples is not None else len(test_set)
        # )
        # metrics["eval_samples"] = min(max_eval_samples, len(kto_trainer["test"]))
        # kto_trainer.log_metrics("eval", metrics)
        # kto_trainer.save_metrics("eval", metrics)

    # Save model and create model card
    kto_trainer.save_model(training_args.output_dir)

    # Save everything else on main process
    if accelerator.is_main_process:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "dataset": list(data_args.dataset_mixer.keys()),
            "dataset_tags": list(data_args.dataset_mixer.keys()),
            "tags": ["alignment-handbook"],
        }
        kto_trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        kto_trainer.model.config.use_cache = True
        kto_trainer.model.config.save_pretrained(training_args.output_dir)
        if training_args.push_to_hub is True:
            kto_trainer.push_to_hub()

    # Ensure we don't timeout on model save / push to Hub
    logger.info("*** Waiting for all processes to finish ***")
    accelerator.wait_for_everyone()

    logger.info("*** Run complete! ***")


if __name__ == "__main__":
    main()
