# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import random
import warnings
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from functools import wraps
from operator import itemgetter
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import math
from datasets import Dataset, concatenate_datasets, interleave_datasets

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_deepspeed_available, tqdm
from accelerate import PartialState
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput, has_length

from trl.import_utils import is_peft_available, is_wandb_available
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.utils import DPODataCollatorWithPadding, disable_dropout_in_model, pad_to_length, trl_sanitze_kwargs_for_tagging, peft_module_casting_to_bf16

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Dict, Optional

from transformers import TrainingArguments



@dataclass
class KTOConfig(TrainingArguments):
    r"""
    KTOConfig collects all training arguments related to the [`KTOTrainer`] class.

    Using [`HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_length (`int`, *optional*, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, *optional*, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_completion_length (`int`, *optional*, defaults to `None`):
            The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        beta (`float`, defaults to 0.1):
            The beta factor in KTO loss. Higher beta means less divergence from the initial policy.
        desirable_weight (`float`, *optional*, defaults to 1.0):
            The desirable losses are weighed by this factor to counter unequal number of desirable and undesirable paris.
        undesirable_weight (`float`, *optional*, defaults to 1.0):
            The undesirable losses are weighed by this factor to counter unequal number of desirable and undesirable pairs.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value if it is different to the tokenizer's pad_token_id.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        generate_during_eval (`bool`, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        precompute_ref_log_probs (`bool`, defaults to `False`):
            Flag to precompute reference model log probabilities for training and evaluation datasets. This is useful if you want to train
            without the reference model and reduce the total GPU memory needed.
        model_init_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string.
        ref_model_init_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the ref model from a string.
        dataset_num_proc: (`Optional[int]`, *optional*, defaults to `None`):
            Number of processes to use for processing the datasets.
    """

    max_length: Optional[int] = None
    """The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator."""
    max_prompt_length: Optional[int] = None
    """The maximum length of the prompt. This argument is required if you want to use the default data collator."""
    max_completion_length: Optional[int] = None
    """The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder."""
    beta: float = 0.1
    """The beta factor in KTO loss. Higher beta means less divergence from the initial policy."""
    desirable_weight: Optional[float] = 1.0
    """The desirable losses are weighed by this factor."""
    undesirable_weight: Optional[float] = 1.0
    """The undesirable losses are weighed by this factor."""

    label_pad_token_id: int = -100
    padding_value: int = None
    truncation_mode: str = "keep_end"
    generate_during_eval: bool = False
    is_encoder_decoder: Optional[bool] = None
    precompute_ref_log_probs: bool = False
    model_init_kwargs: Optional[Dict] = None
    ref_model_init_kwargs: Optional[Dict] = None
    dataset_num_proc: Optional[int] = None

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed

def _get_kl_dataset(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Creates mismatched pairs of prompts and completions for the KL dataset by reversing the order of completions."""
    batch["answer_input_ids"] = batch["answer_input_ids"][::-1]
    batch["answer_attention_mask"] = batch["answer_attention_mask"][::-1]
    return batch


def _tokenize(batch: Dict[str, List[Any]], tokenizer: "PreTrainedTokenizer") -> Dict[str, List[Any]]:
    """Tokenize a batch from a KTO specific dataset."""
    prompt_tokenized = tokenizer(batch["prompt"], add_special_tokens=False)
    prompt_input_ids = prompt_tokenized["input_ids"]
    prompt_attention_mask = prompt_tokenized["attention_mask"]

    prompt_and_completion = [prompt + completion for prompt, completion in zip(batch["prompt"], batch["completion"])]
    full_tokenized = tokenizer(prompt_and_completion, add_special_tokens=False)
    full_input_ids = full_tokenized["input_ids"]
    full_attention_mask = full_tokenized["attention_mask"]

    answer_input_ids = [f[len(p) :] for f, p in zip(full_input_ids, prompt_input_ids)]
    answer_attention_mask = [f[len(p) :] for f, p in zip(full_attention_mask, prompt_attention_mask)]

    # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
    full_concat_input_ids = [np.concatenate([p, a]) for p, a in zip(prompt_input_ids, answer_input_ids)]
    # Prepare input tokens for token by token comparison
    full_input_ids = [np.array(f) for f in full_input_ids]
    for full, concat in zip(full_input_ids, full_concat_input_ids):
        if len(full) != len(concat):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

    # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
    # can be merged together when tokenizing prompt+answer. This could result
    # on the last token from the prompt being different when tokenized on its own
    # vs when done as prompt+answer.
    response_token_ids_start_idx = [len(p) for p in prompt_input_ids]

    # If tokenized prompt is different than both prompt+answer, then it means the
    # last token has changed due to merging.
    for idx, (p, f, r) in enumerate(zip(prompt_input_ids, full_input_ids, response_token_ids_start_idx)):
        if not np.array_equal(p, f[:r]):
            response_token_ids_start_idx[idx] -= 1

    prompt_input_ids = [f[:r] for f, r in zip(full_input_ids, response_token_ids_start_idx)]
    prompt_attention_mask = [f[:r] for f, r in zip(full_attention_mask, response_token_ids_start_idx)]

    for p, m in zip(prompt_input_ids, prompt_attention_mask):
        if len(p) != len(m):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

    answer_input_ids = [f[r:] for f, r in zip(full_input_ids, response_token_ids_start_idx)]
    answer_attention_mask = [f[r:] for f, r in zip(full_attention_mask, response_token_ids_start_idx)]

    return dict(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        answer_input_ids=answer_input_ids,
        answer_attention_mask=answer_attention_mask,
    )


def _process_tokens(example: Dict[str, Any], model: "PreTrainedModel" = None, **kwargs) -> Dict:
    """Process tokens of a KTO specific dataset.

    At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
    in case the prompt + completion responses is/are too long. First
        we truncate the prompt; if we're still too long, we truncate the completion.

    We also create the labels for the completion responses, which are of length equal to
        the sum of the length of the prompt and the completion response, with
        label_pad_token_id  for the prompt tokens.
    """
    prompt = example["prompt"]
    completion = example["completion"]

    batch = {
        f"{kwargs['prefix']}prompt": prompt,
        f"{kwargs['prefix']}completion": completion,
        f"{kwargs['prefix']}label": example["label"],
    }

    if not kwargs["is_encoder_decoder"]:
        # Check issues below for more details
        #  1. https://github.com/huggingface/trl/issues/907
        #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        #  3. https://github.com/LianjiaTech/BELLE/issues/337

        if not isinstance(prompt, str):
            raise ValueError(f"prompt should be an str but got {type(prompt)}")

        if not isinstance(completion, str):
            raise ValueError(f"completion should be an str but got {type(completion)}")

        # keys of format prompt_* refers to just the prompt and answer_* refers to just the answer
        all_tokens = {
            "prompt_input_ids": example["prompt_input_ids"],
            "prompt_attention_mask": example["prompt_attention_mask"],
            "answer_input_ids": example["answer_input_ids"],
            "answer_attention_mask": example["answer_attention_mask"],
        }

        max_length = kwargs["max_length"] - 2
        # if combined sequence is too long (> max_length - 1 for BOS token - 1 for EOS), truncate the prompt
        if len(all_tokens["prompt_input_ids"]) + len(all_tokens["answer_input_ids"]) > max_length:
            for k in ["prompt_input_ids", "prompt_attention_mask"]:
                if kwargs["truncation_mode"] == "keep_start":
                    all_tokens[k] = all_tokens[k][: kwargs["max_prompt_length"]]
                elif kwargs["truncation_mode"] == "keep_end":
                    all_tokens[k] = all_tokens[k][-kwargs["max_prompt_length"] :]
                else:
                    raise ValueError(f"Unknown truncation mode: {kwargs['truncation_mode']}")

        # if that's still too long, truncate the response
        if len(all_tokens["prompt_input_ids"]) + len(all_tokens["answer_input_ids"]) > max_length:
            for k in ["answer_input_ids", "answer_attention_mask"]:
                all_tokens[k] = all_tokens[k][: max_length - kwargs["max_prompt_length"]]

        # for legacy reasons, use the completion_* prefix to now refer to the joint sequence
        # HERe
        batch[f"{kwargs['prefix']}prompt_input_ids"] = [kwargs["tokenizer"].bos_token_id] + all_tokens[
            "prompt_input_ids"
        ]
        batch[f"{kwargs['prefix']}prompt_attention_mask"] = [1] + all_tokens["prompt_attention_mask"]

        if example["label"] == True:
            batch[f"{kwargs['prefix']}completion_input_ids"] = (
                [kwargs["tokenizer"].bos_token_id]
                + all_tokens["prompt_input_ids"]
                + all_tokens["answer_input_ids"]
                + [kwargs["tokenizer"].eos_token_id]
            )
            batch[f"{kwargs['prefix']}completion_attention_mask"] = (
                [1] + all_tokens["prompt_attention_mask"] + all_tokens["answer_attention_mask"] + [1]
            )
        else:
            batch[f"{kwargs['prefix']}completion_input_ids"] = (
                [kwargs["tokenizer"].bos_token_id]
                + all_tokens["prompt_input_ids"]
                + all_tokens["answer_input_ids"]
            )
            batch[f"{kwargs['prefix']}completion_attention_mask"] = (
                [1] + all_tokens["prompt_attention_mask"] + all_tokens["answer_attention_mask"]
            )

        batch[f"{kwargs['prefix']}completion_labels"] = batch[f"{kwargs['prefix']}completion_input_ids"][:]
        batch[f"{kwargs['prefix']}completion_labels"][: len(batch[f"{kwargs['prefix']}prompt_input_ids"])] = [
            kwargs["label_pad_token_id"]
        ] * len(batch[f"{kwargs['prefix']}prompt_input_ids"])
    else:
        completion_tokens = kwargs["tokenizer"](
            completion, truncation=True, max_length=kwargs["max_completion_length"], add_special_tokens=True
        )
        prompt_tokens = kwargs["tokenizer"](
            prompt, truncation=True, max_length=kwargs["max_prompt_length"], add_special_tokens=True
        )

        batch[f"{kwargs['prefix']}prompt_input_ids"] = prompt_tokens["input_ids"]
        batch[f"{kwargs['prefix']}prompt_attention_mask"] = prompt_tokens["attention_mask"]

        batch[f"{kwargs['prefix']}completion_labels"] = completion_tokens["input_ids"]
        batch[f"{kwargs['prefix']}completion_attention_mask"] = completion_tokens["attention_mask"]
        if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
            batch[f"{kwargs['prefix']}completion_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                labels=torch.tensor(batch["completion_labels"])
            )

    return batch


class KTOTrainer(Trainer):
    r"""
    Initialize KTOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        args (`KTOConfig`):
            The arguments to use for training.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        data_collator (`transformers.DataCollator`, *optional*, defaults to `None`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
    """

    _tag_names = ["trl", "kto"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: KTOConfig = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        data_collator: Optional[DataCollator] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[Dict] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
    ):
        if type(args) == TrainingArguments:
            raise ValueError("Please use `KTOConfig` instead TrainingArguments.")

        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the KTOTrainer. But your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs
            model_init_kwargs["torch_dtype"] = (
                model_init_kwargs["torch_dtype"]
                if model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, model_init_kwargs["torch_dtype"])
            )

        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_kwargs to the KTOTrainer. But your ref_model is already instantiated."
            )
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs
            ref_model_init_kwargs["torch_dtype"] = (
                ref_model_init_kwargs["torch_dtype"]
                if ref_model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, ref_model_init_kwargs["torch_dtype"])
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the KTOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the KTOTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it with `pip install peft` to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if args.generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install with `pip install wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or args.precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError(
                "max_length or a tokenizer must be specified when using the default DPODataCollatorWithPadding"
            )
        if args.max_length is None:
            warnings.warn(
                "When using DPODataCollatorWithPadding, you should set `max_length` in the KTOTrainer's init"
                " it will be set to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        if args.max_length is not None:
            max_length = args.max_length

        if args.max_prompt_length is None:
            warnings.warn(
                "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the KTOTrainer's init"
                " it will be set to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128
        if args.max_prompt_length is not None:
            max_prompt_length = args.max_prompt_length

        max_completion_length = None
        if args.max_completion_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using DPODataCollatorWithPadding with an encoder decoder architecture, you should set `max_completion_length` in the KTOTrainer's init"
                " it will be set to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_completion_length = 128
        if args.max_completion_length is not None and self.is_encoder_decoder:
            max_completion_length = args.max_completion_length

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your KTOConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        # disable dropout in the model and reference model
        disable_dropout_in_model(model)
        if self.ref_model is not None:
            disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = args.padding_value if args.padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = args.truncation_mode
        self.max_completion_length = max_completion_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = args.precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        # metric
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # KTO parameter
        self.beta = args.beta
        self.desirable_weight = args.desirable_weight
        self.undesirable_weight = args.undesirable_weight

        with PartialState().local_main_process_first():
            # Shuffle the datasets
            train_dataset = train_dataset.shuffle(seed=args.data_seed)
            if eval_dataset is not None:
                eval_dataset = eval_dataset.shuffle(seed=args.data_seed)
            # Tokenize and prepare the training datasets
            train_dataset = train_dataset.map(
                _tokenize,
                fn_kwargs={"tokenizer": self.tokenizer},
                batched=True,
                desc="Tokenizing train dataset",
            )
            # Get KL datasets
            total_batch_size = (
                max(torch.cuda.device_count(), 1) * args.per_device_train_batch_size * args.gradient_accumulation_steps
            )
            if total_batch_size <= 1:
                raise ValueError(
                    "Batch size is 1 (too small). KTO will not work properly because the KL term will be equivalent to the implied reward."
                )
            # create pairs for estimating the KL term by flipping the matched pairs in each batch of size total_batch_size
            # i.e., (x_1, y_1), ..., (x_n, y_n) --> (x_1, y_n), ..., (x_n, y_1) = (x'_1, y'_1), ..., (x'_n, y'_n)
            train_kl_dataset = train_dataset.map(
                _get_kl_dataset, batched=True, batch_size=total_batch_size, desc="Extracting KL train dataset"
            )
            # Prepare the datasets
            fn_kwargs = {
                "prefix": "",
                "is_encoder_decoder": self.is_encoder_decoder,
                "tokenizer": self.tokenizer,
                "max_length": self.max_length,
                "truncation_mode": self.truncation_mode,
                "label_pad_token_id": self.label_pad_token_id,
                "max_prompt_length": self.max_prompt_length,
            }
            train_dataset = train_dataset.map(
                _process_tokens,
                fn_kwargs=fn_kwargs,
                num_proc=args.dataset_num_proc,
                desc="Processing tokenized train dataset",
            )
            fn_kwargs["prefix"] = "KL_"
            train_kl_dataset = train_kl_dataset.map(
                _process_tokens,
                fn_kwargs=fn_kwargs,
                num_proc=args.dataset_num_proc,
                remove_columns=[c for c in train_kl_dataset.column_names if c in train_dataset.column_names],
                desc="Processing tokenized train KL dataset",
            )

            # merge the datasets
            train_dataset = concatenate_datasets([train_dataset, train_kl_dataset], axis=1)

            if eval_dataset is not None:
                # Tokenize
                eval_dataset = eval_dataset.map(
                    _tokenize,
                    fn_kwargs={"tokenizer": self.tokenizer},
                    batched=True,
                    desc="Tokenizing eval dataset",
                )
                # Get KL dataset
                eval_kl_dataset = eval_dataset.map(
                    _get_kl_dataset, batched=True, batch_size=total_batch_size, desc="Extracting eval KL dataset"
                )
                # Process
                fn_kwargs = {
                    "prefix": "",
                    "is_encoder_decoder": self.is_encoder_decoder,
                    "tokenizer": self.tokenizer,
                    "max_length": self.max_length,
                    "truncation_mode": self.truncation_mode,
                    "label_pad_token_id": self.label_pad_token_id,
                    "max_prompt_length": self.max_prompt_length,
                }
                eval_dataset = eval_dataset.map(
                    _process_tokens,
                    fn_kwargs=fn_kwargs,
                    num_proc=args.dataset_num_proc,
                    desc="Processing tokenized eval dataset",
                )
                fn_kwargs["prefix"] = "KL_"
                eval_kl_dataset = eval_kl_dataset.map(
                    _process_tokens,
                    fn_kwargs=fn_kwargs,
                    num_proc=args.dataset_num_proc,
                    remove_columns=[c for c in eval_kl_dataset.column_names if c in eval_dataset.column_names],
                    desc="Processing tokenized eval KL dataset",
                )

                # merge the datasets
                eval_dataset = concatenate_datasets([eval_dataset, eval_kl_dataset], axis=1)

            desirable = train_dataset.filter(
                lambda x: x["label"], num_proc=args.dataset_num_proc, desc="Filtering desirable examples"
            )
            undesirable = train_dataset.filter(
                lambda x: not x["label"], num_proc=args.dataset_num_proc, desc="Filtering undesirable examples"
            )

            if len(desirable) != len(undesirable):
                # The lower and upper bounds come from Eq. (8) of https://arxiv.org/abs/2402.01306
                des_weight_lower_bound = round((len(undesirable) * self.undesirable_weight / len(desirable)) * 1, 2)
                des_weight_upper_bound = round((len(undesirable) * self.undesirable_weight / len(desirable)) * 1.33, 2)
                und_weight_lower_bound = round((len(desirable) * self.desirable_weight / len(undesirable)) / 1.33, 2)
                und_weight_upper_bound = round((len(desirable) * self.desirable_weight / len(undesirable)) / 1, 2)

                des_weight_in_range = des_weight_lower_bound <= self.desirable_weight <= des_weight_upper_bound
                und_weight_in_range = und_weight_lower_bound <= self.undesirable_weight <= und_weight_upper_bound

                if not (des_weight_in_range or und_weight_in_range):
                    warnings.warn(
                        f"""
                        You have different amounts of desirable/positive and undesirable/negative examples but the
                        weights on the desirable and undesirable losses don't seem to be in an ideal range. Based
                        on your data, we recommend EITHER desirable_weight in [{des_weight_lower_bound}, {des_weight_upper_bound}]
                        or undesirable_weight in [{und_weight_lower_bound}, {und_weight_upper_bound}] (but NOT BOTH).
                        See the documentation on how to optimally set these weights.""",
                        UserWarning,
                    )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))
            reference_completion_logps = []
            reference_KL_logps = []

            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_completion_logp, reference_KL_logp = self.compute_reference_log_probs(padded_batch)

                reference_completion_logp = self.accelerator.gather_for_metrics(reference_completion_logp)
                reference_completion_logps.append(reference_completion_logp.cpu())

                reference_KL_logp = self.accelerator.gather_for_metrics(reference_KL_logp)
                reference_KL_logps.append(reference_KL_logp.cpu())

            self.train_dataset = self.train_dataset.add_column(
                name="reference_logps", column=torch.cat(reference_completion_logps).float().numpy()
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_KL_logps", column=torch.cat(reference_KL_logps).float().numpy()
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

            reference_completion_logps = []
            reference_KL_logps = []

            for padded_batch in tqdm(iterable=data_loader, desc="Eval dataset reference log probs"):
                reference_completion_logp, reference_KL_logp = self.compute_reference_log_probs(padded_batch)

                reference_completion_logp = self.accelerator.gather_for_metrics(reference_completion_logp)
                reference_completion_logps.append(reference_completion_logp.cpu())

                reference_KL_logp = self.accelerator.gather_for_metrics(reference_KL_logp)
                reference_KL_logps.append(reference_KL_logp.cpu())

            eval_dataset = eval_dataset.add_column(
                name="reference_logps", column=torch.cat(reference_completion_logps).float().numpy()
            )
            eval_dataset = eval_dataset.add_column(
                name="reference_KL_logps", column=torch.cat(reference_KL_logps).float().numpy()
            )

            # Save calculated reference_chosen_logps and reference_rejected_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a KTO specific dataset."""
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(
                    self.model
                ).disable_adapter() if self.is_peft_model else nullcontext():
                    if self.is_encoder_decoder:
                        completion_logits = self.model(
                            padded_batch["prompt_input_ids"],
                            attention_mask=padded_batch["prompt_attention_mask"],
                            decoder_input_ids=padded_batch.get("completion_decoder_input_ids"),
                            labels=padded_batch["completion_labels"],
                        ).logits

                        KL_logits = self.model(
                            padded_batch["KL_prompt_input_ids"],
                            attention_mask=padded_batch["KL_prompt_attention_mask"],
                            decoder_input_ids=padded_batch.get("KL_completion_decoder_input_ids"),
                            labels=padded_batch["KL_completion_labels"],
                        ).logits
                    else:
                        completion_logits = self.model(
                            padded_batch["completion_input_ids"],
                            attention_mask=padded_batch["completion_attention_mask"],
                        ).logits

                        KL_logits = self.model(
                            padded_batch["KL_completion_input_ids"],
                            attention_mask=padded_batch["KL_completion_attention_mask"],
                        ).logits
            else:
                if self.is_encoder_decoder:
                    completion_logits = self.ref_model(
                        padded_batch["prompt_input_ids"],
                        attention_mask=padded_batch["prompt_attention_mask"],
                        decoder_input_ids=padded_batch.get("completion_decoder_input_ids"),
                        labels=padded_batch["completion_labels"],
                    ).logits

                    KL_logits = self.ref_model(
                        padded_batch["KL_prompt_input_ids"],
                        attention_mask=padded_batch["KL_prompt_attention_mask"],
                        decoder_input_ids=padded_batch.get("KL_completion_decoder_input_ids"),
                        labels=padded_batch["KL_completion_labels"],
                    ).logits
                else:
                    completion_logits = self.ref_model(
                        padded_batch["completion_input_ids"], attention_mask=padded_batch["completion_attention_mask"]
                    ).logits

                    KL_logits = self.ref_model(
                        padded_batch["KL_completion_input_ids"],
                        attention_mask=padded_batch["KL_completion_attention_mask"],
                    ).logits

        completion_logps = self.get_batch_logps(
            completion_logits,
            padded_batch["completion_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        KL_logps = self.get_batch_logps(
            KL_logits,
            padded_batch["KL_completion_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        return completion_logps, KL_logps

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        else:
            # Fixes end-dec RuntimeError
            labels = labels.clone()

        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        if self.is_encoder_decoder:
            with torch.no_grad():
                KL_logits = model(
                    batch["KL_prompt_input_ids"],
                    attention_mask=batch["KL_prompt_attention_mask"],
                    decoder_input_ids=batch.get("KL_completion_decoder_input_ids"),
                    labels=batch["KL_completion_labels"],
                ).logits

            completion_logits = model(
                batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                decoder_input_ids=batch.get("completion_decoder_input_ids"),
                labels=batch["completion_labels"],
            ).logits
        else:
            with torch.no_grad():
                KL_logits = model(
                    batch["KL_completion_input_ids"],
                    attention_mask=batch["KL_completion_attention_mask"],
                ).logits

            completion_logits = model(
                batch["completion_input_ids"],
                attention_mask=batch["completion_attention_mask"],
            ).logits

        completion_logps = self.get_batch_logps(
            completion_logits,
            batch["completion_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        KL_logps = self.get_batch_logps(
            KL_logits,
            batch["KL_completion_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        if completion_logps.shape[0] != len(batch["label"]):
            raise ValueError(
                "There is a mismatch between the number of examples in this batch and the number of "
                "examples for which an output sequence was predicted."
            )

        chosen_idx = [i for i in range(completion_logps.shape[0]) if batch["label"][i] is True]
        rejected_idx = [i for i in range(completion_logps.shape[0]) if batch["label"][i] is False]

        chosen_logps = completion_logps[chosen_idx, ...]
        rejected_logps = completion_logps[rejected_idx, ...]

        chosen_logits = completion_logits[chosen_idx, ...]
        rejected_logits = completion_logits[rejected_idx, ...]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, KL_logps)

    def kto_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the KTO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (num(chosen) in batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (num(rejected) in batch_size,)
            policy_KL_logps: Log probabilities of the policy model for the KL responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (num(chosen) in batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (num(rejected) in batch_size,)
            reference_KL_logps: Log probabilities of the reference model for the KL responses. Shape: (batch_size,)

        Returns:
            A tuple of four tensors: (losses, chosen_rewards, rejected_rewards, KL).
            The losses tensor contains the KTO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
            The KL tensor contains the detached KL divergence estimate between the policy and reference models.
        """
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        KL = self.accelerator.gather(KL).mean().clamp(min=0)

        if policy_chosen_logps.shape[0] != 0 or reference_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - KL))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # lists can't be empty -- if they are, then accelerate.gather will hang
            chosen_losses = torch.Tensor([]).to(self.accelerator.device)
            chosen_rewards = torch.Tensor([]).to(self.accelerator.device)

        if policy_rejected_logps.shape[0] != 0 or reference_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (KL - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # lists can't be empty -- if they are, then accelerate.gather will hang
            rejected_losses = torch.Tensor([]).to(self.accelerator.device)
            rejected_rewards = torch.Tensor([]).to(self.accelerator.device)

        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses),
            0,
        )

        return losses, chosen_rewards, rejected_rewards, KL

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the KTO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        batch = {k: (v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_KL_logps,
        ) = self.forward(model, batch)

        # if reference_logps in batch use them, otherwise use the reference model
        if "reference_logps" in batch:
            chosen_idx = [i for i in range(batch["reference_logps"].shape[0]) if batch["label"][i] is True]
            rejected_idx = [i for i in range(batch["reference_logps"].shape[0]) if batch["label"][i] is False]

            reference_chosen_logps = batch["reference_logps"][chosen_idx, ...]
            reference_rejected_logps = batch["reference_logps"][rejected_idx, ...]
            reference_KL_logps = batch["reference_KL_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            reference_KL_logps,
                        ) = self.forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        reference_KL_logps,
                    ) = self.forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards, KL = self.kto_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_KL_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_KL_logps,
        )

        num_chosen = torch.Tensor([len(chosen_rewards)]).to(self.accelerator.device)
        num_rejected = torch.Tensor([len(rejected_rewards)]).to(self.accelerator.device)

        all_num_chosen = self.accelerator.gather(num_chosen)
        all_num_rejected = self.accelerator.gather(num_rejected)

        prefix = "eval_" if train_eval == "eval" else ""

        if all_num_chosen.sum().item() > 0:
            metrics[f"{prefix}rewards/chosen"] = (
                (self.accelerator.gather(chosen_rewards.mean()) * all_num_chosen).nansum() / all_num_chosen.sum()
            ).item()
            metrics[f"{prefix}logps/chosen"] = (
                (self.accelerator.gather(policy_chosen_logps.mean()) * all_num_chosen).nansum() / all_num_chosen.sum()
            ).item()

        if all_num_rejected.sum().item() > 0:
            metrics[f"{prefix}rewards/rejected"] = (
                (self.accelerator.gather(rejected_rewards.mean()) * all_num_rejected).nansum() / all_num_rejected.sum()
            ).item()
            metrics[f"{prefix}logps/rejected"] = (
                (self.accelerator.gather(policy_rejected_logps.mean()) * all_num_rejected).nansum()
                / all_num_rejected.sum()
            ).item()

        metrics[f"{prefix}kl"] = KL.item()
        if all_num_chosen.sum().item() > 0 and all_num_rejected.sum().item() > 0:
            metrics[f"{prefix}rewards/margins"] = (
                metrics[f"{prefix}rewards/chosen"] - metrics[f"{prefix}rewards/rejected"]
            )

        return losses.nanmean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            if isinstance(value, list):
                self._stored_metrics[train_eval][key].extend(value)
            else:
                self._stored_metrics[train_eval][key].append(value)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        return SequentialSampler(self.train_dataset)

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

        with generate_context_manager():
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # if reference_output in batch use that otherwise use the reference model
            if "reference_output" in batch:
                reference_output = batch["reference_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                else:
                    reference_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext
        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            target_indicies = [i for i in range(len(random_batch["kl"])) if random_batch["kl"][i] is False]
            target_batch = {
                "prompt_input_ids": itemgetter(*target_indicies)(random_batch["prompt_input_ids"]),
                "prompt_attention_mask": itemgetter(*target_indicies)(random_batch["prompt_attention_mask"]),
                "prompt": itemgetter(*target_indicies)(random_batch["prompt"]),
            }
            policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, target_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[
                            [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                            for prompt, pol, ref in zip(
                                target_batch["prompt"], policy_output_decoded, ref_output_decoded
                            )
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "kto" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)

        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)




class DPOTrainer(Trainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy. For the IPO loss, beta is the regularization parameter denoted by tau in the paper.
        label_smoothing (`float`, defaults to 0):
            The robust DPO label smoothing parameter from the [cDPO](https://ericmitchell.ai/cdpo.pdf) report that should be between 0 and 0.5.
        loss_type (`str`, defaults to `"sigmoid"`):
            The type of DPO loss to use. Either `"sigmoid"` the default DPO loss,`"hinge"` loss from [SLiC](https://arxiv.org/abs/2305.10425) paper, `"ipo"` from [IPO](https://arxiv.org/abs/2310.12036) paper, or `"kto"` from the HALOs [report](https://github.com/ContextualAI/HALOs/blob/main/assets/report.pdf).
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value if it is different to the tokenizer's pad_token_id.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_target_length (`int`, defaults to `None`):
            The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
        generate_during_eval (`bool`, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        precompute_ref_log_probs (`bool`, defaults to `False`):
            Flag to precompute reference model log probabilities and evaluation datasets. This is useful if you want to train
            without the reference model and reduce the total GPU memory needed.
        model_init_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string
        ref_model_init_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the ref model from a string
    """

    _tag_names = ["trl", "dpo"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto"] = "sigmoid",
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
    ):
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the DPOTrainer. But your model is already instantiated.")

        if ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_kwargs to the DPOTrainer. But your ref_model is already instantiated."
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                preprare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    preprare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **preprare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)

        # For models that use gradient_checkpoiting, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default DPODataCollatorWithPadding"
                )
            if max_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_length` in the DPOTrainer's init"
                    " it will be set to `512` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_length = 512
            if max_prompt_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the DPOTrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_prompt_length = 128

            if max_target_length is None and self.is_encoder_decoder:
                warnings.warn(
                    "When using DPODataCollatorWithPadding with an encoder decoder architecture, you should set `max_target_length` in the DPOTrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_target_length = 128

            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.generate_during_eval = generate_during_eval
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        
        print("loss type", loss_type)
        # loss_type = "kto_pair" # added just in case.
        
        if loss_type in ["hinge", "ipo", "kto_pair"] and label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )

        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # tokenize the dataset
        train_dataset = train_dataset.map(self.tokenize_row)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(self.tokenize_row)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            self.train_dataset = self.train_dataset.add_column(
                name="reference_chosen_logps", column=all_reference_chosen_logps
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Eval dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            eval_dataset = eval_dataset.add_column(name="reference_chosen_logps", column=all_reference_chosen_logps)
            eval_dataset = eval_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            # Save calculated reference_chosen_logps and reference_rejected_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """

        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )
    
    # For Improved DPO
    def improv_dpo_mask(self, label_tokens_old):

        label_tokens = label_tokens_old[:]
        str_tokens = self.tokenizer.convert_ids_to_tokens(label_tokens)

        ## MASK "The answer is" ##
        # start_tokens = ["The", " The"]  # start tokens
        # start_token_ids = [self.tokenizer.encode(x, add_special_tokens=False)[0] for x in start_tokens]

        if "The answer is" in self.tokenizer.decode(label_tokens):
            assert(label_tokens[-1] == self.tokenizer.eos_token_id) # Last token is ...
            start_idx = -1

            # find nearest "The" or "_The" token            
            for i in range(len(label_tokens) - 1, 0, -1):
                if "The" in str_tokens[i]:
                    start_idx = i
                    label_tokens[i] = self.label_pad_token_id
                    break
                else:
                    label_tokens[i] = self.label_pad_token_id

            try:
                assert(start_idx != -1)
            except:
                import pdb
                pdb.set_trace()

        ## MASK CORRECT EQUATION ##
        start_token = "<<"  # start tokens
        end_tok = [">>"]  # end token

        # Find all equations.
        enclosed_tokens = []
        inside = False
        temp_tokens = []

        for idx, tok in enumerate(str_tokens):
            if start_token in tok:
                inside = True
                temp_tokens = []
            elif tok in end_tok and inside:
                inside = False
                enclosed_tokens.append(temp_tokens)
                temp_tokens = []
            elif inside:
                temp_tokens.append((tok, idx))

        # Get all correct equations.
        corr_equations = []

        for eq_ in enclosed_tokens:
            eq = ''.join(x[0] for x in eq_)

            # if wrong: 
            try:
                lhs = eq.split('=')[0]
                rhs = eq.split('=')[1]

                if str(eval(lhs)) == str(rhs):
                    corr_equations.append(eq_)
            except:
                continue
                # -r("[error in <<Eval>>]", eq)
                
        # Set pad_id for correct equations.
        for corr in corr_equations:
            mask_idxs = [x[1] for x in corr]
            mask_idxs.append(mask_idxs[-1] + 1) # ">>"
            mask_idxs.append(mask_idxs[0] - 1) # "<<" 

            for idx in mask_idxs:
                label_tokens[idx] = self.label_pad_token_id

        return label_tokens

    # For random masking:
    def random_dpo_mask(self, label_tokens_old):

        label_tokens = label_tokens_old[:]
        str_tokens = self.tokenizer.convert_ids_to_tokens(label_tokens)

        ## MASK "The answer is" ##
        # start_tokens = ["The", " The"]  # start tokens
        # start_token_ids = [self.tokenizer.encode(x, add_special_tokens=False)[0] for x in start_tokens]

        idxs_to_mask = []

        if "The answer is" in self.tokenizer.decode(label_tokens):
            assert(label_tokens[-1] == self.tokenizer.eos_token_id) # Last token is ...
            start_idx = -1

            # find nearest "The" or "_The" token            
            for i in range(len(label_tokens) - 1, 0, -1):
                if "The" in str_tokens[i]:
                    start_idx = i
                    # label_tokens[i] = self.label_pad_token_id
                    idxs_to_mask.append(1) # random dummy
                    break
                else:
                    # label_tokens[i] = self.label_pad_token_id
                    idxs_to_mask.append(1)

            try:
                assert(start_idx != -1)
            except:
                import pdb
                pdb.set_trace()

        ## MASK CORRECT EQUATION ##
        start_token = "<<"  # start tokens
        end_tok = [">>"]  # end token

        # Find all equations.
        enclosed_tokens = []
        inside = False
        temp_tokens = []

        for idx, tok in enumerate(str_tokens):
            if start_token in tok:
                inside = True
                temp_tokens = []
            elif tok in end_tok and inside:
                inside = False
                enclosed_tokens.append(temp_tokens)
                temp_tokens = []
            elif inside:
                temp_tokens.append((tok, idx))

        # Get all correct equations.
        corr_equations = []

        for eq_ in enclosed_tokens:
            eq = ''.join(x[0] for x in eq_)

            # if wrong: 
            try:
                lhs = eq.split('=')[0]
                rhs = eq.split('=')[1]

                if str(eval(lhs)) == str(rhs):
                    corr_equations.append(eq_)
            except:
                continue
                # print("[error in <<Eval>>]", eq)
                
        # Set pad_id for correct equations.
        for corr in corr_equations:
            mask_idxs = [x[1] for x in corr]
            mask_idxs.append(mask_idxs[-1] + 1) # ">>"
            mask_idxs.append(mask_idxs[0] - 1) # "<<" 

            for idx in mask_idxs:
                idxs_to_mask.append(1)
                # label_tokens[idx] = self.label_pad_token_id

        # select randomly len(idxs_to_mask) tokens from label_tokens 
        import random
        random_idxs = random.sample(range(len(label_tokens)), len(idxs_to_mask))
        
        # set as self.label_pad_token_id
        for idx in random_idxs:
            label_tokens[idx] = self.label_pad_token_id 

        return label_tokens

    def tokenize_row(self, feature, model: Union[PreTrainedModel, nn.Module] = None) -> Dict:
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        if not self.is_encoder_decoder:
            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337

            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)}")
            chosen_tokens = self.build_tokenized_answer(prompt, chosen)

            if not isinstance(rejected, str):
                raise ValueError(f"rejected should be an str but got {type(rejected)}")
            rejected_tokens = self.build_tokenized_answer(prompt, rejected)

            # add BOS token to head of prompt
            prompt_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
            chosen_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + chosen_tokens["prompt_input_ids"]
            rejected_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + rejected_tokens["prompt_input_ids"]

            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
            rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

            # add EOS token to end of answer [remove this also for now.]
            chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            chosen_tokens["attention_mask"].append(1)

            # [Edited!!! 02/04]
            # DO not add eos token for both DPO & Step-wise DPO. 
            # rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            # rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            # Create labels
            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(chosen_tokens["prompt_input_ids"])

            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            # For Improved DPO
            # rejected_sequence_tokens["labels"] = self.improv_dpo_mask(rejected_sequence_tokens["labels"])
            rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(rejected_tokens["prompt_input_ids"])

            for k, toks in {
                "chosen_": chosen_sequence_tokens, 
                "rejected_": rejected_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

        else:
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.tokenizer(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=batch["rejected_labels"]
                )
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=batch["chosen_labels"]
                )

        return batch

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        # compute reference logps
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(
                    self.model
                ).disable_adapter() if self.is_peft_model else nullcontext():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, padded_batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, padded_batch)

        return reference_chosen_logps, reference_rejected_logps

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        return concatenated_batch

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if reference_free:
            ref_logratios = 0
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        # 여기한번 보기.
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        # print("Per token Logps", per_token_logps, "\nLogit shape", logits.shape, "\nPer_token_logps shape", per_token_logps.shape)

        # clamp
        # per_token_logps = torch.clamp(per_token_logps, min=math.log(0.1))

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        # print("all log ps", all_logps)

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        # print(f"chosen logps {policy_chosen_logps.detach().cpu()}\n"
        # f"reject logps {policy_rejected_logps.detach().cpu()}\n"
        # f"chosen rewards {chosen_rewards}\n"
        # f"rejected rewards {rejected_rewards}\n"
        # f"rewards {chosen_rewards - rejected_rewards}\n"
        # f"{'---' * 20}")

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        policy_output = model.generate(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # if reference_output in batch use that otherwise use the reference model
        if "reference_output" in batch:
            reference_output = batch["reference_output"]
        else:
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    reference_output = self.model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
            else:
                reference_output = self.ref_model.generate(
                    input_ids=batch["prompt_input_ids"],
                    attention_mask=batch["prompt_attention_mask"],
                    max_length=self.max_length,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, random_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[
                            [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                            for prompt, pol, ref in zip(
                                random_batch["prompt"], policy_output_decoded, ref_output_decoded
                            )
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "sft" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(tag_names=self._tag_names, kwargs=kwargs)

        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)
