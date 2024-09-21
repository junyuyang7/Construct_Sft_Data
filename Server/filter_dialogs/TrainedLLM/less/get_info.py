"""
    This script is used for getting gradients or representations of a pre-trained model, a lora model, or a peft-initialized model for a given task.
"""

import argparse

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))

import pdb
from copy import deepcopy
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from collect_grad import collect_grads
from Server.filter_dialogs.utils import GetData
from configs import GetGradArgument


def load_model(model_name_or_path: str,
               torch_dtype: Any = torch.bfloat16) -> Any:
    """
    Load a model from a given model name or path.

    Args:
        model_name_or_path (str): The name or path of the model.
        torch_dtype (Any, optional): The torch data type. Defaults to torch.bfloat16.

    Returns:
        Any: The loaded model.
    """

    is_peft = os.path.exists(os.path.join(
        model_name_or_path, "adapter_config.json"))
    if is_peft:
        # load this way to make sure that optimizer states match the model structure
        config = LoraConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, torch_dtype=torch_dtype, device_map="auto")
        model = PeftModel.from_pretrained(
            base_model, model_name_or_path, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, device_map="auto")

    for name, param in model.named_parameters():
        if 'lora' in name or 'Lora' in name:
            param.requires_grad = True
    return model

def get_info():
    args = GetGradArgument()
    data_getter = GetData()
    assert args.task is not None or args.train_file is not None

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
    model = load_model(args.model_path, dtype)

    # pad token is not added by default for pretrained models
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.initialize_lora:
        assert not isinstance(model, PeftModel)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)

    if isinstance(model, PeftModel):
        model.print_trainable_parameters()

    adam_optimizer_state = None
    if args.info_type == "grads" and args.gradient_type == "adam":
        optimizer_path = os.path.join(args.model_path, "optimizer.bin")
        adam_optimizer_state = torch.load(
            optimizer_path, map_location="cpu")["state"]

    if args.mode == 'valid':
        dataset = data_getter.get_dataset(args.train_file, tokenizer, args.max_length, sample_percentage=1.0)
        dataloader = data_getter.get_dataloader(dataset, tokenizer=tokenizer)
    else:
        assert args.train_file is not None
        dataset = data_getter.get_dataset(
            args.train_file, tokenizer, args.max_length, sample_percentage=1.0)
        columns = deepcopy(dataset.column_names)
        columns.remove("input_ids")
        columns.remove("labels")
        columns.remove("attention_mask")
        dataset = dataset.remove_columns(columns)
        dataloader = data_getter.get_dataloader(dataset, tokenizer=tokenizer)

    collect_grads(dataloader,
                model,
                args.output_path,
                proj_dim=args.gradient_projection_dimension,
                gradient_type=args.gradient_type,
                adam_optimizer_state=adam_optimizer_state,
                max_samples=args.max_samples)
