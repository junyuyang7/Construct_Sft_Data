import contextlib
from functools import partial
from typing import List, Union

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

import os
import sys 
sys.path.append(os.path.dirname(__file__))

from utils import encode_with_messages_format_with_llama2_chat, encode_with_messages_format

# 获取train_dataset
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class GetData:
    def __init__(self, func_name="encode_with_messages_format"):
        self.func_name = func_name

    def get_dataset(self, train_files: List[str], tokenizer, max_seq_length, sample_percentage=1.0, seed=0):
        """ get training dataset with a specified seed """

        raw_datasets = self.load_raw_dataset(
            train_files, sample_percentage=sample_percentage, seed=seed)
        lm_datasets = self.encode_data(
            raw_datasets, tokenizer, max_seq_length)
        return lm_datasets

    # TODO: 初始数据形式是怎么样的？
    def load_raw_dataset(self, train_files: Union[List[str], str], sample_size=None, sample_percentage=1.0, seed=0):
        """ load raw dataset """
        if isinstance(train_files, str):
            train_files = [train_files]
        processed_datasets = load_dataset(
            "json",
            data_files=train_files,
        )["train"]
        if sample_size is None:
            sample_size = int(len(processed_datasets) * sample_percentage)

        if sample_size == len(processed_datasets):
            return processed_datasets  # not shuffle

        with temp_seed(seed):
            index = np.random.permutation(len(processed_datasets))[:sample_size]

        sampled_dataset = processed_datasets.select(index)

        return sampled_dataset


    def encode_data(self, raw_datasets, tokenizer, max_seq_length, processing_num_workers=10, overwrite_cache=False):
        """ encode data with the specified tokenizer and the chat format. """
        # if already encoded, return
        if "input_ids" in raw_datasets.features:
            return raw_datasets
        encode_function = self.get_encode_function(
            raw_datasets, tokenizer, max_seq_length, self.func_name)
        # To speed up this part, we use multiprocessing.
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=processing_num_workers,
            load_from_cache_file=not overwrite_cache,
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")
        return lm_datasets

    def get_encode_function(self, raw_datasets, tokenizer, max_seq_length):
        # sourcery skip: remove-unnecessary-else, swap-if-else-branches
        """ get encode function based on the dataset. """
        if "messages" in raw_datasets.coluifmn_names:
            if self.func_name == "encode_with_messages_format":
                encode_func = encode_with_messages_format
            else:
                encode_func =  encode_with_messages_format_with_llama2_chat
            encode_function = partial(
                encode_func,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
            )
        else:
            raise ValueError(
                "You need to have either 'messages' in your column names.")
        return encode_function
    
    def get_dataloader(self, dataset, tokenizer, batch_size=1):
        data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer, padding="longest") 
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,  # When getting gradients, we only do this single batch process
                                collate_fn=data_collator)
        print("There are {} examples in the dataset".format(len(dataset)))
        return dataloader




