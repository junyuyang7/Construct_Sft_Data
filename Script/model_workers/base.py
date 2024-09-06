# 1.进行对话
import torch
import os
import sys
import json
import time
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import str_utils


class LLMModelBase:
    def __init__(self, args) -> None:
        self.args = args

    def get_model(self):
        if self.args.engine == "vllm":
            # Create vllm instance  
            llm = LLM(model=self.args.model_path, 
                    dtype=self.args.dtype,
                    trust_remote_code=True,
                    gpu_memory_utilization=self.args.gpu_memory_utilization,
                    max_model_len=self.args.max_model_len,
                    swap_space=self.args.swap_space,
                    tensor_parallel_size=self.args.tensor_parallel_size,
                    seed=self.args.seed if self.args.seed is not None else self.args.timestamp,
                    enable_prefix_caching=True)
            
            def de_md_logits_processor_for_llama3_1(token_ids, logits):
                # Only process the initial logits
                if len(token_ids) == 0:
                    logits[2] = -9999.999 # "#": 2,
                    logits[567] = -9999.999 # "##": 567,
                    logits[14711] = -9999.999 # "###": 14711,
                    logits[827] = -9999.999 # "####": 827,

                return logits

            if self.args.logits_processor and "llama-3.1" in self.args.model_path.lower():
                logits_processor = de_md_logits_processor_for_llama3_1
                print(f"Logits processor applied: {logits_processor}")
            else:
                logits_processor = None
                
            # Define sampling parameters
            sampling_params = SamplingParams(
                n=self.args.n,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                max_tokens=self.args.max_tokens,
                skip_special_tokens=self.args.skip_special_tokens,
                stop=self.stop_tokens,
                stop_token_ids=self.stop_token_ids,
                logits_processors=[logits_processor] if logits_processor else None
            )
            
            return llm, sampling_params
        
        elif self.args.engine == "hf":
            # Load the model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_path,
                device_map={'':torch.cuda.current_device()},
                torch_dtype=torch.bfloat16 if self.args.dtype == "bfloat16" else torch.float16
            )
            return model, tokenizer
        
    def generate(self, prompt):
        if self.args.engine == "vllm":
            llm, sampling_params = self.get_model()
            outputs = llm.generate(prompt, sampling_params)
            # output_list = outputs[0].outputs
            # if self.args.shuffle:
            #     random.shuffle(output_list)
                
        elif self.args.engine == "hf":
            model, tokenizer = self.get_model()
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(torch.cuda.current_device())
            # Gemma-2 bug, so we cannot set num_return_sequences > 1. 
            # Instead, we repeat the input n times.
            # inputs = input.repeat(self.args.n, 1).to(torch.cuda.current_device())
            outputs = model.generate(**inputs,
                                    tokenizer=tokenizer, 
                                    do_sample=True, 
                                    temperature=self.args.temperature, 
                                    top_p=self.args.top_p, 
                                    max_length=self.args.max_tokens, 
                                    num_return_sequences=1,
                                    )
            # Remove the input from the output
            output_list = tokenizer.batch_decode(outputs[i][len(inputs[i]):] for i in range(len(outputs)))
            # Stop on the first stop token
            for i, completion in enumerate(output_list):
                for stop_token in self.stop_tokens:
                    if stop_token in completion:
                        output_list[i] = completion[:completion.index(stop_token)]
        
        return output_list
    
    def generate_mt(self, prompt, history_json):
        model, tokenizer = self.get_model()
        template = tokenizer.apply_chat_template(history_json, tokenize=False, add_generation_prompt=False)
        prompt = prompt.format(history=template)

        if self.args.engine == "vllm":
            llm, sampling_params = self.get_model()
            output = llm.generate(prompt, sampling_params)
                
        elif self.args.engine == "hf":
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(torch.cuda.current_device())
            output = model.generate(**inputs,
                                    tokenizer=tokenizer, 
                                    do_sample=True, 
                                    temperature=self.args.temperature, 
                                    top_p=self.args.top_p, 
                                    max_length=self.args.max_tokens, 
                                    num_return_sequences=1,
                                    )
            # Remove the input from the output
            # output_list = tokenizer.batch_decode(outputs[i][len(inputs[i]):] for i in range(len(outputs)))
            output_text = tokenizer.decode(output[len(inputs):])
            # Stop on the first stop token
            # for i, completion in enumerate(output_list):
            #     for stop_token in self.stop_tokens:
            #         if stop_token in completion:
            #             output_list[i] = completion[:completion.index(stop_token)]
            tmp_output = output_text
            for stop_token in self.stop_tokens:
                if stop_token in tmp_output:
                    output_text = tmp_output[:tmp_output.index(stop_token)]

        return output, tokenizer
    
    