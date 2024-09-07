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
# from accelerate import init_empty_weights, infer_auto_device_map

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# sys.path.append(os.path.dirname(__file__))

from Script.config import Args

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5'

class LLMModelBase:
    def __init__(self, args: Args) -> None:
        self.args = args
        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True)
        if self.args.engine == "vllm":
            # Create vllm instance  
            self.llm = LLM(
                model=self.args.model_path,
                trust_remote_code=True,
                tokenizer=self.args.model_path,
                tokenizer_mode='slow',
                tensor_parallel_size=self.args.tensor_parallel_size
            )
            
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
            self.sampling_params = SamplingParams(
                # n=self.args.n,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                max_tokens=self.args.max_tokens,
                skip_special_tokens=self.args.skip_special_tokens,
                logits_processors=[logits_processor] if logits_processor else None
            )
        
        elif self.args.engine == "hf":
            # # 初始化多卡推理的设备映射
            # with init_empty_weights():
            #     model = AutoModelForCausalLM.from_pretrained(
            #         self.args.model_path,
            #         torch_dtype=torch.bfloat16 if self.args.dtype == "bfloat16" else torch.float16
            #     )

            # # 使用 accelerate 自动推理设备映射
            # device_map = infer_auto_device_map(
            #     model,
            #     max_memory={i: '20GB' for i in range(4)},  # 根据实际 GPU 内存进行配置
            #     no_split_module_classes=["LlamaDecoderLayer"],  # 根据模型结构进行模块划分
            # )
            
            # 加载模型到多 GPU 上
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_path,
                device_map={0: 'cuda:1'},
                torch_dtype=torch.bfloat16 if self.args.dtype == "bfloat16" else torch.float16
            )

    def generate(self, prompt, chat_lst):
        '''chat_lst = [{}, {}, {}]'''
        chat_lst.append({'role': 'user', 'content': prompt})
        template = self.tokenizer.apply_chat_template(chat_lst, tokenize=False, add_generation_prompt=True)
        
        if self.args.engine == "vllm":
            output = self.llm.generate(template, self.sampling_params)
            output = output[0].outputs[0].text
            
        elif self.args.engine == "hf":
            inputs = self.tokenizer(template, return_tensors="pt", padding=True, truncation=True).to('cuda: 1')
            output = self.model.generate(**inputs,
                                    tokenizer=self.tokenizer, 
                                    do_sample=True, 
                                    temperature=self.args.temperature, 
                                    top_p=self.args.top_p, 
                                    max_length=self.args.max_tokens, 
                                    num_return_sequences=1,
                                    )
            # Remove the input from the output
            output = self.tokenizer.decode(output[len(inputs[0]):])
        
        chat_lst.append({'role': 'assistant', 'content': output})    
        
        return output, chat_lst
     
    def generate_mask(self, input_lst, mask_lst):
        # 记录 value=1 的 index 位置
        mask_index = [index for index, value in enumerate(mask_lst) if value == 1]
        filter_input_lst = [input_lst[i] for i in mask_index]
        if self.args.engine == "vllm":
            outputs = self.llm.generate(filter_input_lst, self.sampling_params)
            outputs_lst = [q.outputs[0].text for q in outputs]
            
        elif self.args.engine == "hf":
            inputs = self.tokenizer(filter_input_lst, return_tensors="pt", padding=True, truncation=True).to('cuda: 1')
            outputs = self.model.generate(**inputs,
                                    tokenizer=self.tokenizer, 
                                    do_sample=True, 
                                    temperature=self.args.temperature, 
                                    top_p=self.args.top_p, 
                                    max_length=self.args.max_tokens, 
                                    num_return_sequences=1,
                                    )
            # Remove the input from the output
            outputs_lst = self.tokenizer.batch_decode(outputs[i][len(inputs[0]):] for i in range(len(filter_input_lst)))
        
        return outputs_lst, mask_index
    
    def get_template_dialog(self, dialog_lst, prompt_lst):
        temp_lst = []
        new_dialog_lst = []
        
        for dialog, prompt in zip(dialog_lst, prompt_lst):
            template = self.tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=False)
            new_dialog_lst.append(template)
            template += prompt
            temp_lst.append(template)
            
        return temp_lst, new_dialog_lst
        
    def generate_first(self, query_prompt_lst, answer_prompt_lst):
        # 储存 chat 数据
        chat_lst = []
        
        # 构造指令
        if self.args.engine == "vllm":
            querys = self.llm.generate(query_prompt_lst, self.sampling_params)
            querys_lst = [q.outputs[0].text for q in querys]
            
        elif self.args.engine == "hf":
            inputs = self.tokenizer(query_prompt_lst, return_tensors="pt", padding=True, truncation=True).to('cuda: 1')
            querys = self.model.generate(**inputs,
                                    tokenizer=self.tokenizer, 
                                    do_sample=True, 
                                    temperature=self.args.temperature, 
                                    top_p=self.args.top_p, 
                                    max_length=self.args.max_tokens, 
                                    num_return_sequences=1,
                                    )
            # Remove the input from the output
            querys_lst = self.tokenizer.batch_decode(querys[i][len(inputs[0]):] for i in range(len(query_prompt_lst)))
            
        query_temp_lst = []
        for query in querys_lst:
            chat = [{"role": "user", "content": query}]
            # add_generation_prompt=True 可以添加一些生成提示标记如 <|im_start|>assistant
            template = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            query_temp_lst.append(template)
            chat_lst.append(chat)
            
        new_answer_prompt_lst = []
        for query_temp, answer_prompt in zip(query_temp_lst, answer_prompt_lst):
            answer_prompt = answer_prompt.format(query=query_temp)
            new_answer_prompt_lst.append(answer_prompt)
        
        # 构造指令的答案
        if self.args.engine == "vllm":
            answers = self.llm.generate(new_answer_prompt_lst, self.sampling_params)
            answers_lst = [q.outputs[0].text for q in answers]
            
        elif self.args.engine == "hf":
            inputs = self.tokenizer(new_answer_prompt_lst, return_tensors="pt", padding=True, truncation=True).to('cuda: 1')
            answers = self.model.generate(**inputs,
                                    tokenizer=self.tokenizer, 
                                    do_sample=True, 
                                    temperature=self.args.temperature, 
                                    top_p=self.args.top_p, 
                                    max_length=self.args.max_tokens, 
                                    num_return_sequences=1,
                                    )
            # Remove the input from the output
            answers_lst = self.tokenizer.batch_decode(answers[i][len(inputs[0]):] for i in range(len(answers)))
        
        for i, ans in enumerate(answers_lst):
            chat_lst[i].append({"role": "assistant", "content": ans})
        
        return chat_lst
    
    def generate_mt(self, query_prompt_lst, answer_prompt_lst, turn_lst, first_dialog_lst):
        chat_lst = first_dialog_lst
        max_turn = max(turn_lst)
        
        # 修改首轮对话的格式
        temp_lst, _ = self.get_template_dialog(first_dialog_lst, query_prompt_lst)
        
        for i in range(2, max_turn+1):
            # 标记每个prompt是否已经生成到头了
            is_end_lst = [1 if t >= i else 0 for t in turn_lst]
            # 生成指令 query
            query_lst, mask_index = self.generate_mask(temp_lst, is_end_lst)
            for index, query in zip(mask_index, query_lst):
                chat_lst[index].append({"role": "user", "content": query})  
            temp_lst, _ = self.get_template_dialog(chat_lst, answer_prompt_lst)  
                
            # 生成答案 answer
            answer_lst, mask_index = self.generate_mask(temp_lst, is_end_lst)
            for index, answer in zip(mask_index, answer_lst):
                chat_lst[index].append({"role": "assistant", "content": answer})  
            temp_lst, _ = self.get_template_dialog(chat_lst, answer_prompt_lst)  
        
        return chat_lst
    
    def generate_eval(self, evaluate_prompt_lst, full_dialog_lst):
        temp_lst, dialog_lst = self.get_template_dialog(full_dialog_lst, evaluate_prompt_lst)
        if self.args.engine == "vllm":
            outputs = self.llm.generate(temp_lst, self.sampling_params)
            outputs_lst = [q.outputs[0].text for q in outputs]
            
        elif self.args.engine == "hf":
            inputs = self.tokenizer(temp_lst, return_tensors="pt", padding=True, truncation=True).to('cuda: 1')
            outputs = self.model.generate(**inputs,
                                    tokenizer=self.tokenizer, 
                                    do_sample=True, 
                                    temperature=self.args.temperature, 
                                    top_p=self.args.top_p, 
                                    max_length=self.args.max_tokens, 
                                    num_return_sequences=1,
                                    )
            # Remove the input from the output
            outputs_lst = self.tokenizer.batch_decode(outputs[i][len(inputs[0]):] for i in range(len(temp_lst)))
        
        return outputs_lst, dialog_lst
    
    
if __name__ == '__main__':
    def test_vllm():
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
            "今天天气真好，咱们出去",
            "明天就要开学了，我的作业还没写完，",
            "请你介绍一下你自己。AI："
        ]
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)
        model_path = "/home/yangjy/Study/ChatAgent_RAG/llm_models/chatglm3-6b/"
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tokenizer=model_path,
            tokenizer_mode='slow',
            tensor_parallel_size=2
        )
        outputs = llm.generate(prompts, sampling_params)
        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
    test_vllm()
    
    