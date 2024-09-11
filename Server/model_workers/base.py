# 1.进行对话
import torch
import os
import sys
import json
import time
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaForCausalLM
from vllm import LLM, SamplingParams
import re
import torch
from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model,load_checkpoint_and_dispatch


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# sys.path.append(os.path.dirname(__file__))

from Server.config import Args, device_ids, stop_tokens_dict, llm_model_dict

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_VISIBLE_DEVICES'] = device_ids

class LLMModelBase:
    def __init__(self, args: Args) -> None:
        self.args = args
        cuda_list = device_ids.split(',')
        memory = self.args.max_memory
        max_memory = {int(cuda):memory for cuda in cuda_list}

        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path, 
            trust_remote_code=True, 
            model_max_length=args.max_tokens)
        try:
            print('tokenizer.pad_token: ', self.tokenizer.pad_token)
        except:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载 stop_tokens
        model_config = stop_tokens_dict[model_name]
        stop_tokens = model_config["stop_tokens"]
        stop_tokens_assistant = model_config["stop_tokens_assistant"]
        self.stop_tokens = stop_tokens + stop_tokens_assistant
        self.stop_token_ids = model_config["stop_token_ids"]

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
                stop=self.stop_tokens,
                stop_token_ids=self.stop_token_ids,
                logits_processors=[logits_processor] if logits_processor else None
            )
        
        elif self.args.engine == "hf":
            config = AutoConfig.from_pretrained(self.args.model_path, trust_remote_code=True,)

            # # 初始化多卡推理的设备映射
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(
                    config,
                    torch_dtype=torch.bfloat16 if self.args.dtype == "bfloat16" else torch.float16,
                    trust_remote_code=True,
                )
            no_split_module_classes = self.model._no_split_modules

            # 使用 accelerate 自动推理设备映射
            device_map = infer_auto_device_map(self.model, max_memory=max_memory, no_split_module_classes=no_split_module_classes)
            device_map["lm_head"] = 0

            print('no_split_module_classes: ', no_split_module_classes)
            print('device_map: ', device_map)

            load_checkpoint_in_model(self.model, self.args.model_path, device_map=device_map) #加载权重
            self.model = dispatch_model(self.model, device_map=device_map) 
            
            # 不使用 accelerate
            # self.model = AutoModelForCausalLM.from_pretrained(
            #     self.args.model_path,
            #     device_map='auto,
            #     torch_dtype=torch.bfloat16 if self.args.dtype == "bfloat16" else torch.float16,
            #     trust_remote_code=True,
            # )
            
    def get_loss(self, prompt):
        if self.args.engine == "vllm":
            return 
        
        elif self.args.engine == "hf":
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.model.device) 
            output = self.model.generate(**inputs, do_sample=False)
            # output = self.model.generate(**inputs,
            #                         do_sample=True, 
            #                         temperature=self.args.temperature, 
            #                         top_p=self.args.top_p, 
            #                         max_length=self.args.max_tokens, 
            #                         num_return_sequences=1,
            #                         )

    def generate(self, prompt, chat_lst):
        '''chat_lst = [{}, {}, {}]'''
        chat_lst.append({'role': 'user', 'content': prompt})
        template = self.tokenizer.apply_chat_template(chat_lst, tokenize=False, add_generation_prompt=True)
        
        if self.args.engine == "vllm":
            output = self.llm.generate(template, self.sampling_params)
            output = output[0].outputs[0].text
            
        elif self.args.engine == "hf":
            inputs = self.tokenizer.encode(template, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to('cuda').to(self.model.device) 

            output = self.model.generate(**inputs, do_sample=False)
            output = self.model.generate(inputs,
                                    do_sample=True, 
                                    temperature=self.args.temperature, 
                                    top_p=self.args.top_p, 
                                    max_length=self.args.max_tokens, 
                                    num_return_sequences=1,
                                    output_scores=True, 
                                    return_dict_in_generate=True,
                                    )
            generated_ids = output.sequences
            # Remove the input from the output
            output = self.tokenizer.decode(generated_ids[len(inputs):])
        
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
            inputs = self.tokenizer.batch_encode_plus(filter_input_lst, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(self.model.device) 
            
            inputs_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
            
            outputs = self.model.generate(**inputs, do_sample=False)
            # outputs = self.model.generate(inputs_ids,
            #                             attention_mask=attention_mask,
            #                             do_sample=True, 
            #                             temperature=self.args.temperature, 
            #                             top_p=self.args.top_p, 
            #                             max_length=self.args.max_tokens, 
            #                             num_return_sequences=1,
            #                             output_scores=True, 
            #                             return_dict_in_generate=True
            #                             ) 
            # Remove the input from the output
            generated_ids = outputs.sequences
            # Remove the input from the output
            outputs_lst = self.tokenizer.batch_decode(generated_ids[i][len(inputs_ids[i]):] for i in range(len(inputs_ids)))
        
        return outputs_lst, mask_index
    
    def get_template_dialog(self, dialog_lst, prompt):
        temp_lst = []
        new_dialog_lst = []
        
        for dialog in dialog_lst:
            template = self.tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=False)
            new_dialog_lst.append(template)
            template += prompt
            temp_lst.append(template)
            
        return temp_lst, new_dialog_lst
    
    def generate_first_query(self, first_query_prompt_lst):
        # 构造指令
        if self.args.engine == "vllm":
            querys = self.llm.generate(first_query_prompt_lst, self.sampling_params)
            querys_lst = [q.outputs[0].text for q in querys]
            
        elif self.args.engine == "hf":
            inputs = self.tokenizer.batch_encode_plus(first_query_prompt_lst, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to('cuda')
            
            inputs_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
            
            outputs = self.model.generate(inputs_ids,
                                        attention_mask=attention_mask,
                                        do_sample=True, 
                                        temperature=self.args.temperature, 
                                        top_p=self.args.top_p, 
                                        max_length=self.args.max_tokens, 
                                        num_return_sequences=1,
                                        output_scores=True, 
                                        return_dict_in_generate=True
                                        ).to(self.model.device) 
            # Remove the input from the output
            generated_ids = outputs.sequences
            # Remove the input from the output
            querys_lst = self.tokenizer.batch_decode(generated_ids[i][len(inputs_ids[i]):] for i in range(len(inputs_ids)))
        
        # 去除 1. 这种前缀
        def remove_num_prefix(text):
            return re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)    
        
        new_querys_lst = []
        for q in querys_lst:
            q_lst = q.strip().split('\n')
            q_lst = [remove_num_prefix(sample) for sample in q_lst]
            new_querys_lst.append(q_lst)
            
        return new_querys_lst
        
    def generate_first_answer(self, querys_lst, answer_prompt):
        # 储存 chat 数据
        chat_lst = []
        query_temp_lst = []
        
        for query in querys_lst:
            chat = [{"role": "user", "content": query}]
            # add_generation_prompt=True 可以添加一些生成提示标记如 <|im_start|>assistant
            template = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            query_temp_lst.append(template)
            chat_lst.append(chat)
            
        new_answer_prompt_lst = []
        for query_temp in query_temp_lst:
            fill_answer_prompt = answer_prompt.format(query=query_temp)
            new_answer_prompt_lst.append(fill_answer_prompt)
        
        # 构造指令的答案
        if self.args.engine == "vllm":
            answers = self.llm.generate(new_answer_prompt_lst, self.sampling_params)
            answers_lst = [q.outputs[0].text for q in answers]
            
        elif self.args.engine == "hf":
            inputs = self.tokenizer.batch_encode_plus(new_answer_prompt_lst, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to('cuda')
            
            inputs_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
            
            outputs = self.model.generate(inputs_ids,
                                        attention_mask=attention_mask,
                                        do_sample=True, 
                                        temperature=self.args.temperature, 
                                        top_p=self.args.top_p, 
                                        max_length=self.args.max_tokens, 
                                        num_return_sequences=1,
                                        output_scores=True, 
                                        return_dict_in_generate=True
                                        ).to(self.model.device) 
            # Remove the input from the output
            generated_ids = outputs.sequences
            # Remove the input from the output
            answers_lst = self.tokenizer.batch_decode(generated_ids[i][len(inputs_ids[i]):] for i in range(len(inputs_ids)))
        
        for i, ans in enumerate(answers_lst):
            chat_lst[i].append({"role": "assistant", "content": ans})
        
        return chat_lst
    
    def generate_mt(self, query_prompt, answer_prompt, turn_lst, first_dialog_lst):
        chat_lst = first_dialog_lst
        max_turn = max(turn_lst)
        
        # 修改首轮对话的格式
        temp_lst, _ = self.get_template_dialog(first_dialog_lst, query_prompt)
        
        for i in range(2, max_turn+1):
            # 标记每个prompt是否已经生成到头了
            is_end_lst = [1 if t >= i else 0 for t in turn_lst]
            # 生成指令 query
            query_lst, mask_index = self.generate_mask(temp_lst, is_end_lst)
            for index, query in zip(mask_index, query_lst):
                chat_lst[index].append({"role": "user", "content": query})  
            temp_lst, _ = self.get_template_dialog(chat_lst, answer_prompt)  
                
            # 生成答案 answer
            answer_lst, mask_index = self.generate_mask(temp_lst, is_end_lst)
            for index, answer in zip(mask_index, answer_lst):
                chat_lst[index].append({"role": "assistant", "content": answer})  
            temp_lst, _ = self.get_template_dialog(chat_lst, answer_prompt)  
        
        return chat_lst
    
    def generate_eval(self, evaluate_prompt, full_dialog_lst):
        temp_lst, dialog_lst = self.get_template_dialog(full_dialog_lst, evaluate_prompt)
        
        if self.args.engine == "vllm":
            outputs = self.llm.generate(temp_lst, self.sampling_params)
            outputs_lst = [q.outputs[0].text for q in outputs]
            
        elif self.args.engine == "hf":
            inputs = self.tokenizer(temp_lst, return_tensors="pt", padding=True, truncation=True).to('cuda')
            outputs = self.model.generate(**inputs,
                                    do_sample=True, 
                                    temperature=self.args.temperature, 
                                    top_p=self.args.top_p, 
                                    max_length=self.args.max_tokens, 
                                    num_return_sequences=1,
                                    ).to(self.model.device) 
            # Remove the input from the output
            outputs_lst = self.tokenizer.batch_decode(outputs[i][len(inputs[0]):] for i in range(len(temp_lst)))
        
        return outputs_lst, dialog_lst
    
    
if __name__ == '__main__':
    prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
            "今天天气真好，咱们出去",
            "明天就要开学了，我的作业还没写完，",
            "请你介绍一下你自己。AI："
        ]
    model_name = 'llama-2-13b-chat'
    model_path = llm_model_dict[model_name]
    cuda_list = device_ids.split(',')
    memory = '20GiB'
    max_memory = {int(cuda):memory for cuda in cuda_list}
    
    def test_vllm():
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)
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
            
    def test_hf():
        model_config = stop_tokens_dict[model_name]
        stop_tokens = model_config["stop_tokens"]
        stop_tokens_assistant = model_config["stop_tokens_assistant"]
        stop_tokens += stop_tokens_assistant
        stop_token_ids = model_config["stop_token_ids"]

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            model_max_length=512)
        tokenizer.pad_token = tokenizer.eos_token
        
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True,)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                    config,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
        model.tie_weights()

        # no_split_module_classes = LlamaForCausalLM._no_split_modules
        no_split_module_classes = model._no_split_modules

        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes) #自动划分每个层的设备

        print(no_split_module_classes)
        # print(device_map)
        
        load_checkpoint_in_model(model, model_path, device_map=device_map, offload_folder=None) #加载权重
        model.tie_weights()
        
        model = dispatch_model(model, device_map=device_map) #并分配到具体的设备上

        # 只需要推理
        torch.set_grad_enabled(False)
        model.eval()
        
        inputs = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(model.device)
        inputs_ids = inputs['input_ids']

        print('begin')
        outputs = model.generate(**inputs, 
                                do_sample=False, 
                                temperature=1.0, 
                                top_p=1.0, 
                                num_return_sequences=1,
                                max_new_tokens=1024,
                                output_scores=True, 
                                return_dict_in_generate=True)
        # outputs = model.generate(**inputs,
        #                         do_sample=False, 
        #                         temperature=1.0, 
        #                         top_p=1.0, 
        #                         max_length=1024, 
        #                         num_return_sequences=1,
        #                         output_scores=True, 
        #                         return_dict_in_generate=True
        #                         )
        generated_ids = outputs.sequences
        print('finish')
        # Remove the input from the output
        output_texts = tokenizer.batch_decode(generated_ids[i][len(inputs_ids[i]):] for i in range(len(inputs_ids)))
        # output_texts = tokenizer.batch_decode(generated_ids[i] for i in range(len(inputs_ids)))
        
        # Stop on the first stop token
        for i, completion in enumerate(output_texts):
            for stop_token in stop_tokens:
                if stop_token in completion:
                    output_texts[i] = completion[:completion.index(stop_token)]
        
        # Print the outputs.
        for prompt, generated_text in zip(prompts, output_texts):
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
    def test():
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model
        import torch

        model_path = "/home/yangjy/Study/ChatAgent_RAG/llm_models/Llama-2-13b-chat-hf/"
        cuda_list = device_ids.split(',')
        memory = '20GiB'
        max_memory = {int(cuda):memory for cuda in cuda_list}

        config = AutoConfig.from_pretrained(model_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16) 

        no_split_module_classes = model._no_split_modules

        device_map = infer_auto_device_map(model, max_memory=max_memory,no_split_module_classes=['LlamaDecoderLayer']) #自动划分每个层的设备
        load_checkpoint_in_model(model,model_path,device_map=device_map, offload_folder=None) #加载权重
        model = dispatch_model(model,device_map=device_map) #并分配到具体的设备上

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        torch.set_grad_enabled(False)
        model.eval()
        sents=['你是谁']
        ids = tokenizer(sents,max_length=1800,padding=True,truncation=True,return_tensors="pt")
        ids = ids.to(model.device) 
        # outputs = model.generate(**ids, do_sample=False)

        outputs = model.generate(**ids,
                                do_sample=False, 
                                temperature=1.0, 
                                top_p=1.0, 
                                max_length=1024, 
                                num_return_sequences=1,
                                output_scores=True, 
                                return_dict_in_generate=True
                                )

        print(outputs)

    # test_vllm()
    test_hf()
    # test()
    
    