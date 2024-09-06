from Script.ModelBase import AskModel, JudgeModel, AnswerModel, TopicModel
from Script.model_workers.base import LLMModelBase
from typing import *
import json
import pandas as pd

class DataConstructer:
    def __init__(self, llm_model: LLMModelBase):
        self.llm_model = llm_model

    def load_model(self, model_name):
        pass

    def chat(self, model, prompt, history_json=None):
        if history_json:
            output = self.llm_model.generate_mt(prompt)
        else:
            output = self.llm_model.generate(prompt)
        return output

    def process_history(self, history_json):
        pass

    def construct_first_dialog(self, model, query_prompt_lst, answer_prompt_lst):
        full_data_lst = [[] for _ in range(len(query_prompt_lst))]

        for i, query_prompt, answer_prompt in enumerate(zip(query_prompt_lst, answer_prompt_lst)):
            first_query = self.chat(model, [query_prompt])
            full_data_lst[i].append({'role': 'user', 'content': first_query[0]})

            # answer_prompt 需要有首轮指令
            answer_prompt = answer_prompt.format(query=first_query)
            first_answer = self.chat(model, [answer_prompt])
            full_data_lst[i].append({'role': 'assistant', 'content': first_answer[0]})
        
        return full_data_lst

    def construct_dialog(self, model, query_prompt_lst, answer_prompt_lst, turn_lst, full_data_lst):
        dialog_lst = []

        for j, query_prompt, answer_prompt, history_json, turn in enumerate(zip(query_prompt_lst, answer_prompt_lst, full_data_lst, turn_lst)):
            for i in range(turn - 1):
                # 生成指令，history的格式也需要修改的
                # history = self.process_history(history_json)
                # query_prompt = query_prompt.format(history=history)
                query, tokenizer = self.chat(model, query_prompt, history_json)
                history_json[f'query_{i+1}'] = query[0]

                # history = self.process_history(history_json)
                # answer_prompt = answer_prompt.format(history=history)
                answer, tokenizer = self.chat(model, answer_prompt, history_json)
                history_json[f'answer_{i+1}'] = answer
            
            # 更新一下
            template = tokenizer.apply_chat_template(history_json, tokenize=False, add_generation_prompt=False)
            full_data_lst[j] = history_json
            dialog_lst.append(template)
        
        return full_data_lst, dialog_lst
    
    def evaluate_dialog(self, model, full_data_lst, dialog_lst, evaluate_prompt_lst):
        # filter_data, data_with_filter_reason = [], []
        def parse_reasult(text):
            try:
                text = json.loads(text)
                score, reason = text['rate'], text['reason']
            except:
                score, reason = -1, text
            return score, reason

        score_lst, reason_lst = [], []
        for i, data, eval_prompt in enumerate(zip(dialog_lst, evaluate_prompt_lst)):
            eval_prompt.format(history=data)
            eval_result = self.chat(model, eval_prompt)
            score, reason = parse_reasult(eval_result)
            score_lst.append(score)
            reason_lst.append(reason)

        data_json_lst = [json.dumps(data_json) for data_json in full_data_lst]
        final_df = pd.DataFrame({
            'data_json': data_json_lst,
            'dialog': dialog_lst,
            'score': score_lst ,
            'reason': reason_lst
        })

        return final_df
    
    def construct_dialogs(self, final_prompt_lst: List[dict], model_name):
        # 1.加载模型
        model = self.load_model(model_name)
        query_prompt_lst, answer_prompt_lst, evaluate_prompt_lst, turn_lst = final_prompt_lst['query_prompt'], final_prompt_lst['answer_prompt'], final_prompt_lst['evaluate_prompt'], final_prompt_lst['turn_lst']

        # 2.构造首轮对话数据
        full_data_lst = self.construct_first_dialog(model, query_prompt_lst, answer_prompt_lst)

        # 3.迭代构造后几轮对话
        full_data_lst, dialog_lst = self.construct_dialog(model, query_prompt_lst, answer_prompt_lst, turn_lst, full_data_lst)

        # 4.进行评估
        final_df = self.evaluate_dialog(model, full_data_lst, dialog_lst, evaluate_prompt_lst)

        # 5.返回评估的结果
        return final_df
    
    # 多种指标进行筛选（看论文）
    def filter_dialogs(self, final_df):
        pass
    
