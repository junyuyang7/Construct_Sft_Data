import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Server.ModelBase import AskModel, JudgeModel, AnswerModel, TopicModel
from Server.model_workers.base import LLMModelBase, Args
from typing import *
import json
import pandas as pd
from tqdm import tqdm

from Server.data_base.kb_service import RawDialogDBService

class DataConstructer:
    def __init__(self, llm_model: LLMModelBase):
        self.llm_model = llm_model

    def load_model(self, model_name):
        return 'test_load_model'

    def chat(self, first_query_prompt_lst=None, 
                    querys_lst = None, 
                    query_prompt=None, 
                    answer_prompt=None, 
                    evaluate_prompt=None, 
                    turn_lst=None, 
                    first_dialog_lst=None, 
                    full_dialog_lst=None, 
                    mode='first_query'):
        
        if mode == 'first_query':
            # output = [[], [], []]
            output = self.llm_model.generate_first_query(first_query_prompt_lst)
            
        # output = []
        elif mode == 'first_answer':
            output = self.llm_model.generate_first_answer(querys_lst, answer_prompt) 
            
        elif mode == 'mt':
            output = self.llm_model.generate_mt(query_prompt, answer_prompt, turn_lst, first_dialog_lst)
            
        elif mode == 'eval':
            output = self.llm_model.generate_eval(evaluate_prompt, full_dialog_lst)
            
        return output
    
    def construct_first_query(self, final_prompt):
        first_query_prompt_lst = final_prompt['first_query_prompt']

        first_querys_lst = self.chat(
            first_query_prompt_lst=first_query_prompt_lst, 
            mode='first_query')
        
        return first_querys_lst

    def construct_first_answer(self, querys_lst, answer_prompt):
        # first_dialog = [[{}, {}], [{}, {}, ...], ...]
        first_dialog_lst = self.chat(
            querys_lst=querys_lst, 
            answer_prompt=answer_prompt,
            mode='first_answer')
        
        return first_dialog_lst

    def construct_mt_dialog(self, query_prompt_lst, answer_prompt_lst, turn_lst, first_dialog_lst):
        full_dialog = self.chat(
            query_prompt=query_prompt_lst, 
            answer_prompt=answer_prompt_lst, 
            turn_lst=turn_lst, 
            first_dialog_lst=first_dialog_lst,
            mode='mt')
        
        return full_dialog
    
    def evaluate_dialog(self, evaluate_prompt, full_dialog_lst):
        # filter_data, data_with_filter_reason = [], []
        def parse_reasult(text_lst):
            score_lst, reason_lst = [], []
            for text in text_lst:
                text: str
                text.removeprefix("```json").removesuffix("```")
                try:
                    text = json.loads(text)
                    score, reason = text['rate'], text['reason']
                except:
                    score, reason = -1, text
                score_lst.append(score)
                reason_lst.append(reason)
                
            return score_lst, reason_lst

        eval_output, dialog_lst = self.chat(
            evaluate_prompt=evaluate_prompt,
            full_dialog_lst=full_dialog_lst,
            mode='eval'
        )
        score_lst, reason_lst = parse_reasult(eval_output)

        final_df = pd.DataFrame({
            'full_dialog': full_dialog_lst,
            'dialog': dialog_lst,
            'score': score_lst ,
            'reason': reason_lst
        })

        return final_df
    
    def construct_dialogs(self, final_prompt_lst: List[dict]):
        '''
        final_prompt_lst sample:
        [
        {
        'first_query_prompt': 'xxx',
        'query_prompt': 'xxx',
        'answer_prompt': 'xxx',
        'evaluate_prompt': 'xxx',
        'turn_lst': [5,2,3,4],
        'chat': [
              [{'role': 'user', 'content': 'xxxx'}, {'role': 'assistant', 'content': 'xxxx'}, ...],
              [[{'role': 'user', 'content': 'xxxx'}, {'role': 'assistant', 'content': 'xxxx'}, ...],
              [[{'role': 'user', 'content': 'xxxx'}, {'role': 'assistant', 'content': 'xxxx'}, ...],
              [[{'role': 'user', 'content': 'xxxx'}, {'role': 'assistant', 'content': 'xxxx'}, ...]
            ],
        },
        {
        'first_query_prompt': 'xxx',
        'query_prompt': 'xxx',
        'answer_prompt': 'xxx',
        'evaluate_prompt': 'xxx',
        'turn_lst': [5,2,3,4],
        'chat': [
              [{'role': 'user', 'content': 'xxxx'}, {'role': 'assistant', 'content': 'xxxx'}, ...],
              [[{'role': 'user', 'content': 'xxxx'}, {'role': 'assistant', 'content': 'xxxx'}, ...],
              [[{'role': 'user', 'content': 'xxxx'}, {'role': 'assistant', 'content': 'xxxx'}, ...],
              [[{'role': 'user', 'content': 'xxxx'}, {'role': 'assistant', 'content': 'xxxx'}, ...]
            ],
        }
        ]
        '''
        final_df_lst = []
        
        
        for i, final_prompt in enumerate(final_prompt_lst):
            query_prompt_lst, answer_prompt_lst, evaluate_prompt_lst, turn_lst = \
                final_prompt['query_prompt'], final_prompt['answer_prompt'], final_prompt['evaluate_prompt'], final_prompt['turn_lst']
                
            final_df_all = pd.DataFrame({
                'full_dialog': [],
                'dialog': [],
                'score': [],
                'reason': []
            })
                
            # 1.构造首轮对话 query
            first_querys_lst = self.construct_first_query(final_prompt)
            
            for first_querys, qp, ap, ep, turns in zip(first_querys_lst, query_prompt_lst, answer_prompt_lst, evaluate_prompt_lst, turn_lst):
                # 2.构造首轮对话 answer
                first_dialog_lst = self.construct_first_answer(first_querys, ap)
                
                # 2.迭代构造后几轮对话
                full_dialog_lst = self.construct_mt_dialog(qp, ap, turns, first_dialog_lst)
                final_prompt_lst[i]['chat'] = full_dialog_lst
                
                # # 3.进行评估
                # final_df = self.evaluate_dialog(ep, full_dialog_lst)
                # final_df_all = pd.concat([final_df_all, final_df], ignore_index=True)
                
            final_df_lst.append(final_df_all)

        # 储存最初数据的结果
        self.save_raw_dialog(final_prompt_lst)

        # 5.返回评估的结果
        return final_df_lst, final_prompt_lst
    
    def save_raw_dialog(self, final_prompt_lst):
        chat_list, turn_list = [], []
        db_seveice = RawDialogDBService(tabel_name='raw_dialog')
        for final_prompt in final_prompt_lst:
            for chat, turn in tqdm(zip(final_prompt['chat'], final_prompt['turn_lst']), total=len(final_prompt)):
                chat_list.append(chat)
                turn_list.append(turn)
                status = db_seveice.insert_data(chat, turn)
                if not status:
                    raise "储存 raw_dialog 出现错误"
        
        print("raw data 储存成功")

    
    # 多种指标进行筛选（看论文）
    def filter_dialogs(self, final_df_lst):
        # 筛选的方法
        return final_df_lst
    

if __name__ == '__main__':
    def test_dialog_construct():
        final_prompt_lst = [{
            'first_query_prompt': ['请帮我生成有关物理学-电磁学的一条用户指令，直接返回。', '请帮我生成有关文本编辑-小说的一条用户指令，直接返回。', '请帮我生成有关文本编辑-诗歌的一条用户指令，直接返回。', '请帮我生成有关建议-职业发展-简历编写的一条用户指令，直接返回。', '请帮我生成有关角色扮演-历史人物-科学家的一条用户指令，直接返回。'],
            'query_prompt': ['<|start_header_id|>user<|end_header_id|>\n\n', '<|start_header_id|>user<|end_header_id|>\n\n', '<|start_header_id|>user<|end_header_id|>\n\n', '<|start_header_id|>user<|end_header_id|>\n\n', '<|start_header_id|>user<|end_header_id|>\n\n'],
            'answer_prompt': ['{query}', '{query}', '{query}', '{query}', '{query}'],
            'evaluate_prompt': ["历史对话： {history}  你将会收到一段关于物理学-电磁学的对话，请从正确性、忠实性、互动性3个角度对这段对话进行评分并给出理由。 返回JSON格式如下： { '评分': 'xxx', '理由'：{     '正确性': 'xxx',     '忠实性': 'xxx',     '互动性': 'xxx',     } }", "历史对话： {history}  你将会收到一段关于文本编辑-小说的对话，请从正确性、忠实性、互动性3个角度对这段对话进行评分并给出理由。 返回JSON格式如下： { '评分': 'xxx', '理由'：{     '正确性': 'xxx',     '忠实性': 'xxx',     '互动性': 'xxx',     } }", "历史对话： {history}  你将会收到一段关于文本编辑-诗歌的对话，请从正确性、忠实性、互动性3个角度对这段对话进行评分并给出理由。 返回JSON格式如下： { '评分': 'xxx', '理由'：{     '正确性': 'xxx',     '忠实性': 'xxx',     '互动性': 'xxx',     } }", "历史对话： {$$history}  你将会收到一段关于建议-职业发展-简历编写的对话，请从正确性、忠实性、互动性3个角度对这段对话进行评分并给出理由。 返回JSON格式如下： { '评分': 'xxx', '理由'：{     '正确性': 'xxx',     '忠实性': 'xxx',     '互动性': 'xxx',     } }", "历史对话： {history}  你将会收到一段关于角色扮演-历史人物-科学家的对话，请从正确性、忠实性、互动性3个角度对这段对话进行评分并给出理由。 返回JSON格式如下： { '评分': 'xxx', '理由'：{     '正确性': 'xxx',     '忠实性': 'xxx',     '互动性': 'xxx',     } }"],
            'turn': [5, 2, 3, 4, 5],
            'chat': []
        },{
            'first_query_prompt': ['请帮我生成有关物理学-光学的一条用户指令，直接返回。', '请帮我生成有关文本编辑-散文的一条用户指令，直接返回。', '请帮我生成有关文本编辑-剧本的一条用户指令，直接返回。', '请帮我生成有关建议-职业发展-面试技巧的一条用户指令，直接返回。', '请帮我生成有关角色扮演-历史人物-文学家的一条用户指令，直接返回。'],
            'query_prompt': ['<|start_header_id|>user<|end_header_id|>\n\n', '<|start_header_id|>user<|end_header_id|>\n\n', '<|start_header_id|>user<|end_header_id|>\n\n', '<|start_header_id|>user<|end_header_id|>\n\n', '<|start_header_id|>user<|end_header_id|>\n\n'],
            'answer_prompt': ['{query}', '{query}', '{query}', '{query}', '{query}'],
            'evaluate_prompt': ["历史对话： {history}  你将会收到一段关于物理学-光学的对话，请从正确性、忠实性、互动性3个角度对这段对话进行评分并给出理由。 返回JSON格式如下： { '评分': 'xxx', '理由'：{     '正确性': 'xxx',     '忠实性': 'xxx',     '互动性': 'xxx',     } }", "历史对话： {history}  你将会收到一段关于文本编辑-散文的对话，请从正确性、忠实性、互动性3个角度对这段对话进行评分并给出理由。 返回JSON格式如下： { '评分': 'xxx', '理由'：{     '正确性': 'xxx',     '忠实性': 'xxx',     '互动性': 'xxx',     } }", "历史对话： {history}  你将会收到一段关于文本编辑-剧本的对话，请从正确性、忠实性、互动性3个角度对这段对话进行评分并给出理由。 返回JSON格式如下： { '评分': 'xxx', '理由'：{     '正确性': 'xxx',     '忠实性': 'xxx',     '互动性': 'xxx',     } }", "历史对话： {$$history}  你将会收到一段关于建议-职业发展-面试技巧的对话，请从正确性、忠实性、互动性3个角度对这段对话进行评分并给出理由。 返回JSON格式如下： { '评分': 'xxx', '理由'：{     '正确性': 'xxx',     '忠实性': 'xxx',     '互动性': 'xxx',     } }", "历史对话： {history}  你将会收到一段关于角色扮演-历史人物-文学家的对话，请从正确性、忠实性、互动性3个角度对这段对话进行评分并给出理由。 返回JSON格式如下： { '评分': 'xxx', '理由'：{     '正确性': 'xxx',     '忠实性': 'xxx',     '互动性': 'xxx',     } }"],
            'turn': [5, 2, 3, 4, 5],
            'chat': []
        }]
        model_name = 'llama-2-13b-chat'
        args = Args()
        llm_model = LLMModelBase(args)
        dc = DataConstructer(llm_model)
        final_df_lst, filter_prompt_lst = dc.construct_dialogs(final_prompt_lst)
        for df in final_df_lst:
            print(df)
            
    def test():
        llm_model_dict = {
            'chatglm3-6b': "/home/yangjy/Study/ChatAgent_RAG/llm_models/chatglm3-6b/",
            'test': 'test'
        }
        from dataclasses import dataclass
        @dataclass
        class Args:
            model_name: str = "chatglm3-6b"
            model_path: str = llm_model_dict[model_name]
        args = Args()
        model_name = 'test'
        args.model_path = llm_model_dict[model_name]
        print(args.model_path)
        
    test()
        