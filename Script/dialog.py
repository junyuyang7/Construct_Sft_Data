from Script.DataConstructer import DataConstructer
from Script.ModelBase import AskModel, AnswerModel, JudgeModel, TopicModel
from Script.create_db import DBconnecter
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import json
from Script.db.repository.sft_repository import (add_history_to_db, list_history_from_db, history_exists, delete_history_from_db)
from Script.db.repository.prompt_repository import (add_prompt_to_db, list_prompts_from_db, prompt_exists, delete_prompt_from_db)

class DialogConstruct(DataConstructer):
    def __init__(self, topic_model, ask_model, answer_model, judge_model):
        super().__init__(topic_model, ask_model, answer_model, judge_model)
        self.db = DBconnecter()
        engine = create_engine('sqlite:///test.db', echo=True)
        Session = sessionmaker(bind=engine)
        self.session = Session()

    # def construct_new_prompt(self):


    def get_prompt(self, domain_name, task_name, cls_name, model_type) -> dict:
        prompt = self.db.get_prompt_by_domain_task_cls_type(self.session, domain_name, task_name, cls_name, model_type)
        try:
            return prompt
        except Exception as e:
            raise "模板库中不存在该类型的Prompt"
        
    def it2history(self):
        history = ""
        for query, answer in zip(self.inputs, self.targets):
            history += "[HUMAN]: " + query + '\n'
            history += "[AI]: " + answer + '\n'
        return history

    def it2message(self):
        message = []
        for query, answer in zip(self.inputs, self.targets):
            message.append({'role': 'user', 'content': query})
            message.append({'role': 'assistant', 'content': answer})
        return message

    def construct_one(self, domain_name, task_name, cls_name, score_threhold=5, turn=2, construct_myself=False):
        # 储存在inputs和targets中
        self.inputs = []
        self.targets = []
        self.turn = turn

        # 获取需要的prompt
        self.topic_prompt = self.get_prompt(domain_name, task_name, cls_name, model_type='Topic Model')
        self.ask_prompt = self.get_prompt(domain_name, task_name, cls_name, model_type='Ask Model')
        self.answer_prompt = self.get_prompt(domain_name, task_name, cls_name, model_type='Answer Model')
        self.judge_prompt = self.get_prompt(domain_name, task_name, cls_name, model_type='Judge Model')

        # 自己创建每一步的prompt
        # if construct_myself:
        #     self.topic_prompt, self.ask_prompt, self.answer_prompt, self.judge_prompt = self.construct_new_prompt()
        
        # 构建用户query
        query = self.get_query_from_topic(topic=domain_name, prompt=self.topic_prompt)
        self.inputs.append(query.strip())
        # 构建query的答案
        answer = self.get_answer_from_query(prompt=self.answer_prompt, query=query)
        self.targets.append(answer.strip())

        # 根据给定轮数重复进行构造
        for _ in range(1, turn):
            query = self.get_query_from_answer(prompt=self.ask_prompt, answer=answer)
            self.inputs.append(query.strip())
            answer = self.get_answer_from_query(prompt=self.answer_prompt, query=query)
            self.targets.append(answer.strip())

        # 进行质量打分并返回（设定阈值进行筛选）
        history = self.it2history()
        score = self.judge_dialog(prompt=self.judge_prompt, dialog=history)
        return history, score

    def construct_all(self, task_types):
        for domain_name, task_name, cls_name in task_types:
            history, score = self.construct_one(domain_name, task_name, cls_name)
            prompt = {'topic_prompt': self.topic_prompt,
                      'ask_prompt': self.ask_prompt,
                      'answer_prompt': self.answer_prompt,
                      'judge_prompt': self.judge_prompt}
            prompt = json.dumps(prompt, ensure_ascii=False)
            inputs = json.dumps(self.inputs, ensure_ascii=False)
            targets = json.dumps(self.targets, ensure_ascii=False)
            add_history_to_db(inputs=inputs,
                              targets=targets,
                              turn=self.turn,
                              domain_name=domain_name,
                              task_name=task_name,
                              cls_name=cls_name,
                              prompt=prompt,
                              history=history,
                              score=score,)
            
if __name__ == '__main__':
    topic_model = TopicModel()
    ask_model = AskModel()
    answer_model = AnswerModel()
    judge_model = JudgeModel()
    dc = DialogConstruct(topic_model=topic_model, ask_model=ask_model, answer_model=answer_model, judge_model=judge_model)
    task_types = [('知识问答', '生物', '动物学')]
    dc.construct_all(task_types)


