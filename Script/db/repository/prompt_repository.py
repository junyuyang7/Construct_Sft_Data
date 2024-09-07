from Script.db.models.prompt_base import AnswerPrompt, EvaluatePrompt, QueryPrompt, AllPrompt, FirstQueryPrompt
from Script.db.session import with_session
from typing import List
import pandas as pd

ModelType = {
    'answer_prompt': AnswerPrompt,
    'evaluate_prompt': EvaluatePrompt,
    'query_prompt': QueryPrompt,
    'first_query_prompt': FirstQueryPrompt,
    'all_prompt': AllPrompt,
}

class PromptAction:
    def __init__(self, prompt_type):
        self.prompt_type = prompt_type
        self.PromptModel = ModelType[prompt_type]
    
    @with_session
    def add_prompt_to_db(session, self, domain_name, task_name, cls_name, args, first_query_args, query_args, answer_args, evaluate_args, first_query_prompt=None, query_prompt=None, answer_prompt=None, evaluate_prompt=None, prompt=None):
        '''创建/更新知识库实例加入数据库'''
        # 构造查询条件
        filters = [
            self.PromptModel.domain_name == domain_name,
            self.PromptModel.task_name == task_name,
            self.PromptModel.cls_name == cls_name,
        ]

        if self.prompt_type == 'all_prompt':
            filters.extend([
                self.PromptModel.first_query_prompt == first_query_prompt,
                self.PromptModel.query_prompt == query_prompt,
                self.PromptModel.answer_prompt == answer_prompt,
                self.PromptModel.evaluate_prompt == evaluate_prompt,
                self.PromptModel.first_query_args == first_query_args,
                self.PromptModel.query_args == query_args,
                self.PromptModel.answer_args == answer_args,
                self.PromptModel.evaluate_args == evaluate_args
            ])
        else:
            filters.extend([
                self.PromptModel.prompt == prompt,
                self.PromptModel.args == args
            ])

        # 查找现有的记录
        existing_prompt = session.query(self.PromptModel).filter(*filters).first()

        # 如果记录不存在，创建新记录
        if not existing_prompt:
            if self.prompt_type == 'all_prompt':
                new_prompt = self.PromptModel(
                    domain_name=domain_name,
                    task_name=task_name,
                    cls_name=cls_name,
                    first_query_prompt=first_query_prompt,
                    query_prompt=query_prompt,
                    answer_prompt=answer_prompt,
                    evaluate_prompt=evaluate_prompt,
                    first_query_args=first_query_args,
                    query_args=query_args,
                    answer_args=answer_args,
                    evaluate_args=evaluate_args,
                )
            else:
                new_prompt = self.PromptModel(
                    domain_name=domain_name,
                    task_name=task_name,
                    cls_name=cls_name,
                    prompt=prompt,
                    args=args
                )
            session.add(new_prompt)
        else:
            # 如果记录存在，更新其内容
            existing_prompt.domain_name = domain_name
            existing_prompt.task_name = task_name
            existing_prompt.cls_name = cls_name
            if self.prompt_type == 'all_prompt':
                existing_prompt.first_query_prompt = first_query_prompt
                existing_prompt.query_prompt = query_prompt
                existing_prompt.answer_prompt = answer_prompt
                existing_prompt.evaluate_prompt = evaluate_prompt
                existing_prompt.first_query_args = first_query_args
                existing_prompt.query_args = query_args
                existing_prompt.answer_args = answer_args
                existing_prompt.evaluate_args = evaluate_args
            else:
                existing_prompt.prompt = prompt
                existing_prompt.args = args
            
        return True

    @with_session
    def list_prompts_from_db(session, self) -> List:
        '''列出数据库中含有的prompt'''
        prompts = session.query(self.PromptModel).all()
        prompts_dict_list = [prompt.__dict__ for prompt in prompts]

        # 删除字典中的 _sa_instance_state，这是SQLAlchemy的内部属性
        for item in prompts_dict_list:
            item.pop('_sa_instance_state', None)
        # df = pd.DataFrame(prompts_dict_list)
        # print(df)
        return prompts_dict_list

    @with_session
    def prompt_exists(session, self, prompt):
        '''判断prompt存不存在'''
        if self.prompt_type == 'all_prompt':
            prompt_tmp = session.query(self.PromptModel).filter(self.PromptModel.query_prompt.ilike(prompt)).all()
        else:
            prompt_tmp = session.query(self.PromptModel).filter(self.PromptModel.prompt.ilike(prompt)).first()
        status = True if prompt_tmp else False
        return status

    @with_session
    def find_prompt_from_keyword(session, self, keyword):
        '''根据关键字搜索对应的prompt'''
        if self.prompt_type == 'all_prompt':
            prompt_tmp = session.query(self.PromptModel).filter(self.PromptModel.query_prompt.ilike(f"%{keyword}%")).all()
        else:
            prompt_tmp = session.query(self.PromptModel).filter(self.PromptModel.prompt.ilike(f"%{keyword}%")).all()
        if prompt_tmp:
            prompts_dict_list = [prompt.__dict__ for prompt in prompt_tmp]
            return True, prompts_dict_list
        return False, {}

    @with_session
    def delete_prompt_from_db(session, self, prompt_id):
        '''从数据库中删除对应prompt'''
        prompt = session.query(self.PromptModel).filter(self.PromptModel.id == prompt_id).first()
        if prompt:
            session.delete(prompt)
        return True

    @with_session
    def update_prompt_from_db(session, self, prompt_id, **kwargs):
        '''从数据库中删除对应prompt'''
        prompt = session.query(self.PromptModel).filter(self.PromptModel.id == prompt_id).first()
        if prompt:
            for key, value in kwargs.items():
                if hasattr(prompt, key):
                    setattr(prompt, key, value)
            session.commit()
            return True
        return False
