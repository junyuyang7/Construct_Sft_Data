from Script.db.models.prompt_base import PromptModel
from Script.db.session import with_session
from typing import List
import pandas as pd

@with_session
def add_prompt_to_db(session, domain_name, task_name, cls_name, model_type, prompt, args):
    '''创建/更新知识库实例加入数据库'''
    existing_prompt = session.query(PromptModel).filter(
        PromptModel.domain_name == domain_name,
        PromptModel.task_name == task_name,
        PromptModel.cls_name == cls_name,
        PromptModel.model_type == model_type,
        PromptModel.prompt == prompt,
        PromptModel.args == args
    ).first()
    if not existing_prompt:
        new_prompt = PromptModel(
            domain_name=domain_name,
            task_name=task_name,
            cls_name=cls_name,
            model_type=model_type,
            prompt=prompt,
            args=args
        )
        session.add(new_prompt)
    else: # 如果已经存在就进行更新即可
        existing_prompt.domain_name == domain_name,
        existing_prompt.task_name == task_name,
        existing_prompt.cls_name == cls_name,
        existing_prompt.model_type == model_type,
        existing_prompt.prompt == prompt,
        existing_prompt.args == args
    return True

@with_session
def list_prompts_from_db(session) -> List:
    '''列出数据库中含有的prompt'''
    prompts = session.query(PromptModel).all()
    prompts_dict_list = [prompt.__dict__ for prompt in prompts]

    # 删除字典中的 _sa_instance_state，这是SQLAlchemy的内部属性
    for item in prompts_dict_list:
        item.pop('_sa_instance_state', None)
    # df = pd.DataFrame(prompts_dict_list)
    # print(df)
    return prompts_dict_list

@with_session
def prompt_exists(session, prompt):
    '''判断prompt存不存在'''
    prompt_tmp = session.query(PromptModel).filter(PromptModel.prompt.ilike(prompt)).first()
    status = True if prompt_tmp else False
    return status

@with_session
def find_prompt_from_keyword(session, keyword):
    '''根据关键字搜索对应的prompt'''
    prompt_tmp = session.query(PromptModel).filter(PromptModel.prompt.ilike(f"%{keyword}%")).all()
    if prompt_tmp:
        prompts_dict_list = [prompt.__dict__ for prompt in prompt_tmp]
        return True, prompts_dict_list
    return False, {}

@with_session
def delete_prompt_from_db(session, prompt_id):
    '''从数据库中删除对应prompt'''
    prompt = session.query(PromptModel).filter(PromptModel.id == prompt_id).first()
    if prompt:
        session.delete(prompt)
    return True

@with_session
def update_prompt_from_db(session, prompt_id, **kwargs):
    '''从数据库中删除对应prompt'''
    prompt = session.query(PromptModel).filter(PromptModel.id == prompt_id).first()
    if prompt:
        for key, value in kwargs.items():
            if hasattr(prompt, key):
                setattr(prompt, key, value)
        session.commit()
        return True
    return False
