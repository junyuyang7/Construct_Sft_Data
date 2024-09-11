from Server.db.models.sft_data_base import SFTDataModel
from Server.db.session import with_session
from sqlalchemy.orm import Session
from typing import List

@with_session
def add_history_to_db(session: Session, inputs, targets, turn, domain_name, task_name, cls_name, prompt, history, score):
    '''创建/更新知识库实例加入数据库'''
    existing_data = session.query(SFTDataModel).filter(
        SFTDataModel.inputs == inputs,
        SFTDataModel.targets == targets,
        SFTDataModel.turn == turn,
        SFTDataModel.domain_name == domain_name,
        SFTDataModel.task_name == task_name,
        SFTDataModel.cls_name == cls_name,
        SFTDataModel.prompt == prompt,
        SFTDataModel.history == history,
        SFTDataModel.score == score,
    ).first()
    if not existing_data:
        new_prompt = SFTDataModel(
            inputs=inputs,
            targets=targets,
            turn=turn,
            domain_name=domain_name,
            task_name=task_name,
            cls_name=cls_name,
            prompt=prompt,
            history=history,
            score=score
        )
        session.add(new_prompt)
    else: # 如果已经存在就进行更新即可
        existing_data.inputs == inputs,
        existing_data.targets == targets,
        existing_data.turn == turn,
        existing_data.domain_name == domain_name,
        existing_data.task_name == task_name,
        existing_data.cls_name == cls_name,
        existing_data.prompt == prompt
        existing_data.history == history
        existing_data.score == score
    return True

@with_session
def list_history_from_db(session) -> List:
    '''列出数据库中含有的prompt'''
    return session.query(SFTDataModel).all()

@with_session
def history_exists(session, history):
    '''判断prompt存不存在'''
    prompt_tmp = session.query(SFTDataModel).filter(SFTDataModel.history.ilike(history)).first()
    status = True if prompt_tmp else False
    return status

@with_session
def find_history_from_keyword(session, keyword):
    '''根据关键字搜索对应的prompt'''
    history_tmp = session.query(SFTDataModel).filter(SFTDataModel.history.ilike(f"%{keyword}%")).first()
    status = True if history_tmp else False
    return status, history_tmp

def test():
    pass

@with_session
def delete_history_from_db(session, history):
    '''从数据库中删除对应prompt'''
    his = session.query(SFTDataModel).filter(SFTDataModel.history == history).first()
    if his:
        session.delete(his)
    return True

@with_session
def update_history_from_db(session, prompt_id, **kwargs):
    '''从数据库中删除对应prompt'''
    his = session.query(SFTDataModel).filter(SFTDataModel.id == prompt_id).first()
    if his:
        for key, value in kwargs.items():
            if hasattr(his, key):
                setattr(his, key, value)
        session.commit()
        return True
    return False