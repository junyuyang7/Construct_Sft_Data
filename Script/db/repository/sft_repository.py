from Script.db.models.sft_data_base import SFTDataModel
from Script.db.session import with_session
from sqlalchemy.orm import Session
from typing import List

@with_session
def add_history_to_db(session: Session, inputs, targets, turn, domain_name, task_name, cls_name, prompt, history, score):
    '''创建/更新知识库实例加入数据库'''
    existing_prompt = session.query(SFTDataModel).filter(
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
    if not existing_prompt:
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
        existing_prompt.inputs == inputs,
        existing_prompt.targets == targets,
        existing_prompt.turn == turn,
        existing_prompt.domain_name == domain_name,
        existing_prompt.task_name == task_name,
        existing_prompt.cls_name == cls_name,
        existing_prompt.prompt == prompt
        existing_prompt.history == history
        existing_prompt.score == score
    return True

@with_session
def list_history_from_db(session) -> List:
    '''列出数据库中含有的prompt'''
    return session.query(SFTDataModel).all()

@with_session
def history_exists(session, history):
    '''判断prompt存不存在'''
    prompt_tmp = session.query(SFTDataModel).filter(SFTDataModel.prompt.ilike(history)).first()
    status = True if prompt_tmp else False
    return status

@with_session
def delete_history_from_db(session, history):
    '''从数据库中删除对应prompt'''
    his = session.query(SFTDataModel).filter(SFTDataModel.history == history).first()
    if his:
        session.delete(his)
    return True

# @with_session
# def get_kb_detail(session, kb_name: str) -> dict:
#     '''获取知识库的详细信息'''
#     kb: SFTDataModel = session.query(SFTDataModel).filter(SFTDataModel.id.ilike(kb_name)).first()
#     if kb:
#         return {
#             "kb_name": kb.kb_name,
#             "kb_info": kb.kb_info,
#             "vs_type": kb.vs_type,
#             "embed_model": kb.embed_model,
#             "file_count": kb.file_count,
#             "create_time": kb.create_time,
#         }
#     else:
#         return {}
