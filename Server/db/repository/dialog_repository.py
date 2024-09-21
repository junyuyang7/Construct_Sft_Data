from Server.db.models.dialog_base import RawDialogModel
from Server.db.session import with_session
from sqlalchemy.orm import Session
from typing import List

@with_session
def add_raw_data_to_db(session: Session, message, turn):
    '''创建/更新知识库实例加入数据库'''
    existing_data = session.query(RawDialogModel).filter(
        RawDialogModel.message == message,
        RawDialogModel.turn == turn,
    ).first()
    if not existing_data:
        new_prompt = RawDialogModel(
            message=message,
            turn=turn,
        )
        session.add(new_prompt)
    else: # 如果已经存在就进行更新即可
        existing_data.message == message,
        existing_data.turn == turn
    return True

@with_session
def list_raw_data_from_db(session) -> List:
    '''列出数据库中所有的 sft 数据'''
    return session.query(RawDialogModel).all()

@with_session
def find_raw_data_from_keyword(session, keyword):
    '''根据关键字搜索对应的 sft 数据'''
    history_tmp = session.query(RawDialogModel).filter(RawDialogModel.message.ilike(f"%{keyword}%")).first()
    status = True if history_tmp else False
    return status, history_tmp

def test():
    pass

@with_session
def delete_raw_data_from_db(session, idx):
    '''从数据库中删除对应 sft 数据'''
    his = session.query(RawDialogModel).filter(RawDialogModel.id == idx).first()
    if his:
        session.delete(his)
    return True

@with_session
def update_raw_data_from_db(session, idx, **kwargs):
    '''从数据库中删除对应prompt'''
    his = session.query(RawDialogModel).filter(RawDialogModel.id == idx).first()
    if his:
        for key, value in kwargs.items():
            if hasattr(his, key):
                setattr(his, key, value)
        session.commit()
        return True
    return False