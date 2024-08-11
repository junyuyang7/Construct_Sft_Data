import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import sqlite3
from sqlalchemy import create_engine, and_, or_, MetaData, text
from sqlalchemy.orm import sessionmaker
from Script.db.models.prompt_base import PromptModel
from Script.db.models.sft_data_base import SFTDataModel
from sqlalchemy.exc import SQLAlchemyError

ModelType = {
    'promptmodel': PromptModel,
    'sftdatamodel': SFTDataModel
}

class DBconnecter:
    def __init__(self) -> None:
        pass

    @staticmethod
    def create_table(model_type):
        # 创建 SQLite 数据库（在内存中）
        engine = create_engine('sqlite:///test.db', echo=True)

        # 创建所有定义的表
        ModelType[model_type].metadata.create_all(engine)

        # 创建一个会话
        Session = sessionmaker(bind=engine)
        session = Session()

        # 关闭会话
        session.close()

    @staticmethod
    def recreate_table(model_type):
        # 创建 SQLite 数据库连接
        engine = create_engine('sqlite:///test.db', echo=True)
        
        # 获取模型定义
        model = ModelType[model_type]
        
        # 删除旧表（如果存在）
        try:
            with engine.connect() as connection:
                if engine.dialect.has_table(connection, model.__tablename__):
                    connection.execute(text(f'DROP TABLE IF EXISTS {model.__tablename__}'))
                    print(f"Table '{model.__tablename__}' dropped.")
                else:
                    print(f"Table '{model.__tablename__}' does not exist.")
        except SQLAlchemyError as e:
            print(f"Error dropping table: {e}")

        # 创建新的表
        try:
            model.metadata.create_all(engine)
            print(f"Table '{model.__tablename__}' created.")
        except SQLAlchemyError as e:
            print(f"Error creating table: {e}")

        # 创建一个会话
        Session = sessionmaker(bind=engine)
        session = Session()

        # 关闭会话
        session.close()

    @staticmethod
    def insert_prompt(session, domain_name, task_name, cls_name, model_type, prompt, args):
        # 检查是否存在重复记录（排除id）
        existing_prompt = session.query(PromptModel).filter(
            PromptModel.domain_name == domain_name,
            PromptModel.task_name == task_name,
            PromptModel.cls_name == cls_name,
            PromptModel.model_type == model_type,
            PromptModel.prompt == prompt,
            PromptModel.args == args
        ).first()
        
        if existing_prompt:
            # 记录已经存在，进行提醒
            print("Error: A record with the same values already exists.")
            return existing_prompt  # 或者你可以返回现有记录的ID或其他信息
        
        # 如果没有重复，则插入新记录
        new_prompt = PromptModel(
            domain_name=domain_name,
            task_name=task_name,
            cls_name=cls_name,
            model_type=model_type,
            prompt=prompt,
            args=args
        )
        session.add(new_prompt)
        session.commit()
        return new_prompt.id  # 返回新创建记录的ID

    @staticmethod
    def get_prompt_by_id(session, prompt_id):
        return session.query(PromptModel).filter(PromptModel.id == prompt_id).first()
    
    @staticmethod
    def get_prompt_by_domain_task_cls_type(session, domain_name, task_name, cls_name, model_type):
        prompt_data = session.query(PromptModel).filter(
            and_(
            PromptModel.domain_name == domain_name,
            PromptModel.task_name == task_name,
            PromptModel.cls_name == cls_name,
            PromptModel.model_type == model_type
        )
        ).first()
        if prompt_data:
        # 将SQLAlchemy对象转换为字典
            prompt_dict = {
                "id": prompt_data.id,
                "domain_name": prompt_data.domain_name,
                "task_name": prompt_data.task_name,
                "cls_name": prompt_data.cls_name,
                "model_type": prompt_data.model_type,
                "prompt": prompt_data.prompt,
                "args": prompt_data.args
            }
            return prompt_dict
        else:
            print("Error: the prompt don't exist.")
            return None  # 如果未找到记录，返回None

    @staticmethod
    def get_all_prompts(session, ):
        return session.query(PromptModel).all()

    @staticmethod
    def update_prompt(session, prompt_id, **kwargs):
        prompt = session.query(PromptModel).filter(PromptModel.id == prompt_id).first()
        if prompt:
            for key, value in kwargs.items():
                if hasattr(prompt, key):
                    setattr(prompt, key, value)
            session.commit()
            return True
        return False

    @staticmethod
    def delete_prompt(session, prompt_id):
        prompt = session.query(PromptModel).filter(PromptModel.id == prompt_id).first()
        if prompt:
            session.delete(prompt)
            session.commit()
            return True
        return False

if __name__ == '__main__':
    # 创建数据库连接
    db = DBconnecter()
    # engine = create_engine('sqlite:///test.db', echo=True)

    # # 创建会话
    # Session = sessionmaker(bind=engine)
    # session = Session()

    # import pandas as pd
    # df = pd.read_excel('Script/test.xlsx')

    # for _, row in df.iterrows():
    #     new_id = db.insert_prompt(
    #         session,
    #         domain_name=row['domain_name'],
    #         task_name=row['task_name'],
    #         cls_name=row['cls_name'],
    #         model_type=row['model_type'],
    #         prompt=row['prompt'],
    #         args=row['args']
    #     )

    # session.close()
    db.recreate_table('sftdatamodel')

