import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import sqlite3
from abc import ABC, abstractmethod
from sqlalchemy import create_engine, and_, or_, MetaData, text, inspect
from sqlalchemy.orm import sessionmaker
from Script.db.models import PromptModel, SFTDataModel
from Script.db.repository import (add_prompt_to_db, list_prompts_from_db, find_prompt_from_keyword, prompt_exists, delete_prompt_from_db, update_prompt_from_db, add_history_to_db, list_history_from_db, find_history_from_keyword, history_exists, delete_history_from_db, update_history_from_db)
from sqlalchemy.exc import SQLAlchemyError
from utils import get_db_path
from Script.config import SQLALCHEMY_DATABASE_URI

ModelType = {
    'prompt_model': PromptModel,
    'sftdata_model': SFTDataModel
}

table_lists = ['prompt_model', 'sftdata_model']

db_excutor = {
    'prompt_model': {
        'add': add_prompt_to_db,
        'list_all': list_prompts_from_db,
        'find': find_prompt_from_keyword,
        'delete': delete_prompt_from_db,
        'update': update_prompt_from_db
    },
    'sftdata_model': {
        'add': add_history_to_db,
        'list_all': list_history_from_db,
        'find': find_history_from_keyword,
        'delete': delete_history_from_db,
        'update': update_history_from_db
    }
}

class DBService(ABC):
    '''
    基类：创建 / 增删改查数据库 
    保存向量库 / 创建知识库 / 删除数据库内容 / 添加文件 / 更新知识库 以及各种抽象方法接口
    '''
    def __init__(self,
                 table_name: str,
                 ):
        self.table_name = table_name
        # self.kb_name = knowledge_base_name
        # self.db_path = get_db_path(self.kb_name)
        self.engine = self.create_db_engine()
        self.do_init()

    def create_db_engine(self):
        '''创建数据库引擎'''
        # db_url = f'sqlite:///{self.db_path}'
        engine = create_engine(SQLALCHEMY_DATABASE_URI, echo=True)
        return engine

    def create_kb(self):
        '''创建数据库中的表'''
        ModelType[self.table_name].metadata.create_all(self.engine)

        Session = sessionmaker(bind=self.engine)
        session = Session()
        session.close()
        print(f'{self.table_name} 创建成功')

    def recreate_table(self):
        '''重建数据库中的表'''
        model = ModelType[self.table_name]
        # 删除旧表（如果存在）
        try:
            with self.engine.connect() as connection:
                if self.engine.dialect.has_table(connection, model.__tablename__):
                    connection.execute(text(f'DROP TABLE IF EXISTS {model.__tablename__}'))
                    print(f"Table '{model.__tablename__}' dropped.")
                else:
                    print(f"Table '{model.__tablename__}' does not exist.")
        except SQLAlchemyError as e:
            print(f"{self.table_name} 表不存在: {e}")

        # 创建新的表
        try:
            model.metadata.create_all(self.engine)
            print(f"{self.table_name} 重建成功")
        except SQLAlchemyError as e:
            print(f"{self.table_name} 重建失败: {e}")

        Session = sessionmaker(bind=self.engine)
        session = Session()
        session.close()

    def insert_data(self, domain_name, task_name, cls_name, model_type, prompt, args):
        # 检查是否存在重复记录（排除id）
        status = db_excutor[self.table_name]['add'](domain_name, task_name, cls_name, model_type, prompt, args)
        return status
    
    def get_data_by_keyword(self, keyword):
        status, prompt_dict = db_excutor[self.table_name]['find'](keyword)
        return status, prompt_dict

    def get_all_data(self):
        resp = db_excutor[self.table_name]['list_all']()
        return resp

    def update_data(self, prompt_id, **kwargs):
        status = db_excutor[self.table_name]['update'](prompt_id, **kwargs)
        return status

    def delete_data(self, prompt_id):
        status = db_excutor[self.table_name]['delete'](prompt_id)
        return status
    
    def do_init(self):
        """初始化操作，子类可以扩展或重写此方法"""
        inspector = inspect(self.engine)
        if not inspector.has_table(self.table_name):
            print(f"表 {self.table_name} 并不存在，需要创建")
            self.create_kb()
        else:
            print(f"表 {self.table_name} 已存在，可继续使用")


if __name__ == '__main__':
    db = DBService(table_name='prompt_model')
    status = db.insert_prompt('test', 'test', 'test', 'test', 'test', 'test')
    print(status)
    