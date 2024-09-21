import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import sqlite3
import json
from abc import ABC, abstractmethod
from sqlalchemy import create_engine, and_, or_, MetaData, text, inspect
from sqlalchemy.orm import sessionmaker
from Server.db.models import QueryPrompt, AnswerPrompt, EvaluatePrompt, AllPrompt, SFTDataModel, FirstQueryPrompt, RawDialogModel

from Server.db.repository import (PromptAction, add_history_to_db, list_history_from_db, find_history_from_keyword, history_exists, delete_history_from_db, update_history_from_db)

from Server.db.repository import (add_raw_data_to_db, list_raw_data_from_db, find_raw_data_from_keyword, delete_raw_data_from_db, update_raw_data_from_db)

from sqlalchemy.exc import SQLAlchemyError
from utils import get_db_path
from Server.config import SQLALCHEMY_DATABASE_URI

ModelType = {
    'answer_prompt': AnswerPrompt,
    'evaluate_prompt': EvaluatePrompt,
    'first_query_prompt': FirstQueryPrompt,
    'query_prompt': QueryPrompt,
    'all_prompt': AllPrompt,
    'sftdata_model': SFTDataModel,
    'rawdata_model': RawDialogModel
}

ModelAction = {
    'answer_prompt': PromptAction(prompt_type='answer_prompt'),
    'evaluate_prompt': PromptAction(prompt_type='evaluate_prompt'),
    'first_query_prompt': PromptAction(prompt_type='first_query_prompt'),
    'query_prompt': PromptAction(prompt_type='query_prompt'),
    'all_prompt': PromptAction(prompt_type='all_prompt'),
}

# table_lists = ['answer_prompt', 'sftdata_model']

db_excutor = {
    'answer_prompt': {
        'add': ModelAction['answer_prompt'].add_prompt_to_db,
        'list_all': ModelAction['answer_prompt'].list_prompts_from_db,
        'find': ModelAction['answer_prompt'].find_prompt_from_keyword,
        'delete': ModelAction['answer_prompt'].delete_prompt_from_db,
        'update': ModelAction['answer_prompt'].update_prompt_from_db
    },
    'evaluate_prompt': {
        'add': ModelAction['evaluate_prompt'].add_prompt_to_db,
        'list_all': ModelAction['evaluate_prompt'].list_prompts_from_db,
        'find': ModelAction['evaluate_prompt'].find_prompt_from_keyword,
        'delete': ModelAction['evaluate_prompt'].delete_prompt_from_db,
        'update': ModelAction['evaluate_prompt'].update_prompt_from_db
    },
    'query_prompt': {
        'add': ModelAction['query_prompt'].add_prompt_to_db,
        'list_all': ModelAction['query_prompt'].list_prompts_from_db,
        'find': ModelAction['query_prompt'].find_prompt_from_keyword,
        'delete': ModelAction['query_prompt'].delete_prompt_from_db,
        'update': ModelAction['query_prompt'].update_prompt_from_db
    },
    'first_query_prompt': {
        'add': ModelAction['first_query_prompt'].add_prompt_to_db,
        'list_all': ModelAction['first_query_prompt'].list_prompts_from_db,
        'find': ModelAction['first_query_prompt'].find_prompt_from_keyword,
        'delete': ModelAction['first_query_prompt'].delete_prompt_from_db,
        'update': ModelAction['first_query_prompt'].update_prompt_from_db
    },
    'all_prompt': {
        'add': ModelAction['all_prompt'].add_prompt_to_db,
        'list_all': ModelAction['all_prompt'].list_prompts_from_db,
        'find': ModelAction['all_prompt'].find_prompt_from_keyword,
        'delete': ModelAction['all_prompt'].delete_prompt_from_db,
        'update': ModelAction['all_prompt'].update_prompt_from_db
    },
    'sftdata_model': {
        'add': add_history_to_db,
        'list_all': list_history_from_db,
        'find': find_history_from_keyword,
        'delete': delete_history_from_db,
        'update': update_history_from_db
    },
    'rawdata_model': {
        'add': add_raw_data_to_db,
        'list_all': list_raw_data_from_db,
        'find': find_raw_data_from_keyword,
        'delete': delete_raw_data_from_db,
        'update': update_raw_data_from_db
    }
}

class DBService(ABC):
    '''
    基类：创建 / 增删改查数据库 
    保存向量库 / 创建知识库 / 删除数据库内容 / 添加文件 / 更新知识库 以及各种抽象方法接口
    '''
    def __init__(self,
                 tabel_name: str,
                 ):
        self.tabel_name = tabel_name
        self.engine = self.create_db_engine()
        self.model = ModelType[self.tabel_name]
        self.do_init()

    def create_db_engine(self):
        '''创建数据库引擎'''
        # db_url = f'sqlite:///{self.db_path}'
        engine = create_engine(SQLALCHEMY_DATABASE_URI, echo=True)
        return engine

    def create_kb(self):
        '''创建数据库中的表'''
        self.model.metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            connection.commit()

        Session = sessionmaker(bind=self.engine)
        session = Session()
        session.close()
        print(f'{self.tabel_name} 创建成功')
        # inspector = inspect(self.engine)
        # tables = inspector.get_tabel_names()
        # print(f"数据库中存在的表: {tables}")

    def recreate_table(self):
        '''重建数据库中的表'''
        # model = ModelType[self.tabel_name]
        # 删除旧表（如果存在）
        try:
            with self.engine.connect() as connection:
                if self.engine.dialect.has_table(connection, self.model.__tablename__):
                    connection.execute(text(f'DROP TABLE IF EXISTS {self.model.__tablename__}'))
                    print(f"Table '{self.model.__tablename__}' dropped.")
                else:
                    print(f"Table '{self.model.__tablename__}' does not exist.")
        except SQLAlchemyError as e:
            print(f"{self.tabel_name} 表不存在: {e}")

        # 创建新的表
        try:
            self.model.metadata.create_all(self.engine)
            print(f"{self.tabel_name} 重建成功")
        except SQLAlchemyError as e:
            print(f"{self.tabel_name} 重建失败: {e}")

        Session = sessionmaker(bind=self.engine)
        session = Session()
        session.close()

    def insert_data(self):
        pass
    
    def get_data_by_keyword(self, keyword):
        pass

    def get_all_data(self):
        pass

    def update_data(self, prompt_id, **kwargs):
        pass

    def delete_data(self, prompt_id):
        pass
    
    def do_init(self):
        """初始化操作，子类可以扩展或重写此方法"""
        inspector = inspect(self.engine)
        if not inspector.has_table(self.tabel_name):
            print(f"表 {self.tabel_name} 并不存在，需要创建")
            self.create_kb()
        else:
            print(f"表 {self.tabel_name} 已存在，可继续使用")


if __name__ == '__main__':
    # db = DBService(tabel_name='prompt_model')
    # status = db.insert_data('test', 'test', 'test', 'test', 'test')
    # print(status)
    db = DBService(tabel_name='query_prompt')
    status = db.recreate_table()
    print(status)
    db = DBService(tabel_name='first_query_prompt')
    status = db.recreate_table()
    print(status)
    db = DBService(tabel_name='answer_prompt')
    status = db.recreate_table()
    print(status)
    db = DBService(tabel_name='evaluate_prompt')
    status = db.recreate_table()
    print(status)
    db = DBService(tabel_name='all_prompt')
    status = db.recreate_table()
    print(status)
    