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
from Script.db.models import QueryPrompt, AnswerPrompt, EvaluatePrompt, AllPrompt, SFTDataModel
from Script.db.repository import (PromptAction, add_history_to_db, list_history_from_db, find_history_from_keyword, history_exists, delete_history_from_db, update_history_from_db)
from sqlalchemy.exc import SQLAlchemyError
from utils import get_db_path
from Script.config import SQLALCHEMY_DATABASE_URI

ModelType = {
    'answer_prompt': AnswerPrompt,
    'evaluate_prompt': EvaluatePrompt,
    'query_prompt': QueryPrompt,
    'all_prompt': AllPrompt,
    'sftdata_model': SFTDataModel
}

ModelAction = {
    'answer_prompt': PromptAction(prompt_type='answer_prompt'),
    'evaluate_prompt': PromptAction(prompt_type='evaluate_prompt'),
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

    def insert_data(self, domain_name, task_name, cls_name, args, query_args, answer_args, evaluate_args, query_prompt, answer_prompt, evaluate_prompt, prompt):
        # args 格式 aaa bbb ccc ddd; abc bcd dfg
        args_json, query_json, answer_json, evaluate_json = None, None, None, None

        if self.tabel_name == 'all_prompt':
            query_args, answer_args, evaluate_args = query_args.split(';'), answer_args.split(';'), evaluate_args.split(';')

            query_json = {'input_args': query_args[0].split(),
                        'iter_args': query_args[1].split()}
            answer_json = {'input_args': answer_args[0].split(),
                        'iter_args': answer_args[1].split()}
            evaluate_json = {'input_args': evaluate_args[0].split(),
                        'iter_args': evaluate_args[1].split()}
            
            query_json, answer_json, evaluate_json = json.dumps(query_json), json.dumps(answer_json), json.dumps(evaluate_json)

        else:
            args = args.split(';')
            args_json = {'input_args': args_json[0].split(),
                     'iter_args': args_json[1].split()}
            args_json = json.dumps(args_json)

        # 检查是否存在重复记录（排除id）
        status = db_excutor[self.tabel_name]['add'](domain_name, task_name, cls_name, args_json, query_json, answer_json, evaluate_json, query_prompt, answer_prompt, evaluate_prompt, prompt)
        return status
    
    def get_data_by_keyword(self, keyword):
        status, prompt_dict = db_excutor[self.tabel_name]['find'](keyword)
        return status, prompt_dict

    def get_all_data(self):
        resp = db_excutor[self.tabel_name]['list_all']()
        return resp

    def update_data(self, prompt_id, **kwargs):
        status = db_excutor[self.tabel_name]['update'](prompt_id, **kwargs)
        return status

    def delete_data(self, prompt_id):
        status = db_excutor[self.tabel_name]['delete'](prompt_id)
        return status
    
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
    status = db.insert_data('test', 'test', 'test', 'test', 'test', 'test','test', 'test', 'test', 'test', 'test')
    print(status)
    db = DBService(tabel_name='answer_prompt')
    status = db.recreate_table()
    status = db.insert_data('test', 'test', 'test', 'test', 'test', 'test','test', 'test', 'test', 'test', 'test')
    print(status)
    db = DBService(tabel_name='evaluate_prompt')
    status = db.recreate_table()
    status = db.insert_data('test', 'test', 'test', 'test', 'test', 'test','test', 'test', 'test', 'test', 'test')
    print(status)
    db = DBService(tabel_name='all_prompt')
    status = db.recreate_table()
    status = db.insert_data('test', 'test', 'test', 'test', 'test', 'test','test', 'test', 'test', 'test', 'test')
    print(status)
    # from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table
    # from sqlalchemy.ext.declarative import declarative_base
    # from sqlalchemy.orm import sessionmaker

    # engine = create_engine('sqlite:///D:/Works/Construct_Sft_Data/knowledge_base/test.db', echo=True)

    # # 创建一个基类
    # Base = declarative_base()

    # # 定义表结构
    # class MyTable(Base):
    #     __tablename__ = 'my_table'  # 表名

    #     id = Column(Integer, primary_key=True)
    #     name = Column(String)
    #     age = Column(Integer)

    # # 创建表
    # Base.metadata.create_all(engine)

    # # 验证表是否创建成功
    # Session = sessionmaker(bind=engine)
    # session = Session()

    # # 插入数据测试
    # new_entry = MyTable(name="Alice", age=30)
    # session.add(new_entry)
    # session.commit()

    # # 查询数据测试
    # result = session.query(MyTable).all()
    # for row in result:
    #     print(f"ID: {row.id}, Name: {row.name}, Age: {row.age}")

    # session.close()
    