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
from Server.db.models import RawDialogModel

from Server.db.repository import (add_raw_data_to_db, list_raw_data_from_db, find_raw_data_from_keyword, delete_raw_data_from_db, update_raw_data_from_db)

from sqlalchemy.exc import SQLAlchemyError
from utils import get_db_path
from Server.config import SQLALCHEMY_DATABASE_URI
from base import DBService

class RawDialogDBService(DBService):
    '''
    基类：创建 / 增删改查数据库 
    保存向量库 / 创建知识库 / 删除数据库内容 / 添加文件 / 更新知识库 以及各种抽象方法接口
    '''
    def __init__(self):
        super(self, DBService).__init__()
        self.model = RawDialogModel
    
    def insert_data(self, raw_dialog, turn):
        # 检查是否存在重复记录（排除id）
        status = add_raw_data_to_db(raw_dialog, turn)
        return status
    
    def get_data_by_keyword(self, keyword):
        status, prompt_dict = find_raw_data_from_keyword(keyword)
        return status, prompt_dict

    def get_all_data(self):
        resp = list_raw_data_from_db()
        return resp

    def update_data(self, idx, **kwargs):
        status = update_raw_data_from_db(idx, **kwargs)
        return status

    def delete_data(self, idx):
        status = delete_raw_data_from_db(idx)
        return status


if __name__ == '__main__':
    # db = DBService(tabel_name='prompt_model')
    # status = db.insert_data('test', 'test', 'test', 'test', 'test')
    # print(status)
    db = RawDialogDBService(tabel_name='query_prompt')
    status = db.recreate_table()
    status = db.insert_data('test', 'test', 'test', 'test', 'test', 'test', 'test', 'test','test', 'test', 'test', 'test', 'test')
    print(status)

    # session.close()
    