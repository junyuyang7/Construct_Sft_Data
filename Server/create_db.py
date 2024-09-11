# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# import sqlite3
# from sqlalchemy import create_engine, and_, or_, MetaData, text
# from sqlalchemy.orm import sessionmaker
# from Server.db.models import PromptModel, SFTDataModel
# from Server.db.repository import (add_prompt_to_db, list_prompts_from_db, find_prompt_from_keyword, prompt_exists, delete_prompt_from_db, update_prompt_from_db, add_history_to_db, list_history_from_db, find_history_from_keyword, history_exists, delete_history_from_db, update_history_from_db)
# from sqlalchemy.exc import SQLAlchemyError

# ModelType = {
#     'promptmodel': PromptModel,
#     'sftdatamodel': SFTDataModel
# }

# db_excutor = {
#     'promptmodel': {
#         'add': add_prompt_to_db,
#         'list_all': list_prompts_from_db,
#         'find': find_prompt_from_keyword,
#         'delete': delete_prompt_from_db,
#         'update': update_prompt_from_db
#     },
#     'sftdatamodel': {
#         'add': add_history_to_db,
#         'list_all': list_history_from_db,
#         'find': find_history_from_keyword,
#         'delete': delete_history_from_db,
#         'update': update_history_from_db
#     }
# }

# class DBconnecter:
#     def __init__(self, model_type) -> None:
#         self.model_type = model_type

#     @staticmethod
#     def create_table(model_type):
#         # 创建 SQLite 数据库（在内存中）
#         engine = create_engine('sqlite:///test.db', echo=True)

#         # 创建所有定义的表
#         ModelType[model_type].metadata.create_all(engine)

#         # 创建一个会话
#         Session = sessionmaker(bind=engine)
#         session = Session()

#         # 关闭会话
#         session.close()

#     @staticmethod
#     def recreate_table(model_type):
#         # 创建 SQLite 数据库连接
#         engine = create_engine('sqlite:///test.db', echo=True)
        
#         # 获取模型定义
#         model = ModelType[model_type]
        
#         # 删除旧表（如果存在）
#         try:
#             with engine.connect() as connection:
#                 if engine.dialect.has_table(connection, model.__tablename__):
#                     connection.execute(text(f'DROP TABLE IF EXISTS {model.__tablename__}'))
#                     print(f"Table '{model.__tablename__}' dropped.")
#                 else:
#                     print(f"Table '{model.__tablename__}' does not exist.")
#         except SQLAlchemyError as e:
#             print(f"Error dropping table: {e}")

#         # 创建新的表
#         try:
#             model.metadata.create_all(engine)
#             print(f"Table '{model.__tablename__}' created.")
#         except SQLAlchemyError as e:
#             print(f"Error creating table: {e}")

#         # 创建一个会话
#         Session = sessionmaker(bind=engine)
#         session = Session()

#         # 关闭会话
#         session.close()

#     def insert_prompt(self, domain_name, task_name, cls_name, model_type, prompt, args):
#         # 检查是否存在重复记录（排除id）
#         status = db_excutor[self.model_type]['add'](domain_name, task_name, cls_name, model_type, prompt, args)
#         return status

#     def get_prompt_by_keyword(self, keyword):
#         status, prompt = db_excutor[self.model_type]['find'](keyword)
#         return status, prompt

#     # @staticmethod
#     def get_all_prompts(self):
#         resp = db_excutor[self.model_type]['list']()
#         return resp

#     @staticmethod
#     def update_prompt(self, session, prompt_id, **kwargs):
#         prompt = session.query(PromptModel).filter(PromptModel.id == prompt_id).first()
#         if prompt:
#             for key, value in kwargs.items():
#                 if hasattr(prompt, key):
#                     setattr(prompt, key, value)
#             session.commit()
#             return True
#         return False

#     # @staticmethod
#     def delete_prompt(self, prompt_id):
#         status = db_excutor[self.model_type]['delete'](prompt_id)
#         return status

# if __name__ == '__main__':
#     # 创建数据库连接
#     db = DBconnecter()
#     # engine = create_engine('sqlite:///test.db', echo=True)

#     # # 创建会话
#     # Session = sessionmaker(bind=engine)
#     # session = Session()

#     # import pandas as pd
#     # df = pd.read_excel('Server/test.xlsx')

#     # for _, row in df.iterrows():
#     #     new_id = db.insert_prompt(
#     #         session,
#     #         domain_name=row['domain_name'],
#     #         task_name=row['task_name'],
#     #         cls_name=row['cls_name'],
#     #         model_type=row['model_type'],
#     #         prompt=row['prompt'],
#     #         args=row['args']
#     #     )

#     # session.close()
#     db.recreate_table('sftdatamodel')

