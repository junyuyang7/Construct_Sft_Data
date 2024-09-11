import importlib
import os
from Server.config import KB_ROOT_PATH

def get_db_path(knowledge_base_name: str):
    '''获取知识库路径'''
    return os.path.join(KB_ROOT_PATH, knowledge_base_name)