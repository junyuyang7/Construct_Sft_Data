from datetime import datetime
from sqlalchemy import Column, DateTime, String, Integer, ForeignKey
from Script.db.models.base import BaseModel
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class AllPrompt(BaseModel):
    __tablename__ = 'all_prompt'  # 定义数据库表的名称

    id = Column(Integer, ForeignKey('base_model.id'), primary_key=True)
    domain_name = Column(String, default=None, comment="领域名字")
    task_name = Column(String, default=None, comment="任务名字")
    cls_name = Column(String, default=None, comment="任务名字")
    first_query_prompt = Column(String, default=None, comment="first_query_prompt")
    query_prompt = Column(String, default=None, comment="query_prompt")
    answer_prompt = Column(String, default=None, comment="answer_prompt")
    evaluate_prompt = Column(String, default=None, comment="evaluate_prompt")
    first_query_args = Column(String, default=None, comment="first_query_prompt中的参数")
    query_args = Column(String, default=None, comment="query_prompt中的参数")
    answer_args = Column(String, default=None, comment="answer_prompt中的参数")
    evaluate_args = Column(String, default=None, comment="evaluate_prompt中的参数")

class FirstQueryPrompt(BaseModel):
    __tablename__ = 'first_query_prompt'  # 定义数据库表的名称

    id = Column(Integer, ForeignKey('base_model.id'), primary_key=True)
    domain_name = Column(String, default=None, comment="领域名字")
    task_name = Column(String, default=None, comment="任务名字")
    cls_name = Column(String, default=None, comment="任务名字")
    prompt = Column(String, default=None, comment="prompt")
    args = Column(String, default=None, comment="prompt中的参数")

class QueryPrompt(BaseModel):
    __tablename__ = 'query_prompt'  # 定义数据库表的名称

    id = Column(Integer, ForeignKey('base_model.id'), primary_key=True)
    domain_name = Column(String, default=None, comment="领域名字")
    task_name = Column(String, default=None, comment="任务名字")
    cls_name = Column(String, default=None, comment="任务名字")
    prompt = Column(String, default=None, comment="prompt")
    args = Column(String, default=None, comment="prompt中的参数")

class AnswerPrompt(BaseModel):
    __tablename__ = 'answer_prompt'  # 定义数据库表的名称

    id = Column(Integer, ForeignKey('base_model.id'), primary_key=True)
    domain_name = Column(String, default=None, comment="领域名字")
    task_name = Column(String, default=None, comment="任务名字")
    cls_name = Column(String, default=None, comment="任务名字")
    prompt = Column(String, default=None, comment="prompt")
    args = Column(String, default=None, comment="prompt中的参数")

class EvaluatePrompt(BaseModel):
    __tablename__ = 'evaluate_prompt'  # 定义数据库表的名称

    id = Column(Integer, ForeignKey('base_model.id'), primary_key=True)
    domain_name = Column(String, default=None, comment="领域名字")
    task_name = Column(String, default=None, comment="任务名字")
    cls_name = Column(String, default=None, comment="任务名字")
    prompt = Column(String, default=None, comment="prompt")
    args = Column(String, default=None, comment="prompt中的参数")
