from datetime import datetime
from sqlalchemy import Column, DateTime, String, Integer, ForeignKey
from Script.db.models.base import BaseModel
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class PromptModel(BaseModel):
    __tablename__ = 'prompt_model'  # 定义数据库表的名称

    id = Column(Integer, ForeignKey('base_model.id'), primary_key=True)
    domain_name = Column(String, default=None, comment="领域名字")
    task_name = Column(String, default=None, comment="任务名字")
    cls_name = Column(String, default=None, comment="任务名字")
    model_type = Column(String, default=None, comment="ASK model, Answer model, Topic Model, Judge Model")
    prompt = Column(String, default=None, comment="prompt")
    args = Column(String, default=None, comment="prompt中的参数")
