from datetime import datetime
from sqlalchemy import Column, DateTime, String, Integer, ForeignKey
from Server.db.models.base import BaseModel
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class RawDialogModel(BaseModel):
    __tablename__ = 'raw_dialog'  # 定义数据库表的名称

    id = Column(Integer, ForeignKey('base_model.id'), primary_key=True)
    message = Column(String, default=None, comment="具体对话")
    turn = Column(Integer, default=None, comment="对话轮数")