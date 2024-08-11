from datetime import datetime
from sqlalchemy import Column, DateTime, String, Integer, ForeignKey
from Script.db.models.base import BaseModel

class SFTDataModel(BaseModel):
    __tablename__ = 'sftdata_model'  # 定义数据库表的名称

    id = Column(Integer, ForeignKey('base_model.id'), primary_key=True)
    inputs = Column(String, default=None, comment="输入")
    targets = Column(String, default=None, comment="输出")
    turn = Column(Integer, default=1, comment="对话轮数")
    domain_name = Column(String, default=None, comment="领域名字")
    task_name = Column(String, default=None, comment="任务名字")
    cls_name = Column(String, default=None, comment="任务名字")
    prompt = Column(String, default=None, comment="构造的prompt") # 列表
    score = Column(Integer, default=1, comment="对话轮数")
    history = Column(String, default=None, comment="对话整体")