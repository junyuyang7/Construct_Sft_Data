# 先搞清楚有哪些指标
# 该类的功能只有一个，计算指标返回排序结果
from Server.model_workers.base import LLMModelBase

class FilterBase:
    def __init__(self, final_df, llm_model: LLMModelBase):
        self.final_df = final_df
        self.llm_model = llm_model
        
    def filter(self):
        return 