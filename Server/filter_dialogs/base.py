# 先搞清楚有哪些指标
# 该类的功能只有一个，计算指标返回排序结果

class FilterBase:
    def __init__(self, final_df, ):
        self.final_df = final_df
        
    def filter(self):
        return 