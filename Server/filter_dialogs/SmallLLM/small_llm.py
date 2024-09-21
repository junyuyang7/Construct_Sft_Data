from Server.filter_dialogs.base import FilterBase
from Server.model_workers.base import LLMModelBase
from Server.filter_dialogs.SmallLLM.MoSE import quality_eval, select_high_quality_data, necessity_eval, select_high_necessity_data, train, inference, diversity_eval

class SmallLLMCal(FilterBase):
    def set_target_llm(self, target_llm):
        self.target_llm = target_llm

    # 获取 MoSE Score
    def get_mose(self, eval_data):
        '''
        mode: [Adam, SGD]
        data: eval_data, [input_ids, labels, attention_mask]
        '''
        mose_score = []
        # 1.quality eval
        quality_eval()
        select_high_quality_data()

        # 2.diversity eval
        

        # 3.train and inference
        
        # 4.necess eval
        

        return mose_score



 
