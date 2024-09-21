from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np

from Server.filter_dialogs.base import FilterBase
from Server.model_workers.base import LLMModelBase
from Server.filter_dialogs.utils import single_ppl
from Server.filter_dialogs.TrainedLLM.less.warmup_train_lora import warmup_train

class TrainedLLMCal(FilterBase):
    def set_target_llm(self, target_llm):
        self.target_llm = target_llm

    def get_ifd(self, whole_text, output):
        ppl_alone = self.llm_model.get_loss(prompt=output)
        ppl_condi = self.llm_model.get_loss(prompt=whole_text, output=output)
        ifd = ppl_condi / ppl_alone

        return ifd

    def get_rifd(self, whole_text_reverse, inputt):
        ppl_alone = self.llm_model.get_loss(prompt=inputt)
        ppl_condi = self.llm_model.get_loss(prompt=whole_text_reverse, output=inputt)
        rifd = ppl_condi / ppl_alone

        return rifd

    # 获取 InfAdam
    def get_less(self, eval_data):
        '''
        mode: [Adam, SGD]
        data: eval_data, [input_ids, labels, attention_mask]
        '''
        # 1.warmup training
        warmup_train()

        # 2.get grad_store
        get_info()

        # 3.calculate InfAdam
        inf_score = get_infscore()

        return inf_score



 
