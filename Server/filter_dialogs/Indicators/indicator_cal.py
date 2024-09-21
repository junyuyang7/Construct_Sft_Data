from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np

from Server.filter_dialogs.base import FilterBase
from Server.model_workers.base import LLMModelBase
from Server.filter_dialogs.utils.utils import single_ppl


class IndicatorCal(FilterBase):
    def get_ppl(self, text):
        ppl = single_ppl(text, self.llm_model)
        return ppl

    # Measure of Textual Lexical Diversity
    def get_mtld(self, text: str, threshold=0.72):
        def count_mtld(tokens, threshold):
            types, token_count, factor_count = defaultdict(int), 0, 0
            for token in tokens:
                types[token] += 1
                token_count += 1
                ttr = len(types) / token_count
                if ttr < threshold:
                    factor_count += 1
                    types.clear()
                    token_count = 0
            return factor_count + (1 - ttr) / (1 - threshold) if token_count else factor_count

        tokens = text.split()
        forward_mtld = count_mtld(tokens, threshold)
        reverse_mtld = count_mtld(tokens[::-1], threshold)
        return (len(tokens) / forward_mtld + len(tokens) / reverse_mtld) / 2
    
    def get_length(self, text):
        return len(text)
    
    # 计算 KNN-i 指标（Distance to approximate ith-nearest neighbors）
    def calculate_knn_i(self, sentences, i):
        # Step 1: Load pre-trained SentenceBERT model
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        # Step 2: Encode the sentences to get embeddings
        embeddings = model.encode(sentences)

        # Step 3: Fit NearestNeighbors model to the embeddings
        nbrs = NearestNeighbors(n_neighbors=i+1, algorithm='auto').fit(embeddings)

        # Step 4: Calculate distances to the i-th nearest neighbor
        distances, _ = nbrs.kneighbors(embeddings)
        
        # i-th nearest neighbor distance (index i because 0th is the point itself)
        knn_i_distances = distances[:, i]

        return knn_i_distances
    
    def get_ifd(self, whole_text, output):
        ppl_alone = single_ppl(prompt=output)
        ppl_condi = single_ppl(prompt=whole_text, output=output)

        ifd = ppl_condi / ppl_alone

        return ifd

    def get_rifd(self, whole_text_reverse, inputt):
        ppl_alone = single_ppl(prompt=inputt)
        ppl_condi = single_ppl(prompt=whole_text_reverse, output=inputt)

        rifd = ppl_condi / ppl_alone

        return rifd

    



