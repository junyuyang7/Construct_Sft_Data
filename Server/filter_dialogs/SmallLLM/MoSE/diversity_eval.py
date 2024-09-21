import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import json
import sys
import numpy as np
from transformers import BertTokenizer, BertModel,AutoModel
import torch

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import abc
from sklearn.metrics import pairwise_distances

class SamplingMethod(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __init__(self, X, **kwargs):
    self.X = X

  def flatten_X(self):
    shape = self.X.shape
    flat_X = self.X
    if len(shape) > 2:
      flat_X = np.reshape(self.X, (shape[0],np.product(shape[1:])))
    return flat_X


  @abc.abstractmethod
  def select_batch_(self):
    return

  def select_batch(self, **kwargs):
    return self.select_batch_(**kwargs)

  def to_dict(self):
    return None

class kCenterGreedy(SamplingMethod):
  def __init__(self, X, metric='euclidean'):
    self.X = X
    self.flat_X = self.flatten_X()
    self.name = 'kcenter'
    self.features = self.flat_X
    self.metric = metric
    self.min_distances = None
    self.n_obs = self.X.shape[0]
    self.already_selected = []
    print('shape of features')
    print(X.shape)

  def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
    """Update min distances given cluster centers.

    Args:
      cluster_centers: indices of cluster centers
      only_new: only calculate distance for newly selected points and update
        min_distances.
      rest_dist: whether to reset min_distances.
    """

    if reset_dist:
      self.min_distances = None
    if only_new:
      cluster_centers = [d for d in cluster_centers
                         if d not in self.already_selected]
    if cluster_centers:
      # Update min_distances for all examples given new cluster center.
      x = self.features[cluster_centers]
      dist = pairwise_distances(self.features, x, metric=self.metric)

      if self.min_distances is None:
        self.min_distances = np.min(dist, axis=1).reshape(-1,1)
      else:
        self.min_distances = np.minimum(self.min_distances, dist)

  def select_batch_(self, features, already_selected, N, **kwargs):
    """
    Diversity promoting active learning method that greedily forms a batch
    to minimize the maximum distance to a cluster center among all unlabeled
    datapoints.

    Args:
      model: model with scikit-like API with decision_function implemented
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to minimize distance to cluster centers
    """
    #if already_selected is None:
        #already_selected = []
     #   already_selected = [ index for index in np.random.choice(np.arange(self.n_obs),200,replace=False)]
    try:
      # Assumes that the transform function takes in original data and not
      # flattened data.
      print('Getting transformed features...')
      self.features = features
      print('Calculating distances...')
      self.update_distances(already_selected, only_new=False, reset_dist=True)
    except:
      print('Using flat_X as features.')
      self.update_distances(already_selected, only_new=True, reset_dist=False)

    if already_selected is None:
        already_selected = []
     #   already_selected = np.random.choice(np.arange(self.n_obs),100,replace=False)
    self.already_selected = already_selected
    print(self.already_selected)

    new_batch = []

    for _ in range(N):
      if self.already_selected == []:
        # Initialize centers with a randomly selected datapoint
        ind = np.random.choice(np.arange(self.n_obs))
      else:
        ind = np.argmax(self.min_distances)
      # New examples should not be in already selected since those points
      # should have min_distance of zero to a cluster center.
      assert ind not in already_selected
      
      if self.min_distances is None:
        print('min distances is None')
      else:
        print('Maximum distance from cluster centers is %0.2f'
            % max(self.min_distances))
      
      self.update_distances([ind], only_new=True, reset_dist=False)
      new_batch.append(ind)
      
      if self.already_selected is None:
          self.already_selected = []
      else:
          self.already_selected.append(ind)

    print('Maximum distance from cluster centers is %0.2f'
            % max(self.min_distances))
    
    return self.already_selected

@torch.no_grad()
def bert_embedding(texts,batch=100):
    tokenizer = BertTokenizer.from_pretrained('../models/bert-base-uncased')
    model = AutoModel.from_pretrained('../models/bert-base-uncased').cuda()
    # 将文本转化为BERT模型可识别的token序列
    encoded_texts = tokenizer(texts,return_tensors="pt",truncation=True,padding=True,max_length=96)
    encoded_texts =  encoded_texts.to("cuda")
    cls_hid_li = []
    # 使用BERT模型对每个文本序列进行编码,提取其语义向量
    i= 0
    while i < len(texts):
        last_hids = model(input_ids=encoded_texts["input_ids"][i:i+batch],
                          attention_mask=encoded_texts["attention_mask"][i:i+batch])['last_hidden_state']
        cls_hids = last_hids[:,0,:].squeeze()
        cls_hid_li.append(cls_hids)
        i+= batch
        print(i)
    # 将所有文本的embedding连成特征矩阵
    cls_hids_tensor = torch.concat(cls_hid_li, dim=0)
    np.save("bert_embedding.npy",cls_hids_tensor.cpu())
    return np.array(cls_hids_tensor.cpu())

# 数据采样
def sample_func(text_list,K):
    result = []
    if os.path.exists("bert_embedding.npy"):
        text_embedding = np.load("bert_embedding.npy")
    else:
        text_embedding = bert_embedding(text_list)
        np.save("bert_embedding.npy",text_embedding)
    
    result = []

    k_center = kCenterGreedy(text_embedding)
    
    already_selected = None
    #for _ in range(K):
    result = k_center.select_batch_(text_embedding,already_selected,K)
        #result = result + new_data
        #already_selected += new_data
    return result


def diversity_eval(input_file, output_file, K):
    data = json.load(fp=open(input_file, "r"))
    instruction_list = []
    for d in data:
        instruction_list.append(d["instruction"])
    res = sample_func(text_list = instruction_list, K = K)
    print('data length')
    print(len(data))
    
    print('sampling data:')
    print(len(res))
    print(res)
    data_li = []
    for index in res:
        data_li.append(data[index])
    json.dump(obj=data_li,fp=open(output_file,"w"),indent=2,ensure_ascii=False)

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    K = int(sys.argv[3])
    diversity_eval(input_file, output_file, K)