import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence
from typing import List
from utils import utils
from helpers.BaseReader import BaseReader
from models.BaseModel import GeneralModel

class GTE(GeneralModel):
    reader = 'BaseReader'  # assign a reader class, BaseReader by default
    runner = 'BaseRunner'  # assign a runner class, BaseRunner by default

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self._define_params()
        self.apply(self.init_weights)
        self.user_num = corpus.n_users
        self.item_num = corpus.n_items
        # self.user_rep = corpus.user_rep
        # self.item_rep = corpus.item_rep


    def _define_params(self):
        # define parameters in the model
        self.item_rep = torch.nn.Embedding(self.item_num, self.item_num)
        self.item_rep.weight.data = torch.eye(self.item_num)
        self.user_rep = torch.nn.Embedding(self.user_num, self.item_num)
        self.user_rep.weight.data = torch.zeros(self.user_num, self.item_num)

    def forward(self, feed_dict):
        user_id = feed_dict['user_id']  # [batch_size]
        item_id = feed_dict['item_id']  # [batch_size, num_items]
        # print("user_id shape:", user_id.shape)
        # print("item_id shape:", item_id.shape)

        user_emb = self.user_rep(user_id)  # [batch_size, item_num]
        item_emb = self.item_rep(item_id)  # [item_num, item_num]
        # print("user_emb shape:", user_emb.shape)
        # print("item_emb shape:", item_emb.shape)

        # item_emb = item_emb.view(self.item_num, self.item_num)
        # print("item_emb.T shape:", item_emb.T.shape)
        
        # prediction = torch.matmul(user_emb, item_emb.T)  # [batch_size, item_num]
        prediction = (user_emb[:, None, :] * item_emb).sum(dim=-1)  # [batch_size, item_num]
        out_dict = {'prediction': prediction}
        # print("prediction shape:", prediction.shape)

        return out_dict  # [batch_size * item_num]

    # class Dataset(GeneralModel.Dataset):
    #     def _get_feed_dict(self, index):
    #         feed_dict = super()._get_feed_dict(index)
    #         feed_dict['user_id'] = self.data['user_id'][index]
    #         feed_dict['item_id'] = self.data['item_id'][index]
    #         user_id, item_id = self.data['user_id'][index], self.data['item_id'][index]
    #         feed_dict = {
    #             'user_id': torch.tensor(user_id),
    #             'item_id': torch.tensor(item_id)
    #         }
    #         return feed_dict