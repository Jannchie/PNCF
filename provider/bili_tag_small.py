# %%
import csv
import os
from typing import Literal
import torch
from tqdm import tqdm
import pandas as pd
import csv

from provider.provider import DataProvider
from provider.user_item_sampler import UserItemSampler


class BiliTagProvider(DataProvider):
    def __init__(self):
        super(BiliTagProvider, self).__init__()
        self.raw = list(csv.DictReader(open('./raw/bilibili/bili_tag_small.csv',
                                            'r', encoding='utf-8-sig')))
        self.df = None

    def generate_df(self):
        data = self.raw
        data_dict = {}
        for datum in tqdm(data, desc="Generate Data Dict"):
            mid = int(datum['mid'])
            tag_list = [int(tag_id)
                        for tag_id in datum['tag_list'][1:-1].split(',') if tag_id != '']
            if mid not in data_dict:
                data_dict[mid] = {}
            for tag in tag_list:
                data_dict[mid][tag] = 1 + data_dict[mid].get(tag, 0)
        rows = []
        for mid in tqdm(data_dict.keys(), desc="Generate Dataframe"):
            for tag in data_dict[mid].keys():
                current = data_dict[mid][tag]
                rows.append(
                    [mid, tag, current])
        self.df = pd.DataFrame(
            rows, columns=['user', 'item', 'current'])
