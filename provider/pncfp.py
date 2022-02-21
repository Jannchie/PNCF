
import csv
import os
import random
from typing import List
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
import gensim
from multiprocessing import Pool
import pandas


class PNCFDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, uv: gensim.models.word2vec.Word2Vec, iv: gensim.models.word2vec.Word2Vec, neg_nums: int = 20, cache=True):
        path = f'./temp/pncf-neg{neg_nums}-u{uv.wv.vectors.shape[0]}-{uv.wv.vectors.shape[1]}-v{iv.wv.vectors.shape[0]}-{iv.wv.vectors.shape[1]}-r{len(data)}.csv'
        if not os.path.exists(path) or not cache:
            self.data = []
            uk = uv.wv.index2word
            ik = iv.wv.index2word
            uunk = uv.wv.vocab.get('<UNK>')
            iunk = iv.wv.vocab.get('<UNK>')

            for _, row in tqdm(data.iterrows(), desc="Generate PNCF Dataset"):
                self.data.append([uv.wv.vocab.get(row["user"], uunk).index,
                                  iv.wv.vocab.get(row["item"], iunk).index, 1.])
                for _ in range(neg_nums):
                    self.data.append(
                        [uv.wv.vocab[random.choice(uk)].index, iv.wv.vocab[random.choice(ik)].index, 0.])
            csv.writer(open(path, 'w')).writerows(self.data)
        else:
            print('PNCFDataset: already exists!')
            self.data = pandas.read_csv(
                path, header=None, dtype=np.int32).values.tolist()
            print('Loaded from file!')

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    data = list(csv.reader(open('./data/bt_sm_train_set.csv')))
    pds = PNCFDataset(data[1:])
    pass
