import datetime
import os
from gensim.models import Word2Vec
import torch
from matric import mrr_at_k, ndcg_at_k
from model.pretrain import get_vec_filename, pretrain_vec
from provider.bili_tag_small import BiliTagProvider
from provider.ml import ML100KProvider, ML10MProvider, ML1MProvider, ML20MProvider
from provider.pncfp import PNCFDataset
import csv
from model.PNCF import PNCF
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
from typing import Literal
from provider.provider import generate_samples


def run_PNCF_model(
        dataset_name: str,
        model_name: Literal['PNCF', 'NPNCF'] = 'PNCF',
        batch_size: int = 512,
        bag_size: int = 20,
        core: int = 10,
        vec_size: int = 32,
        vec_neg: int = 5,
        vec_window: int = 5,
        train_neg: int = 20,
        lr: float = 0.001,
        epochs: int = 3):
    """ run PNCF

    Args:
        dataset_name (str): dataset name
        batch_size (int, optional): batch size. Defaults to 512.
        bag_size (int, optional): bag size. Defaults to 20.
        core (int, optional): core. Defaults to 10.
        vec_size (int, optional): vectors size. Defaults to 32.
        vec_neg (int, optional): vectors negative size. Defaults to 5.
        vec_window (int, optional): vectors window length. Defaults to 5.
        train_neg (int, optional): train negative. Defaults to 20.
        lr (float, optional): learning rate. Defaults to 0.001.
        epochs (int, optional): epochs. Defaults to 3.

    Returns:
        list[list]: matrics 
    """
    cache = True
    provider = None
    if dataset_name == 'bilibili':
        provider = BiliTagProvider()
    elif dataset_name == 'ml_100k':
        provider = ML100KProvider()
    elif dataset_name == 'ml_1m':
        provider = ML1MProvider()
    elif dataset_name == 'ml_10m':
        provider = ML10MProvider()
    elif dataset_name == 'ml_20m':
        provider = ML20MProvider()
    generate_samples(provider, dataset_name,
                     bag_size=bag_size, core=core, cache=cache)
    writer = SummaryWriter(
        f'./log/{dataset_name}/{model_name}/{datetime.datetime.now().timestamp()}-{batch_size}-{bag_size}-{core}-{vec_size}-{vec_neg}-{vec_window}-{train_neg}-{lr}-{epochs}')
    pretrain_vec(dataset_name, bag_size=bag_size,
                 size=vec_size, neg=vec_neg, window=vec_window, core=core, cache=cache)
    iv = Word2Vec.load(get_vec_filename(dataset_name, 'i',
                       bag_size, vec_size, vec_neg, vec_window))
    uv = Word2Vec.load(get_vec_filename(dataset_name, 'u',
                       bag_size, vec_size, vec_neg, vec_window))
    data = pd.read_csv(f'./data/{dataset_name}_train_set.csv', dtype=str)
    pds = PNCFDataset(data, uv, iv, neg_nums=train_neg, cache=cache)

    dl = torch.utils.data.DataLoader(
        pds, batch_size=batch_size, shuffle=True)

    m = PNCF()
    if model_name == 'PNCF':
        m.init(writer, uvec=uv.wv.vectors, ivec=iv.wv.vectors, layer_nums=3)
    elif model_name == 'NPNCF':
        m.init_without_pretrain(
            writer, user_num=uv.wv.vectors.shape[0], item_num=iv.wv.vectors.shape[0], dim=vec_size, layer_nums=3)
    m.fit(dl, lr=lr, epochs=epochs)

    df = pd.read_csv(f'./data/{dataset_name}_test_set.csv', dtype=str)
    user2index = {}
    for idx, val in enumerate(uv.wv.index2word):
        user2index[val] = idx

    item2index = {}
    for idx, val in enumerate(iv.wv.index2word):
        item2index[val] = idx

    df_train = pd.read_csv(f'./data/{dataset_name}_train_set.csv', dtype=str)

    def get_test_dict(user2index, item2index, df, type: str = 'train'):
        ua = {}
        temp = list(user2index.values())
        fname = f'./temp/u{temp[0]}-{temp[-1]}-{type}.pkl'
        if os.path.exists(fname):
            return pickle.load(open(fname, 'rb'))
        for row in tqdm(df.iterrows()):
            user = row[1]['user']
            item = row[1]['item']
            uidx = user2index.get(user, user2index['<UNK>'])
            iidx = item2index.get(item, user2index['<UNK>'])
            if uidx not in ua:
                ua[uidx] = set()
            ua[uidx].add(iidx)
        pickle.dump(
            ua, open(fname, 'wb'))
        return ua

    ua = get_test_dict(user2index, item2index, df_train, type='train')
    us = get_test_dict(user2index, item2index, df, type='test')
    preds = {}
    item_nums = len(iv.wv.index2word)
    for idx in tqdm(us.keys(), total=len(us)):
        k = 50
        if idx not in us:
            continue
        users = (torch.ones(item_nums) * idx).cuda().int()
        items = torch.arange(item_nums).cuda()
        pred = m.cuda().forward(users, items)

        # remove in train set
        pred = pred.index_fill(0, torch.tensor(list(ua[idx])).cuda(), 0)

        indexes = torch.topk(pred, k=k)[1]
        res = [1 if int(i) in us[idx] else 0 for i in indexes]
        preds[idx] = res
    metrics = []
    metrics.append(['precision', 'recall', 'hit_ratio', 'ndcg', 'mrr'])
    for k in [1, 5, 10, 20, 30, 50]:
        preds_k = [l[:k] for l in preds.values()]
        precision = (torch.tensor(preds_k).sum(1) / k).mean().float()

        lens = [len(us[idx]) for idx in preds.keys()]
        recall = (torch.tensor(preds_k).sum(1) /
                  torch.tensor(lens)).mean().float()
        hit_ratio = (torch.tensor(preds_k).sum(
            1).nonzero()).shape[0] / len(preds_k)
        ndcg = np.mean([ndcg_at_k(r, k) for r in preds_k])
        # map_k = map_at_k(tmp_preds.values())
        mrr = mrr_at_k(preds_k, k)
        m.writer.add_scalar('Test/precision', precision, k)
        m.writer.add_scalar('Test/recall', recall, k)
        m.writer.add_scalar('Test/hit_ratio', hit_ratio, k)
        m.writer.add_scalar('Test/ndcg', ndcg, k)
        m.writer.add_scalar('Test/mrr', mrr, k)
        m.writer.flush()
        metrics.append([
            round(float(precision), 4),
            round(float(recall), 4),
            round(float(hit_ratio), 4),
            round(float(ndcg), 4),
            round(float(mrr), 4)
        ])
    csv.writer(open(
        f'./res/{dataset_name}-{model_name}-{datetime.datetime.now().timestamp()}.csv', 'w', newline='')).writerows(metrics)
    return metrics
