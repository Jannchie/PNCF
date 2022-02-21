import os
from gensim.models import Word2Vec
import csv


def pretrain_vec(data_name, bag_size=20, size=64, neg=5, window=5, core=0, cache=True):
    """pretain word2vec

    Args:
        data_name (str): data name
        bag_size (int, optional): 'sentences' length. Defaults to 20.
        size (int, optional): vectors size. Defaults to 64.
        neg (int, optional): negative samples. Defaults to 5.
        window (int, optional): window size. Defaults to 5.
        core (int, optional): data core. Defaults to 0.
        cache (int, optional): use cache data. Defaults to True.
    """
    if os.path.exists(get_vec_filename(data_name, 'i', bag_size, size, neg, window)) and cache:
        print('samples exists pretrain vec')
        return

    data = list(csv.reader(
        open(f'./temp/{data_name}_u_{bag_size}_c{core}_sample.csv', 'r', encoding='utf-8-sig')))
    data.insert(0, ['<UNK>'])
    model = Word2Vec(sentences=data, size=size,
                     window=window, min_count=1, workers=4, negative=neg)
    model.save(
        get_vec_filename(data_name, 'u', bag_size, size, neg, window))

    data = list(csv.reader(
        open(f'./temp/{data_name}_i_{bag_size}_c{core}_sample.csv', 'r', encoding='utf-8-sig')))
    data.insert(0, ['<UNK>'])
    model = Word2Vec(sentences=data, size=size,
                     window=window, min_count=1, workers=4, negative=5)
    model.save(
        get_vec_filename(data_name, 'i', bag_size, size, neg, window))


def get_vec_filename(data_name, type, bag_size, size, neg, window):
    return f'./temp/{data_name}_{type}_{bag_size}_s{size}_n{neg}_w{window}_vec.model'


if __name__ == '__main__':
    pretrain_vec()
    pass
