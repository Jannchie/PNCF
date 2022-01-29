import pandas as pd

from provider.bili_tag_small import DataProvider


class ML100KProvider(DataProvider):
    def generate_df(self):
        self.df = pd.read_csv(f'./raw/ml-100k/u.data', sep='\t', header=None,
                              names=['user', 'item', 'rating', 'timestamp'], engine='python')
        self.df.insert(self.df.shape[1], 'current', 1)


class ML1MProvider(DataProvider):
    def generate_df(self):
        self.df = pd.read_csv(f'./raw/ml-1m/ratings.dat', sep='::', header=None,
                              names=['user', 'item', 'rating', 'timestamp'], engine='python')
        self.df.insert(self.df.shape[1], 'current', 1)


class ML10MProvider(DataProvider):
    def generate_df(self):
        self.df = pd.read_csv(f'./raw/ml-10M100K/ratings.dat', sep='::', header=None,
                              names=['user', 'item', 'rating', 'timestamp'], engine='python')
        self.df.insert(self.df.shape[1], 'current', 1)


class ML20MProvider(DataProvider):
    def generate_df(self):
        self.df = pd.read_csv(f'./raw/ml-20m/ratings.csv')
        self.df.rename(columns={'userId': 'user',
                       'movieId': 'item'}, inplace=True)
        self.df.insert(self.df.shape[1], 'current', 1)
