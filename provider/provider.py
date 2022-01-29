import datetime
import logging
import os
from typing import Tuple
import pandas as pd

from provider.user_item_sampler import UserItemSampler


class DataProvider:
    def __init__(self):
        super(DataProvider, self).__init__()
        self.df = None

    def split_df(self, test_radio=.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        started_at = datetime.datetime.now()
        logging.info(f"Splitting dataframe, test radio = {test_radio}")
        test_idx = [x for x in self.df.groupby('user').apply(
            lambda x: x.sample(frac=test_radio).index
        ).explode().values if x != 'nan']
        train_set = self.df[~self.df.index.isin(test_idx)]
        test_set = self.df[self.df.index.isin(test_idx)]
        logging.info(
            f"Splitting dataframe, used {(datetime.datetime.now() - started_at)}")
        return train_set, test_set

    def reduce(self, core_num=10, level='ui'):
        def filter_user(df):
            tmp = df.groupby(['user'], as_index=False)['item'].count()
            tmp.rename(columns={'item': 'cnt_item'}, inplace=True)
            df = df.merge(tmp, on=['user'])
            df = df.query(f'cnt_item >= {core_num}').reset_index(
                drop=True).copy()
            df.drop(['cnt_item'], axis=1, inplace=True)
            return df

        def filter_item(df):
            tmp = df.groupby(['item'], as_index=False)['user'].count()
            tmp.rename(columns={'user': 'cnt_user'}, inplace=True)
            df = df.merge(tmp, on=['item'])
            df = df.query(f'cnt_user >= {core_num}').reset_index(
                drop=True).copy()
            df.drop(['cnt_user'], axis=1, inplace=True)
            return df

        if level == 'ui':
            while 1:
                self.df = filter_user(self.df)
                self.df = filter_item(self.df)
                chk_u = self.df.groupby('user')['item'].count()
                chk_i = self.df.groupby('item')['user'].count()
                if len(chk_i[chk_i < core_num]) <= 0 and len(chk_u[chk_u < core_num]) <= 0:
                    break
        elif level == 'u':
            self.df = filter_user(self.df)
        elif level == 'i':
            self.df = filter_item(self.df)
        else:
            raise ValueError(f'Invalid level value: {level}')


def generate_samples(provider: DataProvider, data_name, bag_size=20, core=10):
    if os.path.exists(f'./temp/{data_name}_u_{bag_size}_c{core}_sample.csv'):
        print('samples exists')
        return
    provider.generate_df()
    provider.reduce(core_num=core, level='ui')
    train_set, test_set = provider.split_df()
    train_set.to_csv(f'./data/{data_name}_train_set.csv', index=False)
    test_set.to_csv(f'./data/{data_name}_test_set.csv', index=False)
    uis = UserItemSampler(train_set, 'user')
    uis.generate_samples(
        f'./temp/{data_name}_u_{bag_size}_c{core}_sample.csv', bag_size)
    iis = UserItemSampler(train_set, 'item')
    iis.generate_samples(
        f'./temp/{data_name}_i_{bag_size}_c{core}_sample.csv', bag_size)
