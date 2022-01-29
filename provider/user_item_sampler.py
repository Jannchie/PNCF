import csv
from typing import Literal
import torch
import pandas as pd
from tqdm import tqdm


class UserItemSampler:
    def __init__(self, train_set: pd.DataFrame, target: Literal['user', 'item'] = 'user'):
        self.group = 'user' if target == 'item' else 'item'
        self.target = target
        self.samples = []
        self.neg_samples = []
        self.grouper_dict = train_set.groupby(
            self.group)['current'].sum().to_dict()
        self.target_dict = train_set.groupby(target)['current'].sum().to_dict()
        self.target_radio = torch.tensor(
            list(self.target_dict.values())) / sum(self.target_dict.values())
        self.target_list = torch.tensor(list(self.target_dict.keys()))
        self.dataset = train_set.set_index([self.group, target])

    def generate_samples(self, out: str, window: int = 4):
        bar = tqdm(desc="Generate Samples", total=len(
            self.grouper_dict))
        csv_writer = csv.writer(
            open(out, 'w', encoding='utf-8-sig', newline=''))
        for id, count in self.grouper_dict.items():
            samples = self._sample(self.target, window, self.group, id, count)
            bar.update()
            csv_writer.writerows(samples)

    def _sample(self, target, window, group, id, count):
        df = self.dataset[self.dataset.index.get_level_values(group) == id]
        radio = torch.tensor(df['current'].to_list()) / count
        samples = []
        # neg_samples = []
        for _ in range(int(count ** 0.75)):
            idxs = torch.multinomial(radio, window, True)
            sample = df.iloc[idxs].index.get_level_values(target).to_list()
            samples.append(sample)
            # for _ in range(neg_nums):
            #     idxs = torch.multinomial(
            #         self.target_radio, window * 2, True)
            #     neg_sample = self.target_list[idxs]
            #     neg_sample = [
            #         w for w in neg_sample if w not in sample][:window]
            # neg_samples.append(neg_sample)
        # return samples, neg_samples
        return samples
