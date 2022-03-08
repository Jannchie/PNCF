#%%
import numpy
import seaborn as sns
from scipy.stats import ttest_ind, ttest_rel, kstest
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
res = pickle.load(open('res.pkl', 'rb'))
# samp = [
#     ['precision', 'recall', 'hit_ratio', 'ndcg', 'mrr'],
#     [0.3001, 0.0231, 0.3001, 0.3001, 0.3001],
#     [0.284, 0.1048, 0.7466, 0.5325, 0.6679],
#     [0.2543, 0.1814, 0.8696, 0.5648, 0.814],
#     [0.2234, 0.2899, 0.9502, 0.5809, 0.9436],
#     [0.1969, 0.3666, 0.9735, 0.5818, 1.0011],
#     [0.1608, 0.4747, 0.9947, 0.5815, 1.0559]
# ]
model_names = 'PNCF', 'NPNCF', 'NeuMF'
j2m = {i: item for i, item in enumerate(
    ('precision', 'recall', 'hit_ratio', 'ndcg', 'mrr'))}
i2k = {i+1: item for i, item in enumerate((1, 5, 10, 20, 30, 50))}

test_data = defaultdict(list)
for model_name in model_names:
    for matric in res[model_name]:
        for i in range(1, len(matric)):
            k = i2k[i]
            for j in range(len(matric[0])):
                m = j2m[j]
                test_data[f'{model_name}-{m}-{k}'].append(matric[i][j])
pickle.dump(test_data, open('test_data.pkl', 'wb'))
#%%
#%%

for m in 'precision', 'recall', 'hit_ratio', 'ndcg', :
    a = test_data[f'PNCF-{m}-20']
    b = test_data[f'NPNCF-{m}-20']
    c = test_data[f'NeuMF-{m}-20']
    bins = numpy.linspace(min(min(a), min(b), min(c)),
                          max(max(a), max(b), max(c)), 20)
    plt.hist(a[:100], bins=bins, alpha=0.5, label='PNCF')
    plt.hist(b[:100], bins=bins, alpha=0.5, label='MLP')
    plt.hist(c[:100], bins=bins, alpha=0.5, label='NeuMF')
    # y axis label is count
    plt.ylabel('Count')
    plt.xlabel(m)
    plt.legend()
    plt.savefig(f'{m}.svg')
    plt.show()
    ttest_ind(a, c)

#%%
for m in 'precision', 'recall', 'hit_ratio', 'ndcg', :
    a = test_data[f'PNCF-{m}-20']
    b = test_data[f'NPNCF-{m}-20']
    c = test_data[f'NeuMF-{m}-20']
    