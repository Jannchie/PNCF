

from pyexpat import model
from runner import run_PNCF_model
import pickle

if __name__ == '__main__':
    batch_size = 1024
    bag_size = 20
    core = 10
    dataset_name = 'ml_100k'
    vec_size = 32
    vec_neg = 5
    vec_window = 5
    train_neg = 5
    lr = 0.001
    epochs = 10
    model_name = 'PNCF'

    res = pickle.load(open('res.pkl', 'rb'))
    for model_name in 'NeuMF',:
        if model_name not in res:
            res[model_name] = []
        for i in range(100):
            metrics = run_PNCF_model(dataset_name, model_name, batch_size, bag_size, core,
                                     vec_size, vec_neg, vec_window, train_neg, lr, epochs, cache=False)
            res[model_name].append(metrics)
            pickle.dump(res, open('res.pkl', 'wb'))
