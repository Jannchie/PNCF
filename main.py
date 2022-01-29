

from runner import run_PNCF_model


if __name__ == '__main__':
    batch_size = 1024
    bag_size = 20
    core = 10
    dataset_name = 'ml_100k'
    model_name = 'PNCF'
    vec_size = 32
    vec_neg = 5
    vec_window = 5
    train_neg = 5
    lr = 0.001
    epochs = 10

    metrics = run_PNCF_model(batch_size, bag_size, core,
                             dataset_name, vec_size, vec_neg, vec_window, train_neg, lr, epochs)
    metrics = {
        'Precision@10': metrics[2][0],
        'Recall@10': metrics[2][1],
        'NDCG@10': metrics[2][3],
        'HR@10': metrics[2][2],
        'MRR@10': metrics[2][4],
    }
