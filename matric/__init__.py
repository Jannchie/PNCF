
import numpy as np


def dcg_at_k(r, k):
    """
    Discounted Cumulative Gain calculation method
    Parameters
    ----------
    r : List, Relevance scores (list or numpy) in rank order
                (first element is the first item)
    k : int, top-K number

    Returns
    -------
    dcg : float, DCG value
    """
    assert k >= 1
    r = np.asfarray(r)[:k] != 0
    if r.size:
        dcg = np.sum(np.subtract(np.power(2, r), 1) /
                     np.log2(np.arange(2, r.size + 2)))
        return dcg
    return 0.


def ndcg_at_k(r, k):
    """
    Normalized Discounted Cumulative Gain calculation method
    Parameters
    ----------
    r : List, Relevance scores (list or numpy) in rank order
            (first element is the first item)
    k : int, top-K number

    Returns
    -------
    ndcg : float, NDCG value
    """
    assert k >= 1
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    ndcg = dcg_at_k(r, k) / idcg

    return ndcg


def mrr_at_k(rs, k):
    """
    Mean Reciprocal Rank calculation method
    Parameters
    ----------
    rs : rank items
    k : int, topK number

    Returns
    -------
    mrr : float, MRR value
    """
    assert k >= 1
    res = 0
    for r in rs:
        r = np.asarray(r)[:k] != 0
        for index, item in enumerate(r):
            if item == 1:
                res += 1 / (index + 1)
    mrr = res / len(rs)

    return mrr
