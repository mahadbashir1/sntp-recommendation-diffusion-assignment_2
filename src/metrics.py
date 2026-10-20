"""Requirement 7 - evaluation measures for recommendation quality.

Relevance  : a held-out (test) rating >= 4 stars.
Candidates : every item the user did NOT rate in training (full ranking, no
             negative sampling), so the measures are absolute but small.
Cut-off    : K = 10 (one screen of recommendations).
Accuracy   : only for a well-defined classification task - for observed test
             ratings, predict "liked" (rating >= 4) vs "not liked" (rating <= 3)
             by thresholding the predicted rating at 3.5. Negatives are
             observed low ratings; unrated items are NOT negatives (missing
             does not mean disliked).
"""
import numpy as np


def rmse(pred, y):
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def mae(pred, y):
    return float(np.mean(np.abs(pred - y)))


def classification(pred, y, thr_true=4.0, thr_pred=3.5):
    t = y >= thr_true
    p = pred >= thr_pred
    tpr = np.mean(p[t]) if t.any() else np.nan
    tnr = np.mean(~p[~t]) if (~t).any() else np.nan
    return {"accuracy": float(np.mean(p == t)), "balanced_accuracy": float((tpr + tnr) / 2),
            "majority_accuracy": float(max(t.mean(), 1 - t.mean()))}


def ranking(lists, relevant, k):
    """lists: (n_users, >=k) item ids, relevant: list of sets (test items rated >= 4)."""
    return {key: float(np.mean(v)) for key, v in ranking_per_user(lists, relevant, k).items()}


def ranking_per_user(lists, relevant, k):
    """Precision@k, recall@k, NDCG@k and hit@k for every user (arrays)."""
    prec, rec, ndcg, hit = [], [], [], []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for row, rel in zip(lists, relevant):
        hits = np.array([it in rel for it in row[:k]], dtype=float)
        prec.append(hits.sum() / k)
        rec.append(hits.sum() / len(rel))
        ideal = discounts[:min(len(rel), k)].sum()
        ndcg.append((hits * discounts).sum() / ideal)
        hit.append(float(hits.any()))
    return {"precision": np.array(prec), "recall": np.array(rec), "ndcg": np.array(ndcg), "hit_rate": np.array(hit)}


def beyond_accuracy(lists, R_train, n_users_total):
    """Catalogue coverage, intra-list diversity, novelty and popularity concentration."""
    n_items = R_train.shape[1]
    rec_items, counts = np.unique(lists, return_counts=True)
    coverage = len(rec_items) / n_items
    # item-item cosine similarity of their (binary) training audiences
    M = R_train.copy().tocsc()
    M.data[:] = 1.0
    sub = M[:, rec_items]
    norms = np.sqrt(np.asarray(sub.multiply(sub).sum(axis=0)).ravel()) + 1e-12
    sim = (sub.T @ sub).toarray() / np.outer(norms, norms)
    pos = np.searchsorted(rec_items, lists)
    k = lists.shape[1]
    iu = np.triu_indices(k, 1)
    ild = np.mean([1.0 - sim[np.ix_(p, p)][iu].mean() for p in pos])
    pop = np.asarray(M.sum(axis=0)).ravel()
    novelty = np.mean(-np.log2((pop[lists] + 1) / n_users_total))
    c = np.sort(np.concatenate([counts, np.zeros(n_items - len(counts))]))
    gini = (np.sum((2 * np.arange(1, n_items + 1) - n_items - 1) * c)) / (n_items * c.sum())
    return {"coverage": coverage, "diversity_ild": float(ild), "novelty": float(novelty), "gini": float(gini)}


def relevant_sets(users, u_test, i_test, r_test, thr):
    rel = {}
    for u, i, r in zip(u_test, i_test, r_test):
        if r >= thr:
            rel.setdefault(u, set()).add(i)
    return [rel.get(u, set()) for u in users]

