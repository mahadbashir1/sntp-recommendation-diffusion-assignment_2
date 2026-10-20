"""Requirements 2 and 3 - recommendation models implemented from scratch.

Numpy / scipy are only used for array arithmetic; the learning and prediction
logic of every recommender is written here.

Models (each is a plain dict produced by a fit_* function):
  * popularity  - non-personalised reference: number of positive (>= 4) ratings.
  * user_knn    - memory-based collaborative filtering. Mean-centred cosine
                  similarity with significance shrinkage, top-k neighbourhood,
                  prediction = user mean + weighted neighbour deviations.
  * mf          - model-based latent factor model r = mu + b_u + b_i + p_u.q_i
                  learned by Alternating Least Squares (closed-form ridge
                  solves, batched for all users / items at once).
  * social mf   - the same MF with a SocialMF-style trust regulariser
                  beta * ||p_u - mean_{f in trusted(u)} p_f||^2 . beta = 0
                  gives back plain MF, so the social contribution can be
                  isolated experimentally by sweeping beta.
"""
import numpy as np
import scipy.sparse as sp


def rating_matrix(n_users, n_items, u, i, r):
    return sp.csr_matrix((r.astype(np.float64), (u, i)), shape=(n_users, n_items))


def trust_matrix(n_users, truster, trustee, row_normalise=True):
    """T[a, b] = 1 if a trusts b (a's 'friends' are the users a chose to trust)."""
    T = sp.csr_matrix((np.ones(len(truster)), (truster, trustee)), shape=(n_users, n_users))
    T.data[:] = 1.0
    if row_normalise:
        deg = np.asarray(T.sum(axis=1)).ravel()
        T = sp.diags(1.0 / np.maximum(deg, 1)) @ T
    return T.tocsr()


# --------------------------------------------------------------------------- popularity
def fit_popularity(R, relevance_threshold=4.0):
    pos = np.asarray((R >= relevance_threshold).sum(axis=0)).ravel().astype(float)
    cnt = np.asarray((R != 0).sum(axis=0)).ravel()
    return {"type": "pop", "item_score": pos + 1e-3 * cnt, "global_mean": R.data.mean()}


# --------------------------------------------------------------------------- user kNN
def fit_user_knn(R, k=50, shrink=10.0, damping=5.0, centering="baseline", chunk=1000):
    """Pre-compute the top-k neighbour graph S (sparse, rows = users).

    Similarity = cosine of the residual (centred) rating vectors, multiplied by
    n_common / (n_common + shrink) so that similarities supported by very few
    co-rated items are distrusted. Only positive similarities are kept.
    """
    n_users = R.shape[0]
    M, user_base, item_base, global_mean, Rc = _centre(R, centering)
    norms = np.sqrt(np.asarray(Rc.multiply(Rc).sum(axis=1)).ravel())
    RcT, MT = Rc.T.tocsr(), M.T.tocsr()
    rows, cols, vals = [], [], []
    for start in range(0, n_users, chunk):
        c = np.arange(start, min(start + chunk, n_users))
        dot = (Rc[c] @ RcT).toarray()
        common = (M[c] @ MT).toarray()
        sim = dot / (norms[c, None] * norms[None, :] + 1e-12)
        sim *= common / (common + shrink)          # distrust similarities based on few co-ratings
        sim[np.arange(len(c)), c] = 0.0
        kk = min(k, n_users - 1)
        top = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        tv = np.take_along_axis(sim, top, axis=1)
        keep = tv > 0
        rows.append(np.repeat(c, kk)[keep.ravel()])
        cols.append(top.ravel()[keep.ravel()])
        vals.append(tv.ravel()[keep.ravel()])
    S = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n_users, n_users))
    L = (R >= 4.0).astype(np.float64).tocsr()          # neighbours' positive feedback, used for ranking
    return {"type": "knn", "S": S, "Rc": Rc, "M": M, "L": L, "user_base": user_base, "item_base": item_base,
            "global_mean": global_mean, "centering": centering, "damping": damping, "k": k}


def knn_from_neighbours(R, S, damping=0.0):
    """kNN predictor with an externally supplied neighbour/weight matrix S.
    Used to compare friends vs similar strangers vs random users (Req. 3)."""
    M, user_base, item_base, global_mean, Rc = _centre(R, "baseline")
    return {"type": "knn", "S": S.tocsr(), "Rc": Rc, "M": M, "user_base": user_base, "item_base": item_base,
            "global_mean": global_mean, "centering": "baseline", "damping": damping, "k": None}


def _centre(R, centering, reg_bias=3.0):
    """Binary mask, per-user/per-item base prediction and residual matrix.

    centering = 'baseline' : residual r - (mu + b_u + b_i)   (default)
                'user'     : residual r - mean_u   (classic mean-centred CF)
                'none'     : raw ratings (no correction of rating scales)
    """
    M = R.copy()
    M.data[:] = 1.0
    n_rated = np.asarray(M.sum(axis=1)).ravel()
    global_mean = R.data.mean()
    user_base = np.where(n_rated > 0, np.asarray(R.sum(axis=1)).ravel() / np.maximum(n_rated, 1), global_mean)
    item_base = np.zeros(R.shape[1])
    if centering == "baseline":
        b = fit_mf(R, n_factors=0, reg=reg_bias, reg_bias=reg_bias, n_iters=5)
        user_base, item_base = b["mu"] + b["bu"], b["bi"]
    Rc = R.copy().tocsr()
    if centering != "none":
        Rc.data = Rc.data - np.repeat(user_base, np.diff(Rc.indptr)) - item_base[Rc.indices]
    return M, user_base, item_base, global_mean, Rc


def knn_rows(model, users, return_support=False):
    S = model["S"][users]
    num = (S @ model["Rc"]).toarray()
    den = (abs(S) @ model["M"]).toarray()
    if model["centering"] != "none":
        # base + damped weighted neighbour residual; damping shrinks towards the
        # base when only a few (or weakly similar) neighbours rated the item
        base = model["user_base"][users, None] + model["item_base"][None, :]
        pred = base + num / (den + model["damping"] + 1e-12)
    else:
        pred = np.where(den > 0, num / np.maximum(den, 1e-12), model["global_mean"])
    pred = np.clip(pred, 1.0, 5.0)
    return (pred, den) if return_support else pred


# --------------------------------------------------------------------------- matrix factorisation
def _ridge_solve(M, Y, X, reg, reg_bias, beta_vec=None, prior=None):
    """Solve, for every row u of M at once, the ridge problem
        min_x  sum_{j in M[u]} (y_uj - X_j . x)^2 + reg ||x[:-1]||^2 + reg_bias x[-1]^2
               + beta_u ||x - prior_u||^2
    via the normal equations (X_u^T X_u + Reg + beta_u I) x = X_u^T y_u + beta_u prior_u.
    The last column of X is the constant 1 so the last coefficient is the bias.
    """
    d = X.shape[1]
    outer = (X[:, :, None] * X[:, None, :]).reshape(X.shape[0], d * d)
    A = np.asarray(M @ outer).reshape(-1, d, d)
    diag = np.arange(d - 1)
    A[:, diag, diag] += reg
    A[:, d - 1, d - 1] += reg_bias
    b = np.asarray(Y @ X)
    if beta_vec is not None:
        f = np.arange(d)
        A[:, f, f] += beta_vec[:, None]
        b += beta_vec[:, None] * prior
    sol = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    return sol[:, :d - 1], sol[:, d - 1]


def fit_mf(R, n_factors=10, reg=10.0, reg_bias=3.0, n_iters=15, seed=0, social=None, beta=0.0):
    """Biased MF trained with ALS; optional trust regularisation (SocialMF).

    The social term beta * ||[p_u, b_u] - mean_{f in trusted(u)} [p_f, b_f]||^2
    pulls a user's taste vector and rating bias towards those of the people
    they trust (Jamali & Ester 2010; Ma et al. 2011). The neighbours' average
    is taken from the previous iteration (block-coordinate ALS).

    n_factors = 0 gives the pure bias baseline mu + b_u + b_i.
    social : row-normalised trust matrix (a -> users that a trusts) or None.
    beta   : strength of the social term; 0 -> ordinary MF.
    """
    rng = np.random.default_rng(seed)
    n_users, n_items = R.shape
    coo = R.tocoo()
    rows, cols, vals = coo.row, coo.col, coo.data
    mu = vals.mean()
    P = 0.1 * rng.standard_normal((n_users, n_factors))
    Q = 0.1 * rng.standard_normal((n_items, n_factors))
    bu, bi = np.zeros(n_users), np.zeros(n_items)
    Mu = sp.csr_matrix((np.ones_like(vals), (rows, cols)), shape=(n_users, n_items))
    Mi = Mu.T.tocsr()
    beta_vec = None
    if social is not None and beta > 0:
        has_friends = (np.asarray(social.sum(axis=1)).ravel() > 0).astype(float)
        beta_vec = beta * has_friends
    for _ in range(n_iters):
        # user step: items fixed, targets with item bias removed
        Y = sp.csr_matrix((vals - mu - bi[cols], (rows, cols)), shape=(n_users, n_items))
        X = np.hstack([Q, np.ones((n_items, 1))])
        # social prior = average profile (factors and bias) of the users u trusts
        prior = social @ np.hstack([P, bu[:, None]]) if beta_vec is not None else None
        P, bu = _ridge_solve(Mu, Y, X, reg, reg_bias, beta_vec, prior)
        # item step: users fixed, targets with user bias removed
        Yt = sp.csr_matrix((vals - mu - bu[rows], (cols, rows)), shape=(n_items, n_users))
        X = np.hstack([P, np.ones((n_users, 1))])
        Q, bi = _ridge_solve(Mi, Yt, X, reg, reg_bias)
    return {"type": "mf", "mu": mu, "bu": bu, "bi": bi, "P": P, "Q": Q, "beta": beta, "reg": reg, "reg_bias": reg_bias,
            "n_factors": n_factors}


def mf_rows(model, users):
    pred = model["mu"] + model["bu"][users, None] + model["bi"][None, :] + model["P"][users] @ model["Q"].T
    return np.clip(pred, 1.0, 5.0)


# --------------------------------------------------------------------------- generic interface
def predict_rows(model, users):
    """Dense predicted ratings (or scores for 'pop') for the given users x all items."""
    users = np.asarray(users)
    if model["type"] == "knn":
        return knn_rows(model, users)
    if model["type"] == "mf":
        return mf_rows(model, users)
    if model["type"] == "pop":
        return np.tile(model["item_score"], (len(users), 1))
    raise ValueError(model["type"])


def rank_rows(model, users):
    """Scores used to build top-N lists.

    For kNN the classic top-N rule is used: score(u, i) = sum of similarities of
    u's neighbours who liked i (rated >= 4). A predicted RATING is a poor
    ranking signal because an item rated 5 by one weak neighbour gets the same
    prediction as one liked by twenty strong neighbours. MF / SocialMF rank by
    predicted rating (their native output) and popularity by its score.
    """
    users = np.asarray(users)
    if model["type"] == "knn" and "L" in model:
        return (model["S"][users] @ model["L"]).toarray() + 1e-6 * predict_rows(model, users)
    return predict_rows(model, users)


def predict_pairs(model, u, i, chunk=500):
    """Predicted rating for each (u[k], i[k]) pair."""
    out = np.empty(len(u))
    users = np.unique(u)
    for s in range(0, len(users), chunk):
        block = users[s:s + chunk]
        pred = predict_rows(model, block)
        pos = np.searchsorted(block, u)
        sel = np.isin(u, block)
        out[sel] = pred[pos[sel], i[sel]]
    return out


def top_n(scores, exclude, users, n):
    """Top-n item indices per row, never recommending items seen in training."""
    scores = scores.copy()
    ex = exclude[users].tocoo()
    scores[ex.row, ex.col] = -np.inf
    top = np.argpartition(-scores, n - 1, axis=1)[:, :n]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)


def recommend(model, R_train, users, n, chunk=500):
    lists = []
    for s in range(0, len(users), chunk):
        block = users[s:s + chunk]
        lists.append(top_n(rank_rows(model, block), R_train, block, n))
    return np.vstack(lists)
