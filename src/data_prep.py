"""Requirement 1 - represent the social environment (Epinions trust + ratings).

Observed data (Massa & Avesani Epinions dump):
  * ratings_data.txt : user_id item_id rating(1..5)      -> explicit preference signal
  * trust_data.txt   : truster_id trustee_id 1             -> directed "A trusts B"
A trust edge A -> B means A reads/relies on B's reviews, so *information and
influence flow from B to A*. The diffusion code therefore uses the reversed
edge B -> A ("influence graph").

Processing steps (all deterministic given the seed in config.json):
  1. drop self-trust statements;
  2. keep items with >= min_item_ratings ratings and users with
     >= min_user_ratings ratings on those items (removes items nobody can learn);
  3. keep users that take part in the trust network among kept users;
  4. snowball (BFS) sample max_users users from a random start so that the
     sample keeps the local social structure (a uniform node sample would
     destroy almost all edges because the graph is sparse);
  5. inside the sample keep items with >= min_item_ratings_sample ratings.
Users whose ratings are mostly dropped are *kept* as network nodes - they model
incomplete interaction histories (cold-start users).

Run:  python src/data_prep.py --profile full
"""
import argparse
import collections
import json
import os

import numpy as np
import pandas as pd

import utils

RAW_DIR = os.path.join(utils.ROOT, "data", "raw")


def load_raw(raw_dir=RAW_DIR):
    """Read the original Epinions text files (space separated, no header)."""
    ratings = pd.read_csv(os.path.join(raw_dir, "ratings_data.txt"), sep=r"\s+", header=None,
                          names=["user", "item", "rating"])
    trust = pd.read_csv(os.path.join(raw_dir, "trust_data.txt"), sep=r"\s+", header=None,
                        names=["truster", "trustee", "value"])
    return ratings, trust


def core_filter(ratings, trust, min_item_ratings, min_user_ratings):
    trust = trust[trust.truster != trust.trustee]
    item_cnt = ratings.item.value_counts()
    ratings = ratings[ratings.item.isin(item_cnt[item_cnt >= min_item_ratings].index)]
    user_cnt = ratings.user.value_counts()
    ratings = ratings[ratings.user.isin(user_cnt[user_cnt >= min_user_ratings].index)]
    users = set(ratings.user)
    trust = trust[trust.truster.isin(users) & trust.trustee.isin(users)]
    social_users = set(trust.truster) | set(trust.trustee)
    ratings = ratings[ratings.user.isin(social_users)]
    return ratings, trust


def bfs_sample(trust, max_users, rng):
    """Snowball sample on the undirected trust graph starting from a random user."""
    adj = collections.defaultdict(list)
    for a, b in zip(trust.truster.values, trust.trustee.values):
        adj[a].append(b)
        adj[b].append(a)
    nodes = np.array(sorted(adj))
    start = nodes[rng.integers(len(nodes))]
    seen, queue, order = {start}, collections.deque([start]), []
    while queue and len(order) < max_users:
        x = queue.popleft()
        order.append(x)
        nbrs = np.array(sorted(set(adj[x])))
        rng.shuffle(nbrs)
        for y in nbrs:
            if y not in seen:
                seen.add(y)
                queue.append(y)
    return order


def build_sample(ratings, trust, max_users, min_item_ratings_sample, seed):
    rng = np.random.default_rng(seed)
    users = bfs_sample(trust, max_users, rng)
    user_set = set(users)
    trust = trust[trust.truster.isin(user_set) & trust.trustee.isin(user_set)].drop_duplicates(["truster", "trustee"])
    ratings = ratings[ratings.user.isin(user_set)]
    item_cnt = ratings.item.value_counts()
    ratings = ratings[ratings.item.isin(item_cnt[item_cnt >= min_item_ratings_sample].index)]
    # contiguous indices
    user_ids = np.array(sorted(user_set))
    item_ids = np.array(sorted(ratings.item.unique()))
    uidx = {u: k for k, u in enumerate(user_ids)}
    iidx = {i: k for k, i in enumerate(item_ids)}
    r = pd.DataFrame({"u": ratings.user.map(uidx).values, "i": ratings.item.map(iidx).values,
                      "rating": ratings.rating.values.astype(np.float32)})
    t = pd.DataFrame({"truster": trust.truster.map(uidx).values, "trustee": trust.trustee.map(uidx).values})
    return r.sort_values(["u", "i"]).reset_index(drop=True), t, user_ids, item_ids


def prepare(profile):
    cfg = utils.load_config(profile)
    d = cfg["data"]
    utils.log(f"loading raw Epinions files from {RAW_DIR}")
    ratings, trust = load_raw()
    raw_stats = {"raw_ratings": len(ratings), "raw_users_with_ratings": int(ratings.user.nunique()),
                 "raw_items": int(ratings.item.nunique()), "raw_trust_edges": len(trust),
                 "raw_self_trust": int((trust.truster == trust.trustee).sum())}
    ratings, trust = core_filter(ratings, trust, d["min_item_ratings"], d["min_user_ratings"])
    core_stats = {"core_ratings": len(ratings), "core_users": int(ratings.user.nunique()),
                  "core_items": int(ratings.item.nunique()), "core_trust_edges": len(trust)}
    r, t, user_ids, item_ids = build_sample(ratings, trust, d["max_users"], d["min_item_ratings_sample"], cfg["seed"])
    out = utils.ensure_dir(utils.processed_dir(profile))
    r.to_csv(os.path.join(out, "ratings.csv"), index=False)
    t.to_csv(os.path.join(out, "trust.csv"), index=False)
    pd.DataFrame({"u": np.arange(len(user_ids)), "epinions_user_id": user_ids}).to_csv(os.path.join(out, "users.csv"), index=False)
    pd.DataFrame({"i": np.arange(len(item_ids)), "epinions_item_id": item_ids}).to_csv(os.path.join(out, "items.csv"), index=False)
    per_user = r.groupby("u").size().reindex(range(len(user_ids)), fill_value=0)
    stats = dict(raw_stats, **core_stats)
    stats.update({
        "n_users": len(user_ids), "n_items": len(item_ids), "n_ratings": len(r), "n_trust_edges": len(t),
        "density": len(r) / (len(user_ids) * len(item_ids)),
        "users_without_ratings": int((per_user == 0).sum()),
        "users_with_lt5_ratings": int((per_user < 5).sum()),
        "median_ratings_per_user": float(per_user.median()),
        "rating_distribution": {str(int(k)): int(v) for k, v in r.rating.value_counts().sort_index().items()},
        "reciprocal_trust_fraction": reciprocity(t),
    })
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    utils.log(f"sample written to {out}: {stats['n_users']} users, {stats['n_items']} items, "
              f"{stats['n_ratings']} ratings, {stats['n_trust_edges']} trust edges")
    return stats


def reciprocity(t):
    pairs = set(zip(t.truster.values, t.trustee.values))
    return float(np.mean([(b, a) in pairs for a, b in pairs])) if pairs else 0.0


def load_dataset(profile):
    """Load the processed sample as a dict of numpy arrays."""
    folder = utils.processed_dir(profile)
    if not os.path.exists(os.path.join(folder, "ratings.csv")):
        prepare(profile)
    r = pd.read_csv(os.path.join(folder, "ratings.csv"))
    t = pd.read_csv(os.path.join(folder, "trust.csv"))
    with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return {"n_users": meta["n_users"], "n_items": meta["n_items"],
            "u": r.u.values.astype(np.int64), "i": r.i.values.astype(np.int64),
            "r": r.rating.values.astype(np.float64),
            "truster": t.truster.values.astype(np.int64), "trustee": t.trustee.values.astype(np.int64),
            "meta": meta}


def split_per_user(u, frac, seed, min_ratings=3):
    """Random per-user hold-out. Returns a boolean mask of held-out positions.

    Every user with >= min_ratings ratings keeps at least one rating on each
    side, so test users always have training history (no user leaks wholly to
    test) and evaluation is not dominated by users with no training data.
    """
    rng = np.random.default_rng(seed)
    key = rng.random(len(u))
    order = np.lexsort((key, u))
    u_sorted = u[order]
    counts = np.bincount(u_sorted, minlength=u.max() + 1)
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rank = np.arange(len(u)) - start[u_sorted]
    n_out = np.where(counts >= min_ratings, np.maximum(1, np.round(frac * counts)), 0).astype(int)
    held = np.zeros(len(u), dtype=bool)
    held[order] = rank < n_out[u_sorted]
    return held


def make_split(data, test_frac, seed, val_frac=0.0):
    """Return index arrays for train / (val) / test of the rating list."""
    test_mask = split_per_user(data["u"], test_frac, seed)
    idx_train = np.where(~test_mask)[0]
    idx_test = np.where(test_mask)[0]
    idx_val = np.array([], dtype=np.int64)
    if val_frac > 0:
        val_mask = split_per_user(data["u"][idx_train], val_frac, seed + 1000)
        idx_val = idx_train[val_mask]
        idx_train = idx_train[~val_mask]
    return {"train": idx_train, "val": idx_val, "test": idx_test}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=["demo", "full"])
    args = ap.parse_args()
    prepare(args.profile)
