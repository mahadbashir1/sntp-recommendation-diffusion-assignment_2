"""Requirement 6 - recommending one item to a group with conflicting tastes.

Aggregation strategies over the members' predicted ratings (m x items):
  * average satisfaction : maximise the mean predicted rating (utilitarian)
  * least misery         : maximise the minimum (protects the least happy member)
  * most pleasure        : maximise the maximum (pleases the most enthusiastic member)
"""
import numpy as np

STRATEGIES = {"average": lambda P: P.mean(axis=0),
              "least_misery": lambda P: P.min(axis=0),
              "most_pleasure": lambda P: P.max(axis=0)}


def social_groups(g_undirected, n_candidates, size, rng):
    """Groups made of a user plus (size-1) random direct contacts (friends
    deciding together, e.g. a shared watch party)."""
    indptr, indices = g_undirected
    users = np.where(np.diff(indptr) >= size - 1)[0]
    groups = []
    for u in rng.choice(users, min(n_candidates, len(users)), replace=False):
        nb = indices[indptr[u]:indptr[u + 1]]
        groups.append(np.concatenate([[u], rng.choice(nb, size - 1, replace=False)]))
    return groups


def conflict(P):
    """Taste conflict of a group over candidate items (members x items).

    Each member's own mean prediction is removed first: a member who rates
    everything one star lower shifts all scores equally and can never change
    which item a strategy picks. What remains is disagreement about the
    *relative* order of items; we average its across-member std over items.
    """
    Pc = P - P.mean(axis=1, keepdims=True)
    return float(Pc.std(axis=0).mean())


def recommend_for_group(P, candidates, strategy, n=1):
    score = STRATEGIES[strategy](P[:, candidates])
    order = np.argsort(-score, kind="stable")[:n]
    return candidates[order], score[order]


def satisfaction(true_pref, item):
    """Members' (simulated) true preference for the chosen item."""
    v = true_pref[:, item]
    return {"sat_mean": v.mean(), "sat_min": v.min(), "sat_max": v.max(), "sat_std": v.std(),
            "frac_satisfied": float(np.mean(v >= 4.0)), "frac_miserable": float(np.mean(v < 3.0))}
