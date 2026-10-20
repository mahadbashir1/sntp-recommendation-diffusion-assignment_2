"""Requirement 5 - choosing whom to promote an item to under a budget k.

Strategy: CELF lazy-greedy on Monte-Carlo "live-edge worlds" (snapshots).
  * Under IC, a diffusion run is equivalent to flipping every edge coin in
    advance: edge u->v is "live" with probability attn_uv * a_v(i), and a
    promoted user v adopts on the recommendation itself with probability a_v(i).
    The adopters are then exactly the users reachable from adopting seeds over
    live edges. We sample R such worlds once and reuse them for every
    candidate (common random numbers -> candidate comparisons are much less
    noisy than with independent simulations).
  * The marginal gain of candidate c given the current seed set S is
    gain_w(c) = |reach_w(c) \\ reach_w(S)| in world w. It counts only the
    ADDITIONAL users c brings, so a hub whose audience overlaps with already
    chosen seeds gets a small gain (unlike degree-based selection).
  * Uncertainty: the R per-world gains give a mean and a standard error. The
    selection score can be the mean (default) or a lower confidence bound
    mean - z * SE, which prefers candidates whose contribution is reliably
    large rather than large only in a few lucky worlds.
  * Expected spread is monotone and submodular, so a candidate's marginal gain
    can only shrink as S grows. CELF keeps candidates in a max-heap keyed by
    their last computed gain and only re-evaluates the top one -> the same
    result as plain greedy with far fewer evaluations (Leskovec et al. 2007).

Baselines: random users and highest out-degree (users trusted by most people).
A third reference, "top individual spread", ranks users by their stand-alone
expected spread and ignores overlap between their audiences.
"""
import heapq

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

import diffusion as dif


def sample_worlds(g, attn, adopt_p, n_worlds, rng):
    p_live = attn * adopt_p[g["dst"]]
    worlds = []
    for _ in range(n_worlds):
        live = rng.random(len(p_live)) < p_live
        cnt = np.bincount(g["src"][live], minlength=g["n"])
        worlds.append({"indptr": np.concatenate([[0], np.cumsum(cnt)]), "indices": g["dst"][live],
                       "seed_live": rng.random(g["n"]) < adopt_p, "out": cnt})
    return worlds


def reach(world, sources, blocked):
    """Users newly reached from `sources` over live edges, never entering `blocked`.
    Anything reachable from an already covered user is itself covered, so the
    BFS can stop at covered users - this pruning keeps marginal-gain
    evaluations cheap once a few seeds are chosen."""
    visited = blocked.copy()
    frontier = np.unique(sources[~visited[sources]])
    visited[frontier] = True
    found = [frontier]
    while frontier.size:
        nb = world["indices"][dif.out_edges(world["indptr"], frontier)]
        nb = np.unique(nb[~visited[nb]])
        visited[nb] = True
        found.append(nb)
        frontier = nb
    return np.concatenate(found)


def candidate_gains(worlds, c, covered):
    g = np.zeros(len(worlds))
    for w, world in enumerate(worlds):
        if world["seed_live"][c] and not covered[w, c]:
            if world["out"][c] == 0:
                g[w] = 1.0
            else:
                g[w] = len(reach(world, np.array([c]), covered[w]))
    return g


def initial_gains(worlds, n, use_scc=True):
    """Stand-alone gain of every user in every world.

    Fast paths: a user without live out-edges reaches only themself; and
    (use_scc=True) all users in the same strongly connected component of a
    live-edge world reach exactly the same set, so one BFS per component is
    enough. In the viral regime one giant component holds most users, which
    turns thousands of identical BFS runs into one (measured in cost_analysis).
    """
    G = np.zeros((len(worlds), n))
    empty = np.zeros(n, dtype=bool)
    for w, world in enumerate(worlds):
        live = world["seed_live"]
        if not use_scc:
            G[w] = live.astype(float)
            for c in np.where(live & (world["out"] > 0))[0]:
                G[w, c] = len(reach(world, np.array([c]), empty))
            continue
        A = sp.csr_matrix((np.ones(len(world["indices"])), world["indices"], world["indptr"]), shape=(n, n))
        n_comp, label = connected_components(A, directed=True, connection="strong")
        _, rep = np.unique(label, return_index=True)          # one representative user per component
        need = np.bincount(label, weights=live, minlength=n_comp) > 0
        size = np.ones(n_comp)
        for c in np.where(need & (world["out"][rep] > 0))[0]:
            size[c] = len(reach(world, np.array([rep[c]]), empty))
        G[w] = np.where(live, size[label], 0.0)
    return G


def _score(g, mode, z):
    m = g.mean()
    if mode == "lcb":
        return m - z * g.std(ddof=1) / np.sqrt(len(g))
    return m


def celf(worlds, n, k, mode="mean", z=1.96, G0=None):
    """Lazy greedy seed selection. Returns seeds, per-step diagnostics and the
    number of marginal-gain evaluations performed."""
    R = len(worlds)
    covered = np.zeros((R, n), dtype=bool)
    G0 = initial_gains(worlds, n) if G0 is None else G0
    evals = n
    last = {c: G0[:, c] for c in range(n)}       # most recent per-world gains
    heap = [(-_score(G0[:, c], mode, z), c, 0) for c in range(n)]
    heapq.heapify(heap)
    seeds, steps = [], []
    while len(seeds) < k and heap:
        neg, c, stamp = heapq.heappop(heap)
        if stamp == len(seeds):
            # gain is up to date w.r.t. the current seed set and still the best
            g = last[c]
            seeds.append(c)
            for w, world in enumerate(worlds):
                if world["seed_live"][c]:
                    covered[w, reach(world, np.array([c]), covered[w])] = True
            se = g.std(ddof=1) / np.sqrt(R)
            steps.append({"step": len(seeds), "seed": int(c), "gain_mean": g.mean(), "gain_se": se,
                          "gain_ci95": 1.96 * se, "spread_est": covered.sum(axis=1).mean()})
        else:
            last[c] = candidate_gains(worlds, c, covered)
            evals += 1
            heapq.heappush(heap, (-_score(last[c], mode, z), c, len(seeds)))
    return seeds, steps, evals


def naive_greedy(worlds, n, k, G0=None):
    """Plain greedy (re-evaluates every candidate every round) - used only to
    measure the benefit of the CELF optimisation."""
    R = len(worlds)
    covered = np.zeros((R, n), dtype=bool)
    G0 = initial_gains(worlds, n) if G0 is None else G0
    evals = n
    seeds = [int(np.argmax(G0.mean(axis=0)))]
    for w, world in enumerate(worlds):
        if world["seed_live"][seeds[0]]:
            covered[w, reach(world, np.array([seeds[0]]), covered[w])] = True
    while len(seeds) < k:
        best, best_c = -1.0, -1
        for c in range(n):
            if c in seeds:
                continue
            m = candidate_gains(worlds, c, covered).mean()
            evals += 1
            if m > best:
                best, best_c = m, c
        seeds.append(best_c)
        for w, world in enumerate(worlds):
            if world["seed_live"][best_c]:
                covered[w, reach(world, np.array([best_c]), covered[w])] = True
    return seeds, evals


def degree_seeds(g, k):
    return list(np.argsort(-g["out_deg"], kind="stable")[:k])


def random_seeds(n, k, rng):
    return list(rng.choice(n, k, replace=False))


def top_individual_seeds(G0, k):
    return list(np.argsort(-G0.mean(axis=0), kind="stable")[:k])


def world_spread(worlds, seeds, n):
    seeds = np.asarray(seeds)
    tot = []
    for world in worlds:
        s = seeds[world["seed_live"][seeds]]
        tot.append(len(reach(world, s, np.zeros(n, dtype=bool))) if s.size else 0)
    return np.array(tot, dtype=float)


def audience_overlap(g, seeds):
    """Mean pairwise Jaccard similarity of the seeds' direct audiences (followers)."""
    aud = [set(g["dst"][g["indptr"][s]:g["indptr"][s + 1]]) for s in seeds]
    vals = [len(a & b) / max(1, len(a | b)) for x, a in enumerate(aud) for b in aud[x + 1:]]
    return float(np.mean(vals)) if vals else 0.0
