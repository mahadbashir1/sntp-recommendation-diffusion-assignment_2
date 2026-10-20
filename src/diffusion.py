"""Requirement 4 - adoption and information spread.

Influence graph
  Trust edge a -> b (a trusts b) becomes influence edge b -> a: when b adopts
  an item and shares it, a may see it in their feed.

Edge strength (inferred, not observed)
  s_ba = (#co-rated items on which a and b agree within +-1 star + 1) / (#co-rated + 2)
  computed from TRAINING ratings only (Laplace smoothing -> 0.5 with no evidence).
  w_ba = s_ba / sum_{b' trusted by a} s_b'a     (weighted-cascade normalisation:
  the total social pull on a user is 1, so users who trust many people are
  influenced less by each of them).

Three distinct events in the platform
  1. RECEIVE a recommendation : platform pushes item i to seed user v.
  2. EXPOSED via a neighbour  : an in-neighbour u adopted i and shared it, and
                                v noticed the share; probability attn_uv = min(1, lambda * w_uv).
  3. ADOPT                    : after receiving/being exposed, v adopts with
                                probability a_v(i) = sigmoid(kappa * (pref_vi - theta)),
                                where pref_vi is v's (simulated) preference.
  lambda = global influence strength, kappa/theta = how strongly preferences gate adoption.

Models
  * Independent Cascade (IC): a new adopter is "active" for exactly one step
    and gets a single chance to expose each follower.
  * SIR epidemic: S = has not adopted, I = adopted and still actively sharing,
    R = adopted but no longer sharing. Each step every I node exposes its
    followers, then recovers with probability gamma (mean sharing time 1/gamma).
    gamma = 1 reduces SIR to IC, so the difference between the models is the
    *repeated* exposure attempts during the infectious period.
"""
import numpy as np


# --------------------------------------------------------------------------- graph construction
def edge_agreement(src, dst, R_train, tolerance=1.0):
    """Smoothed fraction of co-rated items on which src and dst agree."""
    R = R_train.tocsr()
    M = R.copy()
    M.data[:] = 1.0
    common = np.asarray(M[src].multiply(M[dst]).sum(axis=1)).ravel()
    agree = np.zeros(len(src))
    levels = np.arange(1, 6)
    onehot = {}
    for x in levels:
        m = R.copy()
        m.data = (np.abs(m.data - x) < 0.5).astype(float)
        m.eliminate_zeros()
        onehot[x] = m
    for x in levels:
        near = sum(onehot[y] for y in levels if abs(x - y) <= tolerance)
        agree += np.asarray(onehot[x][src].multiply(near[dst]).sum(axis=1)).ravel()
    return (agree + 1.0) / (common + 2.0), common


def assemble_graph(n, src, dst, s):
    """Sort edges by source (CSR layout) and compute weighted-cascade weights."""
    keep = src != dst
    src, dst, s = src[keep], dst[keep], s[keep]
    key = src.astype(np.int64) * n + dst
    _, first = np.unique(key, return_index=True)          # drop duplicate edges
    src, dst, s = src[first], dst[first], s[first]
    order = np.lexsort((dst, src))
    src, dst, s = src[order], dst[order], s[order]
    in_total = np.bincount(dst, weights=s, minlength=n)
    w = s / in_total[dst]
    out_deg = np.bincount(src, minlength=n)
    indptr = np.concatenate([[0], np.cumsum(out_deg)])
    return {"n": n, "src": src, "dst": dst, "s": s, "w": w, "indptr": indptr,
            "out_deg": out_deg, "in_deg": np.bincount(dst, minlength=n)}


def influence_graph(n_users, truster, trustee, R_train, tolerance=1.0):
    src, dst = trustee.astype(np.int64), truster.astype(np.int64)   # influence flows trustee -> truster
    s, _ = edge_agreement(src, dst, R_train, tolerance)
    return assemble_graph(n_users, src, dst, s)


def rewire_degree_preserving(g, rng):
    """Configuration-model null graph: every user keeps its in- and out-degree
    but who-follows-whom (and therefore clustering / audience overlap) is randomised."""
    dst = rng.permutation(g["dst"])
    return assemble_graph(g["n"], g["src"].copy(), dst, rng.permutation(g["s"]))


def random_graph(g, rng):
    """Erdos-Renyi null graph with the same number of nodes and edges."""
    m = len(g["src"])
    src = rng.integers(0, g["n"], m)
    dst = rng.integers(0, g["n"], m)
    return assemble_graph(g["n"], src, dst, rng.permutation(g["s"]))


def attention_prob(g, lam):
    return np.minimum(1.0, lam * g["w"])


def adoption_prob(pref, kappa=2.0, theta=3.5):
    """Probability that a user adopts an item they received/were exposed to."""
    return 1.0 / (1.0 + np.exp(-kappa * (np.asarray(pref, dtype=float) - theta)))


# --------------------------------------------------------------------------- simulation
def out_edges(indptr, nodes):
    """Indices of all outgoing edges of `nodes` (edges are stored grouped by source)."""
    starts, ends = indptr[nodes], indptr[nodes + 1]
    lens = ends - starts
    total = lens.sum()
    if total == 0:
        return np.empty(0, dtype=np.int64)
    offsets = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
    return offsets + np.arange(total)


def _cap_fanout(edges, src, m, rng):
    """Keep at most m randomly chosen outgoing edges per sharing user."""
    order = np.lexsort((rng.random(len(edges)), src[edges]))
    e = edges[order]
    s = src[e]
    first = np.concatenate([[True], s[1:] != s[:-1]])
    group_start = np.maximum.accumulate(np.where(first, np.arange(len(e)), 0))
    return e[(np.arange(len(e)) - group_start) < m]


def simulate(g, attn, adopt_p, seeds, rng, gamma=1.0, max_steps=200, max_fanout=0,
             receive_mask=None, share_mask=None, return_nodes=False):
    """One stochastic diffusion run (IC if gamma == 1, otherwise discrete-time SIR).

    attn         : per-edge probability that the follower notices a share
    adopt_p      : per-user adoption probability for this item
    seeds        : users who RECEIVE the platform's recommendation at t = 0
    max_fanout   : intervention - each share reaches at most this many followers (0 = no cap)
    receive_mask : intervention - users allowed to be exposed through neighbours
    share_mask   : intervention - users allowed to share (blocked accounts = False)
    """
    n = g["n"]
    seeds = np.unique(np.asarray(seeds, dtype=np.int64))
    adopted = np.zeros(n, dtype=bool)
    exposed = np.zeros(n, dtype=bool)
    t_adopt = np.full(n, -1)
    exposed[seeds] = True
    first = seeds[rng.random(len(seeds)) < adopt_p[seeds]]
    adopted[first] = True
    t_adopt[first] = 0
    infectious = first if share_mask is None else first[share_mask[first]]
    curve = [len(first)]
    exposure_events = 0
    step = 0
    while infectious.size and step < max_steps:
        step += 1
        edges = out_edges(g["indptr"], infectious)
        if max_fanout and edges.size:
            edges = _cap_fanout(edges, g["src"], max_fanout, rng)
        targets = g["dst"][edges]
        if receive_mask is not None:
            ok = receive_mask[targets]
            edges, targets = edges[ok], targets[ok]
        seen = rng.random(len(edges)) < attn[edges]
        cand = targets[seen]
        cand = cand[~adopted[cand]]
        exposure_events += len(cand)
        exposed[cand] = True
        new = np.unique(cand[rng.random(len(cand)) < adopt_p[cand]])
        adopted[new] = True
        t_adopt[new] = step
        curve.append(len(new))
        if share_mask is not None:
            new = new[share_mask[new]]
        if gamma >= 1.0:
            infectious = new
        else:
            still = infectious[rng.random(len(infectious)) >= gamma]
            infectious = np.concatenate([still, new])
    curve = np.cumsum(curve)
    final = curve[-1]
    t90 = int(np.argmax(curve >= 0.9 * final)) if final > 0 else 0
    exposed[seeds] = False          # count only users reached *through neighbours*
    out = {"adopters": int(final), "seed_adopters": int(len(first)), "received": int(len(seeds)),
           "exposed_social": int(exposed.sum()),
           "exposure_events": int(exposure_events), "steps": step, "t90": t90, "curve": curve}
    if return_nodes:
        out["adopted_mask"] = adopted
        out["exposed_mask"] = exposed
    return out


def run_many(g, attn, adopt_p, seeds, n_sims, rng, **kw):
    """Repeat `simulate` n_sims times; returns arrays of per-run outcomes."""
    runs = [simulate(g, attn, adopt_p, seeds, rng, **kw) for _ in range(n_sims)]
    res = {k: np.array([r[k] for r in runs]) for k in ("adopters", "seed_adopters", "exposed_social",
                                                      "exposure_events", "steps", "t90")}
    max_len = max(len(r["curve"]) for r in runs)
    res["curves"] = np.array([np.pad(r["curve"], (0, max_len - len(r["curve"])), mode="edge") for r in runs])
    if kw.get("return_nodes"):
        res["adopted_masks"] = [r["adopted_mask"] for r in runs]
        res["exposed_masks"] = [r["exposed_mask"] for r in runs]
    return res
