"""Requirement 8 (second half) - runtime / memory scaling and the CELF optimisation.

Runtime is measured with time.perf_counter without memory tracing; peak memory
is measured in a separate run of the same operation under tracemalloc (tracing
slows Python code down, so the two are never mixed).
"""
import numpy as np
import pandas as pd

import data_prep as dp
import diffusion as dif
import influence_max as im
import recommenders as rec
import utils
from experiments import PALETTE


def _profile(name, fn, rows, extra, memory=True):
    out, secs, _ = utils.measure(fn)
    peak = utils.measure(fn, track_memory=True)[2] if memory else np.nan
    rows.append(dict(extra, operation=name, seconds=secs, peak_mb=peak))
    return out


def _pipeline_ops(R, T, truster, trustee, hp, cfg, rows, extra, k=10):
    """Run and profile every major stage of the system on one instance."""
    dc = cfg["diffusion"]
    n = R.shape[0]
    _profile("knn_fit", lambda: rec.fit_user_knn(R, k=hp["knn_k"], shrink=hp["knn_shrink"], damping=hp["knn_damping"]), rows, extra)
    mf = _profile("mf_fit", lambda: rec.fit_mf(R, hp["mf_factors"], hp["mf_reg"], hp["reg_bias"], hp["mf_iters"]), rows, extra)
    _profile("social_mf_fit", lambda: rec.fit_mf(R, hp["mf_factors"], hp["social_reg"], hp["reg_bias"], hp["mf_iters"],
                                                 social=T, beta=hp["social_beta"]), rows, extra)
    g = _profile("graph_build", lambda: dif.influence_graph(n, truster, trustee, R), rows, extra)
    item = int(np.argmax(np.diff(R.tocsc().indptr)))
    a = dif.adoption_prob(rec.mf_rows(mf, np.arange(n))[:, item], dc["kappa"], dc["theta"])
    attn = dif.attention_prob(g, dc["lambda"])
    seeds = np.random.default_rng(0).choice(n, min(20, n), replace=False)
    _profile("ic_100_simulations", lambda: dif.run_many(g, attn, a, seeds, 100, np.random.default_rng(1)), rows, extra)
    worlds = im.sample_worlds(g, attn, a, cfg["cost"]["n_worlds"], np.random.default_rng(2))
    G0 = _profile("im_initial_gains", lambda: im.initial_gains(worlds, n), rows, extra)
    _profile(f"im_celf_k{k}", lambda: im.celf(worlds, n, k, G0=G0), rows, extra)
    return g, worlds, G0


def exp_cost(ctx):
    cfg, hp = ctx["cfg"], ctx["hp"]
    cc = cfg["cost"]
    plt = utils.setup_matplotlib()
    ratings, trust = dp.load_raw()
    core_r, core_t = dp.core_filter(ratings, trust, cfg["data"]["min_item_ratings"], cfg["data"]["min_user_ratings"])
    # ---- (1) growing network -------------------------------------------------
    net_rows = []
    for N in cc["network_sizes"]:
        r, t, _, _ = dp.build_sample(core_r, core_t, N, cfg["data"]["min_item_ratings_sample"], cfg["seed"])
        R = rec.rating_matrix(N, int(r.i.max()) + 1, r.u.values, r.i.values, r.rating.values)
        T = rec.trust_matrix(N, t.truster.values, t.trustee.values)
        extra = {"users": N, "items": R.shape[1], "ratings": R.nnz, "edges": len(t)}
        _pipeline_ops(R, T, t.truster.values, t.trustee.values, hp, cfg, net_rows, extra)
        utils.log(f"cost: network size {N} done")
    net = pd.DataFrame(net_rows)
    utils.save_table(net, cfg["profile"], "cost_network_scaling")
    # ---- (2) growing item collection (fixed users) -----------------------------
    item_rows = []
    R_all, n = ctx["R_train"], ctx["data"]["n_users"]
    order = np.argsort(-np.diff(R_all.tocsc().indptr), kind="stable")
    users = np.arange(n)
    for I in cc["item_sizes"]:
        R = R_all[:, np.sort(order[:min(I, len(order))])].tocsr()
        extra = {"items": R.shape[1], "ratings": R.nnz}
        mf = _profile("mf_fit", lambda: rec.fit_mf(R, hp["mf_factors"], hp["mf_reg"], hp["reg_bias"], hp["mf_iters"]), item_rows, extra)
        knn = _profile("knn_fit", lambda: rec.fit_user_knn(R, k=hp["knn_k"], shrink=hp["knn_shrink"]), item_rows, extra)
        _profile("mf_score_top10_all_users", lambda: rec.recommend(mf, R, users, 10), item_rows, extra)
        _profile("knn_score_top10_all_users", lambda: rec.recommend(knn, R, users, 10), item_rows, extra)
    items = pd.DataFrame(item_rows)
    utils.save_table(items, cfg["profile"], "cost_item_scaling")
    # ---- (3) growing promotion budget on the main network ------------------------
    g = ctx["graph"]
    dc = cfg["diffusion"]
    item = ctx.get("campaign_items", {}).get("medium", int(order[0]))
    a = dif.adoption_prob(ctx["oracle"][:, item], dc["kappa"], dc["theta"])
    attn = dif.attention_prob(g, dc["lambda"])
    worlds = im.sample_worlds(g, attn, a, cc["n_worlds"], np.random.default_rng(3))
    G0, t_g0, _ = utils.measure(im.initial_gains, worlds, n)
    bud_rows = []
    for k in cc["budgets"]:
        (seeds, _, evals), secs, _ = utils.measure(im.celf, worlds, n, k, G0=G0)
        peak = utils.measure(im.celf, worlds, n, k, G0=G0, track_memory=True)[2]
        bud_rows.append({"budget": k, "celf_seconds_excl_initial": secs, "initial_gain_seconds": t_g0,
                         "gain_evaluations": evals, "peak_mb": peak})
    bud = pd.DataFrame(bud_rows)
    utils.save_table(bud, cfg["profile"], "cost_budget_scaling")
    # ---- (4) optimisation: CELF vs naive greedy (identical worlds) ----------------
    N = cc["naive_network_size"]
    r, t, _, _ = dp.build_sample(core_r, core_t, N, cfg["data"]["min_item_ratings_sample"], cfg["seed"])
    R = rec.rating_matrix(N, int(r.i.max()) + 1, r.u.values, r.i.values, r.rating.values)
    gs = dif.influence_graph(N, t.truster.values, t.trustee.values, R)
    mf = rec.fit_mf(R, hp["mf_factors"], hp["mf_reg"], hp["reg_bias"], hp["mf_iters"])
    it = int(np.argsort(-np.diff(R.tocsc().indptr))[len(np.diff(R.tocsc().indptr)) // 20])
    a_s = dif.adoption_prob(rec.mf_rows(mf, np.arange(N))[:, it], dc["kappa"], dc["theta"])
    ws = im.sample_worlds(gs, dif.attention_prob(gs, dc["lambda"]), a_s, cc["n_worlds"], np.random.default_rng(4))
    G0s = im.initial_gains(ws, N)
    opt_rows = []
    for k in range(1, cc["naive_budget"] + 1):
        (s_naive, ev_naive), t_naive, _ = utils.measure(im.naive_greedy, ws, N, k, G0=G0s)
        (s_celf, _, ev_celf), t_celf, _ = utils.measure(im.celf, ws, N, k, G0=G0s)
        opt_rows.append({"users": N, "budget": k, "naive_seconds": t_naive, "celf_seconds": t_celf,
                         "naive_evaluations": ev_naive, "celf_evaluations": ev_celf, "speedup": t_naive / max(t_celf, 1e-9),
                         "same_seed_set": set(map(int, s_naive)) == set(map(int, s_celf)),
                         "naive_spread_est": im.world_spread(ws, s_naive, N).mean(),
                         "celf_spread_est": im.world_spread(ws, s_celf, N).mean()})
    opt = pd.DataFrame(opt_rows)
    utils.save_table(opt, cfg["profile"], "cost_celf_vs_naive")
    # ---- (5) optimisation: one BFS per strongly connected component ----------------
    scc_rows = []
    for lam in (dc["lambda"], 2 * dc["lambda"]):
        w = im.sample_worlds(g, dif.attention_prob(g, lam), a, cc["n_worlds"], np.random.default_rng(5))
        G_fast, t_fast, _ = utils.measure(im.initial_gains, w, n, use_scc=True)
        G_slow, t_slow, _ = utils.measure(im.initial_gains, w, n, use_scc=False)
        scc_rows.append({"lam": lam, "worlds": len(w), "seconds_one_bfs_per_user": t_slow, "seconds_one_bfs_per_scc": t_fast,
                         "speedup": t_slow / t_fast, "identical_gains": bool(np.array_equal(G_fast, G_slow)),
                         "max_individual_spread": G_fast.mean(axis=0).max()})
    utils.save_table(pd.DataFrame(scc_rows), cfg["profile"], "cost_scc_optimisation")
    # ---- bottleneck breakdown on the main instance --------------------------------
    per_eval = opt.naive_seconds.iloc[-1] / max(opt.naive_evaluations.iloc[-1] - N, 1)
    k_full = max(cfg["im"]["budgets"])
    bott = net[net.users == max(cc["network_sizes"])][["operation", "seconds", "peak_mb"]].copy()
    bott = pd.concat([bott, pd.DataFrame([{"operation": f"naive_greedy_k{k_full}_estimated (main network)",
                                           "seconds": per_eval * n * k_full * (len(worlds) / cc["n_worlds"])}])],
                     ignore_index=True)
    utils.save_table(bott, cfg["profile"], "cost_bottleneck")
    # ---- figures ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
    for j, op in enumerate(net.operation.unique()):
        d = net[net.operation == op]
        ax[0].plot(d.users, d.seconds, marker="o", ms=3, lw=1.5, color=PALETTE[j % 8], label=op)
        ax[1].plot(d.users, d.peak_mb, marker="o", ms=3, lw=1.5, color=PALETTE[j % 8], label=op)
    ax[0].set(xscale="log", yscale="log", xlabel="users in network", ylabel="seconds", title="Runtime vs network size")
    ax[1].set(xscale="log", yscale="log", xlabel="users in network", ylabel="peak MB (tracemalloc)", title="Memory vs network size")
    ax[0].legend(fontsize=6)
    for j, op in enumerate(items.operation.unique()):
        d = items[items.operation == op]
        ax[2].plot(d["items"], d.seconds, marker="o", ms=3, lw=1.5, color=PALETTE[j], label=op)
    ax[2].set(xscale="log", yscale="log", xlabel="items in catalogue", ylabel="seconds", title="Runtime vs catalogue size")
    ax[2].legend(fontsize=6)
    ax[3].plot(opt.budget, opt.naive_seconds, marker="o", ms=3, lw=1.5, color=PALETTE[0], label="naive greedy")
    ax[3].plot(opt.budget, opt.celf_seconds, marker="o", ms=3, lw=1.5, color=PALETTE[1], label="CELF (lazy)")
    ax[3].set(yscale="log", xlabel="budget k", ylabel="seconds (excl. initial gains)", title=f"CELF optimisation ({N} users)")
    ax[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(utils.figure_path(cfg["profile"], "fig10_cost"))
    plt.close(fig)
    return {"celf_speedup_at_max_k": float(opt.speedup.iloc[-1]), "same_seeds_all": bool(opt.same_seed_set.all()),
            "naive_evals": int(opt.naive_evaluations.iloc[-1]), "celf_evals": int(opt.celf_evaluations.iloc[-1]),
            "scc_speedup": [r["speedup"] for r in scc_rows],
            "bottleneck": bott.to_dict("records")}
