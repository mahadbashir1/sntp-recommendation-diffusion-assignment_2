"""Experiments for requirements 1-8. Each exp_* function writes CSV tables to
results/<profile>/tables, PNG figures to results/<profile>/figures and returns
a small dict of headline numbers (collected in results/<profile>/summary.json).

Observed vs simulated outcomes
  * Recommendation metrics (RMSE, precision, ...) are computed on OBSERVED
    held-out Epinions ratings -> they support claims about real users.
  * Diffusion outcomes come from the SIMULATOR (IC / SIR) driven by a
    simulated "true preference" matrix -> they support only comparative,
    what-if conclusions under the stated model assumptions.
"""
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

import data_prep as dp
import diffusion as dif
import group_rec as grp
import influence_max as im
import metrics as met
import recommenders as rec
import utils

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_ORDER = ["popularity", "bias", "knn", "mf", "social_mf", "knn_usermean", "knn_raw"]


# =========================================================================== shared context
def fit_models(R, T, hp, seed, names=("popularity", "bias", "knn", "mf", "social_mf")):
    """Fit the requested models on training matrix R; returns {name: (model, seconds)}."""
    makers = {
        "popularity": lambda: rec.fit_popularity(R),
        "bias": lambda: rec.fit_mf(R, 0, reg_bias=hp["reg_bias"], n_iters=5),
        "knn": lambda: rec.fit_user_knn(R, k=hp["knn_k"], shrink=hp["knn_shrink"], damping=hp["knn_damping"]),
        "knn_usermean": lambda: rec.fit_user_knn(R, k=hp["knn_k"], shrink=hp["knn_shrink"],
                                                 damping=hp["knn_damping"], centering="user"),
        "knn_raw": lambda: rec.fit_user_knn(R, k=hp["knn_k"], shrink=hp["knn_shrink"], centering="none"),
        "mf": lambda: rec.fit_mf(R, hp["mf_factors"], hp["mf_reg"], hp["reg_bias"], hp["mf_iters"], seed),
        "social_mf": lambda: rec.fit_mf(R, hp["mf_factors"], hp["social_reg"], hp["reg_bias"], hp["mf_iters"],
                                        seed, social=T, beta=hp["social_beta"]),
    }
    out = {}
    for name in names:
        t0 = time.perf_counter()
        out[name] = (makers[name](), time.perf_counter() - t0)
    return out


def split_matrices(data, idx):
    return rec.rating_matrix(data["n_users"], data["n_items"], data["u"][idx], data["i"][idx], data["r"][idx])


def exp_tuning(ctx):
    """Hyper-parameters chosen on a validation split carved out of the training
    data of split seed 0 - the test ratings are never used for tuning."""
    cfg, data = ctx["cfg"], ctx["data"]
    rc = cfg["rec"]
    s = dp.make_split(data, cfg["split"]["test_frac"], cfg["split"]["split_seeds"][0], cfg["split"]["val_frac"])
    R = split_matrices(data, s["train"])
    v = s["val"]
    uv, iv, yv = data["u"][v], data["i"][v], data["r"][v]
    rows = []
    for k in rc["knn_k_grid"]:
        for damp in rc["knn_damping_grid"]:
            m = rec.fit_user_knn(R, k=k, shrink=rc["knn_shrink"], damping=damp)
            rows.append({"model": "knn", "k": k, "damping": damp, "val_rmse": met.rmse(rec.predict_pairs(m, uv, iv), yv)})
    for f in rc["mf_factors_grid"]:
        for reg in rc["mf_reg_grid"]:
            m = rec.fit_mf(R, f, reg, rc["mf_reg_bias"], rc["mf_iters"], 0)
            rows.append({"model": "mf", "factors": f, "reg": reg, "val_rmse": met.rmse(rec.predict_pairs(m, uv, iv), yv)})
    tab = pd.DataFrame(rows)
    best_knn = tab[tab.model == "knn"].sort_values("val_rmse").iloc[0]
    best_mf = tab[tab.model == "mf"].sort_values("val_rmse").iloc[0]
    srows = []
    for reg in rc["mf_reg_grid"]:
        for beta in rc["social_beta_grid"]:
            m = rec.fit_mf(R, int(best_mf.factors), reg, rc["mf_reg_bias"], rc["mf_iters"], 0, social=ctx["T"], beta=beta)
            srows.append({"model": "social_mf", "factors": int(best_mf.factors), "reg": reg, "beta": beta,
                          "val_rmse": met.rmse(rec.predict_pairs(m, uv, iv), yv)})
    stab = pd.DataFrame(srows)
    best_soc = stab[stab.beta > 0].sort_values("val_rmse").iloc[0]
    tab = pd.concat([tab, stab], ignore_index=True)
    utils.save_table(tab, cfg["profile"], "tuning_validation")
    hp = {"knn_k": int(best_knn.k), "knn_damping": float(best_knn.damping), "knn_shrink": rc["knn_shrink"],
          "mf_factors": int(best_mf.factors), "mf_reg": float(best_mf.reg), "mf_iters": rc["mf_iters"],
          "reg_bias": rc["mf_reg_bias"], "social_reg": float(best_soc.reg), "social_beta": float(best_soc.beta)}
    utils.save_json(hp, cfg["profile"], "tuned_hyperparameters")
    return hp


def build_context(profile):
    cfg = utils.load_config(profile)
    data = dp.load_dataset(profile)
    ctx = {"cfg": cfg, "data": data, "profile": profile,
           "T": rec.trust_matrix(data["n_users"], data["truster"], data["trustee"])}
    utils.log("tuning hyper-parameters on the validation split")
    ctx["hp"] = exp_tuning(ctx)
    utils.log(f"tuned: {ctx['hp']}")
    # ---- simulator world (requirements 4-8) -----------------------------------
    s0 = dp.make_split(data, cfg["split"]["test_frac"], cfg["split"]["split_seeds"][0])
    ctx["split0"] = s0
    ctx["R_train"] = split_matrices(data, s0["train"])
    ctx["R_full"] = split_matrices(data, np.arange(len(data["r"])))
    ctx["graph"] = dif.influence_graph(data["n_users"], data["truster"], data["trustee"], ctx["R_train"],
                                       cfg["diffusion"]["agreement_tolerance"])
    ctx["oracle"] = simulated_true_preferences(ctx)
    ctx["models0"] = fit_models(ctx["R_train"], ctx["T"], ctx["hp"], 0)
    return ctx


def simulated_true_preferences(ctx):
    """'True' preference of every user for every item used by the simulator.

    observed rating where one exists (ALL observed ratings), otherwise the
    average of a kNN and an MF model fitted on all observed ratings. This
    matrix is INFERRED information - it is the simulator's world model, never
    used to train or to evaluate the recommenders on observed data.
    """
    hp = ctx["hp"]
    R = ctx["R_full"]
    knn = rec.fit_user_knn(R, k=hp["knn_k"], shrink=hp["knn_shrink"], damping=hp["knn_damping"])
    mf = rec.fit_mf(R, hp["mf_factors"], hp["mf_reg"], hp["reg_bias"], hp["mf_iters"], 7)
    users = np.arange(R.shape[0])
    pref = 0.5 * (rec.predict_rows(knn, users) + rec.predict_rows(mf, users))
    coo = R.tocoo()
    pref[coo.row, coo.col] = coo.data
    return pref.astype(np.float32)


# =========================================================================== requirement 1
def exp_dataset(ctx):
    cfg, data, plt = ctx["cfg"], ctx["data"], utils.setup_matplotlib()
    meta = data["meta"]
    R, n, g = ctx["R_train"], data["n_users"], ctx["graph"]
    utils.save_table(pd.DataFrame([{"quantity": k, "value": v} for k, v in meta.items()
                                   if not isinstance(v, dict)]), cfg["profile"], "dataset_statistics")
    # observed social correlation: is a user more likely to rate (in test) an
    # item that one of the people they trust rated (in train)?
    Tb = rec.trust_matrix(n, data["truster"], data["trustee"], row_normalise=False)
    Mtr = R.copy()
    Mtr.data[:] = 1
    friend_rated = (Tb @ Mtr).toarray() > 0
    te = ctx["split0"]["test"]
    test_mask = np.zeros_like(friend_rated)
    test_mask[data["u"][te], data["i"][te]] = True
    unseen = Mtr.toarray() == 0
    p_f = test_mask[friend_rated & unseen].mean()
    p_nf = test_mask[~friend_rated & unseen].mean()
    # agreement of trusting pairs vs random pairs on co-rated items (observed)
    rng = np.random.default_rng(cfg["seed"])
    rs, rd = rng.integers(0, n, len(g["src"])), rng.integers(0, n, len(g["src"]))
    s_rand, c_rand = dif.edge_agreement(rs, rd, R)
    s_trust, c_trust = dif.edge_agreement(g["src"], g["dst"], R)
    raw_agree = lambda s, c: ((s * (c + 2) - 1) / np.maximum(c, 1))[c >= 3].mean()
    social = {"p_test_given_friend_rated": p_f, "p_test_given_no_friend_rated": p_nf, "lift": p_f / p_nf,
              "agreement_trust_pairs": raw_agree(s_trust, c_trust), "agreement_random_pairs": raw_agree(s_rand, c_rand),
              "frac_trust_pairs_with_3plus_corated": float(np.mean(c_trust >= 3)),
              "frac_random_pairs_with_3plus_corated": float(np.mean(c_rand >= 3))}
    utils.save_table(pd.DataFrame([social]), cfg["profile"], "observed_social_correlation")
    per_user = np.bincount(data["u"], minlength=n)
    per_item = np.bincount(data["i"], minlength=data["n_items"])
    fig, ax = plt.subplots(1, 4, figsize=(12, 2.8))
    ax[0].hist(per_user, bins=np.logspace(0, np.log10(per_user.max() + 1), 30), color=PALETTE[0])
    ax[0].set(xscale="log", yscale="log", title="Ratings per user", xlabel="ratings", ylabel="users")
    ax[1].hist(per_item, bins=np.logspace(np.log10(5), np.log10(per_item.max() + 1), 30), color=PALETTE[0])
    ax[1].set(xscale="log", yscale="log", title="Ratings per item", xlabel="ratings", ylabel="items")
    vals = np.bincount(data["r"].astype(int), minlength=6)[1:]
    ax[2].bar(np.arange(1, 6), vals, color=PALETTE[0], width=0.6)
    ax[2].set(title="Rating values", xlabel="stars", ylabel="count")
    ax[3].hist(g["out_deg"] + 1, bins=np.logspace(0, np.log10(g["out_deg"].max() + 2), 30), color=PALETTE[0],
               alpha=0.9, label="followers (trusted by)")
    ax[3].hist(g["in_deg"] + 1, bins=np.logspace(0, np.log10(g["in_deg"].max() + 2), 30), color=PALETTE[1],
               alpha=0.6, label="followees (trusts)")
    ax[3].set(xscale="log", yscale="log", title="Trust degree (+1)", xlabel="degree", ylabel="users")
    ax[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(utils.figure_path(cfg["profile"], "fig1_dataset"))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 2.6))
    ax.hist(g["s"], bins=30, color=PALETTE[0])
    ax.set(title="Inferred edge strength s (agreement)", xlabel="s", ylabel="edges")
    fig.tight_layout()
    fig.savefig(utils.figure_path(cfg["profile"], "fig1b_edge_strength"))
    plt.close(fig)
    return {"social_lift": social["lift"], "agreement_trust": social["agreement_trust_pairs"],
            "agreement_random": social["agreement_random_pairs"]}


# =========================================================================== requirements 2 + 7 (recommendation)
def _user_features(data, R, T):
    n = data["n_users"]
    cnt = np.diff(R.indptr)
    s1 = np.asarray(R.sum(axis=1)).ravel()
    s2 = np.asarray(R.multiply(R).sum(axis=1)).ravel()
    mean = np.where(cnt > 0, s1 / np.maximum(cnt, 1), np.nan)
    std = np.sqrt(np.maximum(np.where(cnt > 0, s2 / np.maximum(cnt, 1), np.nan) - mean ** 2, 0))
    friends = np.diff(T.indptr)
    return {"n_train": cnt, "mean": mean, "std": std, "friends": friends}


def exp_recommendation(ctx):
    cfg, data, hp = ctx["cfg"], ctx["data"], ctx["hp"]
    K, thr = cfg["rec"]["top_k"], cfg["rec"]["relevance_threshold"]
    names = ("popularity", "bias", "knn", "mf", "social_mf", "knn_usermean", "knn_raw")
    rows, pair_rows, user_rows = [], [], []
    for seed in cfg["split"]["split_seeds"]:
        s = dp.make_split(data, cfg["split"]["test_frac"], seed)
        R = split_matrices(data, s["train"])
        te = s["test"]
        ut, it, yt = data["u"][te], data["i"][te], data["r"][te]
        users = np.unique(ut[yt >= thr])
        relevant = met.relevant_sets(users, ut, it, yt, thr)
        feats = _user_features(data, R, ctx["T"])
        item_pop = np.diff(R.tocsc().indptr)
        models = fit_models(R, ctx["T"], hp, seed, names)
        for name, (m, secs) in models.items():
            row = {"seed": seed, "model": name, "fit_seconds": secs}
            if name != "popularity":
                pred = rec.predict_pairs(m, ut, it)
                row.update({"rmse": met.rmse(pred, yt), "mae": met.mae(pred, yt)})
                row.update(met.classification(pred, yt, thr, cfg["rec"]["like_threshold_pred"]))
                pair_rows.append(pd.DataFrame({"seed": seed, "model": name, "u": ut, "i": it, "y": yt, "pred": pred,
                                               "n_train": feats["n_train"][ut], "user_mean": feats["mean"][ut],
                                               "user_std": feats["std"][ut], "item_pop": item_pop[it],
                                               "friends": feats["friends"][ut]}))
            lists = rec.recommend(m, R, users, K)
            row.update(met.ranking(lists, relevant, K))
            row.update(met.beyond_accuracy(lists, R, data["n_users"]))
            per = met.ranking_per_user(lists, relevant, K)
            user_rows.append(pd.DataFrame({"seed": seed, "model": name, "u": users, "ndcg": per["ndcg"],
                                           "recall": per["recall"], "n_train": feats["n_train"][users]}))
            rows.append(row)
        utils.log(f"recommendation split seed {seed} done")
    per_seed = pd.DataFrame(rows)
    utils.save_table(per_seed, cfg["profile"], "rec_per_seed")
    value_cols = [c for c in per_seed.columns if c not in ("seed", "model")]
    summary = utils.summarise_runs(per_seed, ["model"], value_cols)
    summary["model"] = pd.Categorical(summary.model, MODEL_ORDER)
    summary = summary.sort_values("model")
    utils.save_table(summary, cfg["profile"], "rec_summary")
    pairs = pd.concat(pair_rows, ignore_index=True)
    pairs["err2"] = (pairs.pred - pairs.y) ** 2
    ctx["pairs"] = pairs
    peruser = pd.concat(user_rows, ignore_index=True)
    out = _bucket_analysis(ctx, pairs, peruser)
    out.update(_cold_start_simulation(ctx))
    out["summary"] = summary.set_index("model")[["rmse_mean", "precision_mean", "ndcg_mean", "coverage_mean"]].to_dict()
    return out


def _bucket_rmse(pairs, col, bins, labels):
    p = pairs.copy()
    p["bucket"] = pd.cut(p[col], bins=bins, labels=labels, right=True)
    t = p.groupby(["model", "bucket"], observed=True).agg(rmse=("err2", lambda e: np.sqrt(e.mean())),
                                                          n=("err2", "size")).reset_index()
    t["factor"] = col
    return t


def _bucket_analysis(ctx, pairs, peruser):
    cfg, plt = ctx["cfg"], utils.setup_matplotlib()
    specs = [("n_train", [0, 3, 8, 20, 50, 10 ** 6], ["1-3", "4-8", "9-20", "21-50", ">50"], "user history (train ratings)"),
             ("item_pop", [-1, 5, 10, 20, 50, 10 ** 6], ["0-5", "6-10", "11-20", "21-50", ">50"], "item popularity (train ratings)"),
             ("user_mean", [0, 3.0, 3.5, 4.0, 4.5, 5.01], ["<=3.0", "3-3.5", "3.5-4", "4-4.5", ">4.5"], "user mean rating"),
             ("user_std", [-0.01, 0.3, 0.7, 1.0, 1.3, 5], ["<=0.3", "0.3-0.7", "0.7-1.0", "1.0-1.3", ">1.3"], "user rating std")]
    tabs = [_bucket_rmse(pairs, c, b, l) for c, b, l, _ in specs]
    tab = pd.concat(tabs, ignore_index=True)
    utils.save_table(tab, cfg["profile"], "rec_rmse_by_bucket")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3))
    show = ["bias", "knn", "mf", "social_mf", "knn_usermean", "knn_raw"]
    for ax, t, (c, _, labels, title) in zip(axes, tabs, specs):
        for j, mname in enumerate(show):
            d = t[t.model == mname].set_index("bucket").reindex(labels)
            ax.plot(labels, d.rmse.values, marker="o", ms=3, lw=1.5, color=PALETTE[j], label=mname)
        ax.set(title=title, ylabel="test RMSE")
        ax.tick_params(axis="x", labelsize=7)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(utils.figure_path(cfg["profile"], "fig2_rmse_buckets"))
    plt.close(fig)
    peruser["bucket"] = pd.cut(peruser.n_train, [0, 3, 8, 20, 50, 10 ** 6], labels=["1-3", "4-8", "9-20", "21-50", ">50"])
    rk = peruser.groupby(["model", "bucket"], observed=True)[["ndcg", "recall"]].mean().reset_index()
    utils.save_table(rk, cfg["profile"], "rec_ranking_by_history")
    return {"rmse_cold_users": tab[(tab.factor == "n_train") & (tab.bucket == "1-3")].set_index("model").rmse.to_dict()}


def _cold_start_simulation(ctx):
    """Artificially truncate the training history of users with long histories
    to m ratings and measure the error on their (unchanged) test ratings."""
    cfg, data, hp = ctx["cfg"], ctx["data"], ctx["hp"]
    rng = np.random.default_rng(cfg["seed"])
    s = ctx["split0"]
    tr = s["train"]
    cnt = np.bincount(data["u"][tr], minlength=data["n_users"])
    pool = np.where(cnt >= 20)[0]
    chosen = rng.choice(pool, min(cfg["rec"]["cold_users"], len(pool)), replace=False)
    is_chosen = np.zeros(data["n_users"], dtype=bool)
    is_chosen[chosen] = True
    te = s["test"][is_chosen[data["u"][s["test"]]]]
    rows = []
    for m in cfg["rec"]["cold_history_sizes"]:
        keep_idx = []
        for u in chosen:
            ui = tr[data["u"][tr] == u]
            keep_idx.append(rng.choice(ui, m, replace=False))
        tr_m = np.concatenate([tr[~is_chosen[data["u"][tr]]]] + keep_idx)
        R = split_matrices(data, tr_m)
        models = fit_models(R, ctx["T"], hp, 0, ("bias", "knn", "mf", "social_mf"))
        for name, (model, _) in models.items():
            pred = rec.predict_pairs(model, data["u"][te], data["i"][te])
            rows.append({"history_size": m, "model": name, "rmse": met.rmse(pred, data["r"][te])})
    tab = pd.DataFrame(rows)
    utils.save_table(tab, cfg["profile"], "rec_cold_start_simulation")
    plt = utils.setup_matplotlib()
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for j, name in enumerate(["bias", "knn", "mf", "social_mf"]):
        d = tab[tab.model == name]
        ax.plot(d.history_size, d.rmse, marker="o", ms=4, lw=1.5, color=PALETTE[j], label=name)
    ax.set(xscale="log", xlabel="training ratings kept per user", ylabel="test RMSE",
           title="Simulated cold start (truncated histories)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(utils.figure_path(cfg["profile"], "fig3_cold_start"))
    plt.close(fig)
    return {"cold_start": tab.pivot(index="history_size", columns="model", values="rmse").to_dict()}


# =========================================================================== requirement 3 (social)
def exp_social(ctx):
    cfg, data, hp = ctx["cfg"], ctx["data"], ctx["hp"]
    plt = utils.setup_matplotlib()
    rows, user_rows = [], []
    for seed in cfg["split"]["split_seeds"]:
        s = dp.make_split(data, cfg["split"]["test_frac"], seed)
        R = split_matrices(data, s["train"])
        te = s["test"]
        ut, it, yt = data["u"][te], data["i"][te], data["r"][te]
        feats = _user_features(data, R, ctx["T"])
        g = dif.influence_graph(data["n_users"], data["truster"], data["trustee"], R)
        # average agreement between a user and the people they trust (train ratings only)
        agree = np.bincount(g["dst"], weights=g["s"], minlength=data["n_users"]) / np.maximum(g["in_deg"], 1)
        cold = feats["n_train"][ut] <= 5
        for beta in cfg["rec"]["social_beta_grid"]:
            m = rec.fit_mf(R, hp["mf_factors"], hp["social_reg"], hp["reg_bias"], hp["mf_iters"], seed,
                           social=ctx["T"], beta=beta)
            pred = rec.predict_pairs(m, ut, it)
            e2 = (pred - yt) ** 2
            rows.append({"seed": seed, "beta": beta, "rmse": np.sqrt(e2.mean()), "rmse_cold_le5": np.sqrt(e2[cold].mean()),
                         "rmse_warm_gt5": np.sqrt(e2[~cold].mean())})
            user_rows.append(pd.DataFrame({"seed": seed, "beta": beta, "u": ut, "err2": e2,
                                           "n_train": feats["n_train"][ut], "friends": feats["friends"][ut],
                                           "agreement": agree[ut]}))
    tab = pd.DataFrame(rows)
    sweep = utils.summarise_runs(tab, ["beta"], ["rmse", "rmse_cold_le5", "rmse_warm_gt5"])
    utils.save_table(sweep, cfg["profile"], "social_beta_sweep")
    ur = pd.concat(user_rows, ignore_index=True)
    ur["agree_bucket"] = pd.qcut(ur.agreement.rank(method="first"), 3, labels=["low", "mid", "high"])
    ur["hist_bucket"] = pd.cut(ur.n_train, [0, 5, 20, 10 ** 6], labels=["<=5", "6-20", ">20"])
    by = ur.groupby(["beta", "agree_bucket", "hist_bucket"], observed=True).err2.mean().pipe(np.sqrt).reset_index(name="rmse")
    base = by[by.beta == 0].set_index(["agree_bucket", "hist_bucket"]).rmse
    by["delta_vs_mf"] = by.rmse - base.reindex(pd.MultiIndex.from_frame(by[["agree_bucket", "hist_bucket"]])).values
    utils.save_table(by, cfg["profile"], "social_effect_by_agreement_history")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    x = sweep.beta.replace(0, 0.3)
    for j, (col, lab) in enumerate([("rmse", "all users"), ("rmse_cold_le5", "<=5 train ratings"), ("rmse_warm_gt5", ">5 train ratings")]):
        ax[0].errorbar(x, sweep[col + "_mean"], yerr=sweep[col + "_std"], marker="o", ms=4, lw=1.5, capsize=2,
                       color=PALETTE[j], label=lab)
    ax[0].set(xscale="log", xlabel="social weight beta (0 plotted at 0.3)", ylabel="test RMSE", title="SocialMF: effect of beta")
    ax[0].legend()
    best_beta = hp["social_beta"]
    d = by[by.beta == best_beta]
    for j, hb in enumerate(["<=5", "6-20", ">20"]):
        dd = d[d.hist_bucket == hb].set_index("agree_bucket").reindex(["low", "mid", "high"])
        ax[1].plot(["low", "mid", "high"], dd.delta_vs_mf.values, marker="o", ms=4, lw=1.5, color=PALETTE[j], label=f"history {hb}")
    ax[1].axhline(0, color="#52514e", lw=0.8)
    ax[1].set(xlabel="agreement with trusted users (tercile)", ylabel="RMSE change vs MF (beta=0)",
              title=f"Where social info helps (beta={best_beta:g})")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(utils.figure_path(cfg["profile"], "fig4_social_beta"))
    plt.close(fig)
    out = {"beta_sweep": sweep.set_index("beta")[["rmse_mean", "rmse_cold_le5_mean"]].to_dict()}
    out.update(_friends_vs_similar(ctx))
    return out


def _friends_vs_similar(ctx):
    """Do trusted users carry more preference information than equally many
    users who are similar but socially unconnected (or random strangers)?"""
    cfg, data, hp = ctx["cfg"], ctx["data"], ctx["hp"]
    rng = np.random.default_rng(cfg["seed"] + 1)
    R, n = ctx["R_train"], data["n_users"]
    s = ctx["split0"]
    te = s["test"]
    Tb = rec.trust_matrix(n, data["truster"], data["trustee"], row_normalise=False)
    linked = ((Tb + Tb.T) > 0).tocsr()
    simm = rec.fit_user_knn(R, k=min(400, n - 1), shrink=hp["knn_shrink"])["S"].tocsr()
    pools = {"friends": ([], [], []), "similar_unconnected": ([], [], []), "random_unconnected": ([], [], [])}
    for u in range(n):
        fr = Tb.indices[Tb.indptr[u]:Tb.indptr[u + 1]]
        if fr.size == 0:
            continue
        m = min(fr.size, 20)
        link = set(linked.indices[linked.indptr[u]:linked.indptr[u + 1]]) | {u}
        cand = simm.indices[simm.indptr[u]:simm.indptr[u + 1]]
        cw = simm.data[simm.indptr[u]:simm.indptr[u + 1]]
        order = np.argsort(-cw)
        sim_pool = [c for c in cand[order] if c not in link][:m]
        rand_pool = []
        while len(rand_pool) < m:
            c = int(rng.integers(n))
            if c not in link and c not in rand_pool:
                rand_pool.append(c)
        for name, members in (("friends", rng.choice(fr, m, replace=False)), ("similar_unconnected", sim_pool),
                              ("random_unconnected", rand_pool)):
            pools[name][0].extend([u] * len(members))
            pools[name][1].extend(members)
            pools[name][2].append(len(members))
    ut, it, yt = data["u"][te], data["i"][te], data["r"][te]
    rows, preds = [], {}
    for name, (a, b, sizes) in pools.items():
        S = sp.csr_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
        model = rec.knn_from_neighbours(R, S, damping=hp["knn_damping"])
        preds[name] = rec.predict_pairs(model, ut, it)
        # residual agreement between u and pool members on co-rated training items
        A, B = np.array(a), np.array(b)
        corr, common = _residual_correlation(model["Rc"], model["M"], A, B)
        rows.append({"pool": name, "pairs": len(a), "mean_pool_size": np.mean(sizes),
                     "pairs_with_3plus_corated": float(np.mean(common >= 3)),
                     "residual_corr_mean": float(np.nanmean(np.where(common >= 3, corr, np.nan)))})
    # coverage: does at least one pool member have a training rating for the test item?
    for row in rows:
        a, b, _ = pools[row["pool"]]
        S = sp.csr_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
        Mtr = R.copy()
        Mtr.data[:] = 1
        sup = (S @ Mtr).tocsr()
        cover = np.asarray(sup[ut, it]).ravel() > 0
        row["test_coverage"] = float(cover.mean())
        row["cover_mask"] = cover
    all_cov = np.logical_and.reduce([r["cover_mask"] for r in rows])
    bias = ctx["models0"]["bias"][0]
    base = rec.predict_pairs(bias, ut, it)
    for row in rows:
        p = preds[row["pool"]]
        cov = row.pop("cover_mask")
        row["rmse_all_test"] = met.rmse(p, yt)
        row["rmse_where_all_pools_cover"] = met.rmse(p[all_cov], yt[all_cov])
        row["rmse_baseline_same_pairs"] = met.rmse(base[all_cov], yt[all_cov])
        row["rmse_where_pool_covers"] = met.rmse(p[cov], yt[cov]) if cov.any() else np.nan
    tab = pd.DataFrame(rows)
    utils.save_table(tab, ctx["cfg"]["profile"], "social_friends_vs_similar")
    return {"friends_vs_similar": tab.set_index("pool")[["residual_corr_mean", "rmse_where_all_pools_cover", "test_coverage"]].to_dict()}


def _residual_correlation(Rc, M, a, b, chunk=50000):
    """Pearson-style correlation of the two users' residuals over co-rated items."""
    corr, common = np.zeros(len(a)), np.zeros(len(a))
    for s in range(0, len(a), chunk):
        x, y = a[s:s + chunk], b[s:s + chunk]
        Ra, Rb, Ma, Mb = Rc[x], Rc[y], M[x], M[y]
        num = np.asarray(Ra.multiply(Rb).sum(axis=1)).ravel()
        na = np.asarray(Ra.multiply(Ra).multiply(Mb).sum(axis=1)).ravel()
        nb = np.asarray(Rb.multiply(Rb).multiply(Ma).sum(axis=1)).ravel()
        corr[s:s + chunk] = num / np.sqrt(na * nb + 1e-12)
        common[s:s + chunk] = np.asarray(Ma.multiply(Mb).sum(axis=1)).ravel()
    return corr, common


# =========================================================================== requirement 4 (diffusion)
def campaign_items(ctx):
    """Three items with high / medium / low average appeal among items with
    at least 20 training ratings (so their preferences are well estimated)."""
    pop = np.diff(ctx["R_train"].tocsc().indptr)
    cfg = ctx["cfg"]
    a = dif.adoption_prob(ctx["oracle"], cfg["diffusion"]["kappa"], cfg["diffusion"]["theta"]).mean(axis=0)
    cand = np.where(pop >= 20)[0]
    order = cand[np.argsort(a[cand])]
    pick = {"high": order[int(0.9 * (len(order) - 1))], "medium": order[int(0.5 * (len(order) - 1))],
            "low": order[int(0.1 * (len(order) - 1))]}
    return pick, a


def exp_diffusion(ctx):
    cfg, data, g = ctx["cfg"], ctx["data"], ctx["graph"]
    dc = cfg["diffusion"]
    plt = utils.setup_matplotlib()
    rng = np.random.default_rng(cfg["seed"] + 4)
    items, mean_appeal = campaign_items(ctx)
    seeds = rng.choice(data["n_users"], dc["n_seeds_spread"], replace=False)
    ns = dc["n_sims"]
    rows = []

    def record(**kw):
        cond = {k: v for k, v in kw.items() if k not in ("graph", "adopt", "gamma", "attn")}
        res = dif.run_many(kw["graph"], kw["attn"], kw["adopt"], seeds, ns, rng, gamma=kw["gamma"])
        m, sd, ci = utils.mean_ci(res["adopters"])
        rows.append(dict(cond, adopters_mean=m, adopters_std=sd, adopters_ci95=ci,
                         exposed_mean=res["exposed_social"].mean(), seed_adopters_mean=res["seed_adopters"].mean(),
                         t90_mean=res["t90"].mean(), t90_std=res["t90"].std(), steps_mean=res["steps"].mean()))
        return res

    # (a) influence strength x item appeal x model
    curves = {}
    for level, item in items.items():
        a = dif.adoption_prob(ctx["oracle"][:, item], dc["kappa"], dc["theta"])
        for lam in dc["lambda_grid"]:
            for model, gamma in (("IC", 1.0), ("SIR", dc["sir_gamma"])):
                res = record(study="lambda", item_appeal=level, item=item, lam=lam, model=model, graph=g,
                             attn=dif.attention_prob(g, lam), adopt=a, gamma=gamma)
                if level == "medium" and lam == dc["lambda"]:
                    curves[model] = res["curves"]
    # (b) SIR infectious period
    item = items["medium"]
    a = dif.adoption_prob(ctx["oracle"][:, item], dc["kappa"], dc["theta"])
    for gamma in dc["gamma_grid"]:
        for lam in dc["lambda_grid"]:
            record(study="gamma", item_appeal="medium", item=item, lam=lam, model=f"SIR g={gamma}", graph=g,
                   attn=dif.attention_prob(g, lam), adopt=a, gamma=gamma)
    # (c) preferences: heterogeneous vs preference-blind adoption, and kappa
    for kappa in (0.0, 1.0, 2.0, 4.0):
        ak = dif.adoption_prob(ctx["oracle"][:, item], kappa, dc["theta"])
        record(study="kappa", item_appeal="medium", item=item, lam=dc["lambda"], model="IC", kappa=kappa, graph=g,
               attn=dif.attention_prob(g, dc["lambda"]), adopt=ak, gamma=1.0)
    flat = np.full(data["n_users"], a.mean())
    record(study="pref_blind", item_appeal="medium", item=item, lam=dc["lambda"], model="IC", variant="personal",
           graph=g, attn=dif.attention_prob(g, dc["lambda"]), adopt=a, gamma=1.0)
    record(study="pref_blind", item_appeal="medium", item=item, lam=dc["lambda"], model="IC", variant="same mean, no personal",
           graph=g, attn=dif.attention_prob(g, dc["lambda"]), adopt=flat, gamma=1.0)
    # (d) network structure
    graphs = {"real": g, "degree_preserving": dif.rewire_degree_preserving(g, rng), "erdos_renyi": dif.random_graph(g, rng)}
    for gname, gg in graphs.items():
        for lam in dc["lambda_grid"]:
            for model, gamma in (("IC", 1.0), ("SIR", dc["sir_gamma"])):
                record(study="structure", item_appeal="medium", item=item, lam=lam, model=model, network=gname,
                       graph=gg, attn=dif.attention_prob(gg, lam), adopt=a, gamma=gamma)
    tab = pd.DataFrame(rows)
    utils.save_table(tab, cfg["profile"], "diffusion_experiments")
    struct = pd.DataFrame([{"network": k, "edges": len(v["src"]), "reciprocity": _reciprocity(v),
                            "clustering": _clustering(v, rng)} for k, v in graphs.items()])
    utils.save_table(struct, cfg["profile"], "diffusion_network_structure")
    # figures
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
    lam_t = tab[tab.study == "lambda"]
    j = 0
    for level in ("high", "medium", "low"):
        for model, ls in (("IC", "-"), ("SIR", "--")):
            d = lam_t[(lam_t.item_appeal == level) & (lam_t.model == model)]
            ax[0].errorbar(d.lam, d.adopters_mean, yerr=d.adopters_std, ls=ls, marker="o", ms=3, lw=1.5, capsize=2,
                           color=PALETTE[["high", "medium", "low"].index(level)], label=f"{level} appeal, {model}")
            j += 1
    ax[0].set(xscale="log", yscale="log", xlabel="influence strength lambda", ylabel="adopters (mean +- sd)",
              title="Spread vs influence strength (20 random seeds)")
    ax[0].legend(fontsize=6)
    st = tab[tab.study == "structure"]
    for k, gname in enumerate(graphs):
        for model, ls in (("IC", "-"), ("SIR", "--")):
            d = st[(st.network == gname) & (st.model == model)]
            ax[1].plot(d.lam, d.adopters_mean, ls=ls, marker="o", ms=3, lw=1.5, color=PALETTE[k], label=f"{gname}, {model}")
    ax[1].set(xscale="log", yscale="log", xlabel="influence strength lambda", ylabel="adopters (mean)",
              title="Spread vs network structure (medium item)")
    ax[1].legend(fontsize=6)
    for k, (model, cv) in enumerate(curves.items()):
        mean = cv.mean(axis=0)
        ax[2].plot(np.arange(len(mean)), mean, lw=2, color=PALETTE[k], label=model)
        ax[2].fill_between(np.arange(len(mean)), np.percentile(cv, 10, axis=0), np.percentile(cv, 90, axis=0),
                           color=PALETTE[k], alpha=0.15, lw=0)
    ax[2].set(yscale="log", xlabel="time step", ylabel="cumulative adopters (log)",
              title=f"Adoption curves (lambda={dc['lambda']:g}, 10-90% band)")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(utils.figure_path(cfg["profile"], "fig5_diffusion"))
    plt.close(fig)
    ctx["campaign_items"] = items
    return {"items": items, "mean_appeal": {k: float(mean_appeal[v]) for k, v in items.items()},
            "lambda_table": lam_t[["item_appeal", "lam", "model", "adopters_mean", "t90_mean"]].to_dict("records")}


def _reciprocity(g):
    key = set(zip(g["src"].tolist(), g["dst"].tolist()))
    return float(np.mean([(b, a) in key for a, b in key]))


def _clustering(g, rng, samples=500):
    """Average local clustering of the undirected graph (sampled users)."""
    A = sp.csr_matrix((np.ones(len(g["src"])), (g["src"], g["dst"])), shape=(g["n"], g["n"]))
    A = ((A + A.T) > 0).astype(float).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    users = rng.choice(g["n"], min(samples, g["n"]), replace=False)
    vals = []
    for u in users:
        nb = A.indices[A.indptr[u]:A.indptr[u + 1]]
        if len(nb) < 2:
            continue
        links = A[nb][:, nb].sum() / 2
        vals.append(links / (len(nb) * (len(nb) - 1) / 2))
    return float(np.mean(vals))


# =========================================================================== requirement 5 (seeding)
def exp_influence_max(ctx):
    cfg, data, g = ctx["cfg"], ctx["data"], ctx["graph"]
    dc, ic = cfg["diffusion"], cfg["im"]
    plt = utils.setup_matplotlib()
    rng = np.random.default_rng(cfg["seed"] + 5)
    items = ctx.get("campaign_items") or campaign_items(ctx)[0]
    item = items["medium"]
    a = dif.adoption_prob(ctx["oracle"][:, item], dc["kappa"], dc["theta"])
    rows, step_rows, unc_rows = [], [], []
    kmax = max(ic["budgets"])
    n = g["n"]
    for lam in (dc["lambda"], 2 * dc["lambda"]):
        attn = dif.attention_prob(g, lam)
        worlds = im.sample_worlds(g, attn, a, ic["n_worlds"], rng)
        t0 = time.perf_counter()
        G0 = im.initial_gains(worlds, n)
        t_init = time.perf_counter() - t0
        sel = {}
        t0 = time.perf_counter()
        sel["greedy_celf"], steps, evals = im.celf(worlds, n, kmax, G0=G0)
        t_celf = time.perf_counter() - t0 + t_init
        sel["greedy_celf_lcb"], _, _ = im.celf(worlds, n, kmax, mode="lcb", z=ic["z"], G0=G0)
        blind_w = im.sample_worlds(g, attn, np.full(n, a.mean()), ic["n_worlds"], rng)
        sel["greedy_pref_blind"], _, _ = im.celf(blind_w, n, kmax)
        sel["top_individual"] = im.top_individual_seeds(G0, kmax)
        sel["high_degree"] = im.degree_seeds(g, kmax)
        for st in steps:
            step_rows.append(dict(st, lam=lam))
        eval_rng = np.random.default_rng(cfg["seed"] + 55)          # fresh randomness, not the selection worlds
        for k in ic["budgets"]:
            for method, seeds in sel.items():
                res = dif.run_many(g, attn, a, seeds[:k], ic["eval_sims"], eval_rng)
                m, sd, ci = utils.mean_ci(res["adopters"])
                est = im.world_spread(worlds, seeds[:k], n).mean()
                rows.append({"lam": lam, "budget": k, "method": method, "spread_mean": m, "spread_std": sd,
                             "spread_ci95": ci, "t90_mean": res["t90"].mean(), "selection_estimate": est,
                             "overlap_jaccard": im.audience_overlap(g, seeds[:k]),
                             "redundancy": 1 - est / max(G0[:, seeds[:k]].mean(axis=0).sum(), 1e-9)})
            rand_spreads, rand_overlap = [], []
            for _ in range(ic["random_repeats"]):
                seeds = im.random_seeds(n, k, rng)
                rand_spreads.append(dif.run_many(g, attn, a, seeds, ic["eval_sims"] // ic["random_repeats"] + 1,
                                                 eval_rng)["adopters"].mean())
                rand_overlap.append(im.audience_overlap(g, seeds))
            m, sd, ci = utils.mean_ci(rand_spreads)
            rows.append({"lam": lam, "budget": k, "method": "random", "spread_mean": m, "spread_std": sd,
                         "spread_ci95": ci, "overlap_jaccard": np.mean(rand_overlap)})
        rows.append({"lam": lam, "budget": kmax, "method": "_celf_runtime", "spread_mean": t_celf, "spread_std": evals})
        # uncertainty: how many worlds are needed for a reliable estimate of each candidate's contribution?
        if lam == dc["lambda"]:
            k_unc = min(10, kmax)
            for R in ic["worlds_grid"]:
                for rep in range(3):
                    w = im.sample_worlds(g, attn, a, R, rng)
                    seeds, st, _ = im.celf(w, n, k_unc)
                    res = dif.run_many(g, attn, a, seeds, ic["eval_sims"], eval_rng)
                    unc_rows.append({"n_worlds": R, "rep": rep, "budget": k_unc, "estimated_spread": st[-1]["spread_est"],
                                     "evaluated_spread": res["adopters"].mean(),
                                     "first_gain_ci95": st[0]["gain_ci95"], "first_gain_mean": st[0]["gain_mean"]})
        utils.log(f"influence maximisation lambda={lam} done")
    tab = pd.DataFrame(rows)
    rt = tab[tab.method == "_celf_runtime"]
    tab = tab[tab.method != "_celf_runtime"]
    utils.save_table(tab, cfg["profile"], "im_budget_comparison")
    utils.save_table(pd.DataFrame(step_rows), cfg["profile"], "im_celf_steps")
    unc = pd.DataFrame(unc_rows)
    utils.save_table(unc, cfg["profile"], "im_uncertainty_worlds")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
    methods = ["greedy_celf", "greedy_celf_lcb", "greedy_pref_blind", "top_individual", "high_degree", "random"]
    for j, lam in enumerate(sorted(tab.lam.unique())):
        a_ = ax[j]
        for k, mth in enumerate(methods):
            d = tab[(tab.lam == lam) & (tab.method == mth)]
            a_.errorbar(d.budget, d.spread_mean, yerr=d.spread_ci95, marker="o", ms=3, lw=1.5, capsize=2,
                        color=PALETTE[k], label=mth)
        a_.set(xlabel="budget k (promoted users)", ylabel="adopters (mean, 95% CI)", title=f"Seeding strategies, lambda={lam:g}")
        a_.legend(fontsize=6)
    st = pd.DataFrame(step_rows)
    d = st[st.lam == dc["lambda"]]
    ax[2].errorbar(d.step, d.gain_mean, yerr=d.gain_ci95, marker="o", ms=3, lw=1.2, capsize=2, color=PALETTE[0])
    ax[2].set(yscale="log", xlabel="greedy step", ylabel="marginal gain (95% CI)", title="Diminishing marginal gains (CELF)")
    fig.tight_layout()
    fig.savefig(utils.figure_path(cfg["profile"], "fig6_influence_max"))
    plt.close(fig)
    return {"celf_seconds": rt.spread_mean.tolist(), "celf_evals": rt.spread_std.tolist(),
            "uncertainty": unc.groupby("n_worlds")[["estimated_spread", "evaluated_spread"]].agg(["mean", "std"]).round(2).to_dict()}


# =========================================================================== requirement 6 (groups)
def exp_groups(ctx):
    cfg, data, g = ctx["cfg"], ctx["data"], ctx["graph"]
    gc, dc = cfg["group"], cfg["diffusion"]
    plt = utils.setup_matplotlib()
    rng = np.random.default_rng(cfg["seed"] + 6)
    model = ctx["models0"]["social_mf"][0]
    R = ctx["R_train"]
    n = data["n_users"]
    A = sp.csr_matrix((np.ones(len(g["src"])), (g["src"], g["dst"])), shape=(n, n))
    A = ((A + A.T) > 0).tocsr()
    groups = grp.social_groups((A.indptr, A.indices), gc["candidate_groups"], gc["group_size"], rng)
    pop = np.diff(R.tocsc().indptr)
    knn = ctx["models0"]["knn"][0]
    scored = []
    for members in groups:
        # nominate-then-decide: every member nominates their personal top-5
        # (kNN top-N model) among eligible items nobody in the group has rated;
        # the group then picks one nomination using SocialMF predicted ratings.
        seen = np.unique(R[members].indices)
        votes = rec.rank_rows(knn, members)
        votes[:, seen] = -np.inf
        votes[:, pop < 10] = -np.inf
        cand = np.unique(np.argsort(-votes, axis=1)[:, :5])
        P = rec.predict_rows(model, members)
        scored.append((grp.conflict(P[:, cand]), members, cand, P))
    scored.sort(key=lambda x: -x[0])
    chosen = scored[:gc["n_groups"]]
    attn = dif.attention_prob(g, dc["lambda"])
    rows, cand_rows = [], []
    for gid, (conf, members, cand, P) in enumerate(chosen):
        tops = {s: grp.recommend_for_group(P, cand, s, gc["top_per_strategy"]) for s in grp.STRATEGIES}
        pool = np.unique(np.concatenate([t[0] for t in tops.values()]))
        spread = {}
        for item in pool:
            a = dif.adoption_prob(ctx["oracle"][:, item], dc["kappa"], dc["theta"])
            res = dif.run_many(g, attn, a, members, gc["n_sims"], rng)
            spread[item] = (res["adopters"].mean(), res["adopters"].std(), res["seed_adopters"].mean())
        best_item = max(spread, key=lambda x: spread[x][0])
        for item in pool:
            cand_rows.append({"group": gid, "item": item, "spread": spread[item][0],
                              **{f"score_{s}": float(f(P[:, [item]])[0]) for s, f in grp.STRATEGIES.items()}})
        for s, (items, scores) in tops.items():
            top = items[0]
            sat = grp.satisfaction(ctx["oracle"][members].astype(float), top)
            rows.append({"group": gid, "conflict": conf, "strategy": s, "item": top, **sat,
                         "spread_mean": spread[top][0], "spread_std": spread[top][1], "member_adopters": spread[top][2],
                         "is_max_spread_item": top == best_item, "spread_ratio_to_best": spread[top][0] / spread[best_item][0]})
    tab = pd.DataFrame(rows)
    utils.save_table(tab, cfg["profile"], "group_per_group")
    cands = pd.DataFrame(cand_rows)
    utils.save_table(cands, cfg["profile"], "group_candidate_spread")
    summ = tab.groupby("strategy")[["sat_mean", "sat_min", "sat_max", "sat_std", "frac_satisfied", "frac_miserable",
                                     "spread_mean", "member_adopters", "is_max_spread_item", "spread_ratio_to_best"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    corr = []
    for s in grp.STRATEGIES:
        rho = cands.groupby("group").apply(lambda d: d[f"score_{s}"].rank().corr(d.spread.rank()) if len(d) > 2 else np.nan,
                                           include_groups=False)
        corr.append({"strategy": s, "spearman_score_vs_spread_mean": rho.mean(), "spearman_std": rho.std()})
    summ = summ.merge(pd.DataFrame(corr), on="strategy")
    utils.save_table(summ, cfg["profile"], "group_summary")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    strat = list(grp.STRATEGIES)
    x = np.arange(3)
    for j, (col, lab) in enumerate([("sat_mean", "mean member pref"), ("sat_min", "least happy member"), ("sat_max", "happiest member")]):
        vals = summ.set_index("strategy").reindex(strat)[col + "_mean"]
        ax[0].bar(x + (j - 1) * 0.26, vals, 0.24, color=PALETTE[j], label=lab)
    ax[0].set(xticks=x, xticklabels=strat, ylabel="true preference (1-5)", ylim=(1, 5), title="Group satisfaction by strategy")
    ax[0].legend(fontsize=7)
    vals = summ.set_index("strategy").reindex(strat)
    ax[1].bar(x, vals.spread_mean_mean, 0.5, yerr=vals.spread_mean_std, color=PALETTE[0], capsize=3)
    ax[1].set(xticks=x, xticklabels=strat, ylabel="adopters when seeded via group", title="Spread of the chosen item")
    fig.tight_layout()
    fig.savefig(utils.figure_path(cfg["profile"], "fig7_groups"))
    plt.close(fig)
    return {"group_summary": summ.set_index("strategy")[["sat_mean_mean", "sat_min_mean", "spread_mean_mean",
                                                          "is_max_spread_item_mean", "spearman_score_vs_spread_mean"]].to_dict()}


# =========================================================================== requirement 7 (interaction)
def _campaign(ctx, rec_items, targets, n_sims, rng, lam=None, item_receive_mask=None, **sim_kw):
    """Each target user receives their recommended item; every distinct item
    then diffuses (independently) from the targets who received it. Returns
    per-run totals of adoption and satisfaction (true preference of adopters)."""
    cfg, g = ctx["cfg"], ctx["graph"]
    dc = cfg["diffusion"]
    attn = dif.attention_prob(g, dc["lambda"] if lam is None else lam)
    oracle = ctx["oracle"]
    totals = np.zeros((n_sims, 7))
    for item in np.unique(rec_items):
        seeds = targets[rec_items == item]
        a = dif.adoption_prob(oracle[:, item], dc["kappa"], dc["theta"])
        pref = oracle[:, item]
        if item_receive_mask is not None:
            sim_kw["receive_mask"] = item_receive_mask(item)
        for r in range(n_sims):
            out = dif.simulate(g, attn, a, seeds, rng, return_nodes=True, **sim_kw)
            ad, ex = out["adopted_mask"], out["exposed_mask"]
            totals[r] += [ad.sum(), out["seed_adopters"], pref[ad].sum(), (pref[ad] >= 4).sum(), ex.sum(),
                          (ex & (pref < 3)).sum(), (ex & ~ad).sum()]
    adopters = totals[:, 0]
    return {"adopters": adopters, "seed_adopters": totals[:, 1], "secondary_adopters": adopters - totals[:, 1],
            "adopter_pref": totals[:, 2] / np.maximum(adopters, 1), "adopters_satisfied_frac": totals[:, 3] / np.maximum(adopters, 1),
            "exposed": totals[:, 4], "unwanted_exposures": totals[:, 5], "exposed_not_adopted": totals[:, 6]}


def _hybrid_scores(ctx, model, targets, eta):
    """Spread-aware hybrid: (1-eta) * personal score + eta * social appeal, where
    social appeal of item i for target u = sum over u's followers v of
    w_uv * sigmoid(predicted pref of v for i). Uses only training data."""
    g, dc = ctx["graph"], ctx["cfg"]["diffusion"]
    n = g["n"]
    W = sp.csr_matrix((g["w"], (g["src"], g["dst"])), shape=(n, n))
    pred_all = rec.predict_rows(model, np.arange(n))
    appeal = W[targets] @ dif.adoption_prob(pred_all, dc["kappa"], dc["theta"])
    personal = (pred_all[targets] - 1) / 4
    norm = lambda x: (x - x.min(axis=1, keepdims=True)) / (np.ptp(x, axis=1, keepdims=True) + 1e-12)
    return (1 - eta) * norm(personal) + eta * norm(appeal)


def exp_interaction(ctx):
    cfg, data = ctx["cfg"], ctx["data"]
    icf = cfg["interaction"]
    plt = utils.setup_matplotlib()
    rng = np.random.default_rng(cfg["seed"] + 7)
    R = ctx["R_train"]
    te = ctx["split0"]["test"]
    test_like = set(zip(data["u"][te][data["r"][te] >= 4], data["i"][te][data["r"][te] >= 4]))
    # candidate-item selection: only items with >= 10 training ratings may be
    # promoted (the platform does not push items it knows almost nothing about)
    ineligible = np.diff(R.tocsc().indptr) < 10
    rows = []
    for rep in range(icf["target_repeats"]):
        targets = np.sort(rng.choice(data["n_users"], icf["n_targets"], replace=False))
        variants = {name: rec.rank_rows(ctx["models0"][name][0], targets)
                    for name in ("popularity", "knn", "mf", "social_mf")}
        for eta in icf["eta_grid"]:
            variants[f"hybrid_eta={eta:g}"] = _hybrid_scores(ctx, ctx["models0"]["social_mf"][0], targets, eta)
        for name, scores in variants.items():
            scores[:, ineligible] = -np.inf
            top = rec.top_n(scores, R, targets, 1)[:, 0]
            out = _campaign(ctx, top, targets, icf["n_sims"], rng)
            pref_t = ctx["oracle"][targets, top]
            row = {"rep": rep, "method": name, "distinct_items": len(np.unique(top)),
                   "target_true_pref": float(pref_t.mean()), "target_pref_ge4": float(np.mean(pref_t >= 4)),
                   "observed_hit_rate": float(np.mean([(u, i) in test_like for u, i in zip(targets, top)]))}
            row.update({k: float(v.mean()) for k, v in out.items()})
            row["adopters_std"] = float(out["adopters"].std())
            rows.append(row)
        utils.log(f"interaction repeat {rep} done")
    tab = pd.DataFrame(rows)
    utils.save_table(tab, cfg["profile"], "interaction_per_repeat")
    cols = [c for c in tab.columns if c not in ("rep", "method")]
    summ = utils.summarise_runs(tab, ["method"], cols)
    utils.save_table(summ, cfg["profile"], "interaction_summary")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    s_ = summ.set_index("method")
    for j, name in enumerate(["popularity", "knn", "mf", "social_mf"]):
        r = s_.loc[name]
        for a_, ycol in ((ax[0], "target_true_pref"), (ax[1], "distinct_items")):
            a_.errorbar(r.adopters_mean, r[ycol + "_mean"], xerr=r.adopters_std, yerr=r[ycol + "_std"],
                        marker="o", ms=6, color=PALETTE[j], capsize=2, label=name, ls="none")
    hyb = s_.loc[[m for m in s_.index if m.startswith("hybrid")]]
    for a_, ycol in ((ax[0], "target_true_pref"), (ax[1], "distinct_items")):
        a_.plot(hyb.adopters_mean, hyb[ycol + "_mean"], marker="s", ms=5, lw=1.2, color=PALETTE[4], label="hybrid (eta 0..1)")
        for m, r in hyb.iterrows():
            a_.annotate(m.split("=")[1], (r.adopters_mean, r[ycol + "_mean"]), fontsize=6, xytext=(3, 3),
                        textcoords="offset points", color="#52514e")
    ax[0].set(xlabel="total adopters (simulated)", ylabel="targets' true preference for their item",
              title="Individual satisfaction vs widespread adoption")
    ax[1].set(xlabel="total adopters (simulated)", ylabel="distinct items promoted to 100 users",
              title="Diversity of what spreads")
    ax[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(utils.figure_path(cfg["profile"], "fig8_tradeoff"))
    plt.close(fig)
    return {"interaction": summ.set_index("method")[["adopters_mean", "target_true_pref_mean", "adopter_pref_mean",
                                                     "distinct_items_mean", "observed_hit_rate_mean"]].to_dict()}


# =========================================================================== requirement 8 (intervention)
def exp_intervention(ctx):
    """Scenario: a medium-appeal item is pushed to the 20 most-followed users in
    the viral regime (lambda = 2 x default). Many users dislike the item, so
    unchecked spread produces many unwanted exposures. Interventions:
      * fan-out cap     : a share is shown to at most m followers;
      * relevance gate  : a share is shown only to followers whose PREDICTED
                          preference (recommender, not the hidden truth) >= tau;
      * account blocking: the b most-followed (or b random) users cannot share.
    """
    cfg, data, g = ctx["cfg"], ctx["data"], ctx["graph"]
    iv = cfg["intervention"]
    plt = utils.setup_matplotlib()
    rng = np.random.default_rng(cfg["seed"] + 8)
    model = ctx["models0"]["social_mf"][0]
    n = data["n_users"]
    items = ctx.get("campaign_items") or campaign_items(ctx)[0]
    item = items["medium"]
    by_deg = np.argsort(-g["out_deg"], kind="stable")
    targets = np.sort(by_deg[:20])
    top = np.full(len(targets), item)
    lam = 2 * cfg["diffusion"]["lambda"]
    pred_all = rec.predict_rows(model, np.arange(n))
    conds = [("none", "-", {})]
    for m in iv["fanout_caps"]:
        if m:
            conds.append(("fan-out cap", m, {"max_fanout": m}))
    for tau in iv["gate_thresholds"]:
        if tau:
            conds.append(("relevance gate", tau, {"gate": tau}))
    for b in iv["block_counts"]:
        if b:
            # never block the promoted users themselves: the platform still delivers the campaign
            for how, order in (("block top-degree", by_deg[20:]), ("block random", rng.permutation(np.setdiff1d(np.arange(n), targets)))):
                mask = np.ones(n, dtype=bool)
                mask[order[:b]] = False
                conds.append((how, b, {"share_mask": mask}))
    rows = []
    for name, level, kw in conds:
        if "gate" in kw:
            tau = kw["gate"]
            res = _campaign(ctx, top, targets, iv["n_sims"], rng, lam=lam, item_receive_mask=lambda i: pred_all[:, i] >= tau)
        else:
            res = _campaign(ctx, top, targets, iv["n_sims"], rng, lam=lam, **kw)
        row = {"intervention": name, "level": level}
        for k, v in res.items():
            row[k + "_mean"] = float(v.mean())
            row[k + "_std"] = float(v.std())
        row["unwanted_exposure_rate"] = row["unwanted_exposures_mean"] / max(row["exposed_mean"], 1e-9)
        row["exposure_conversion"] = row["secondary_adopters_mean"] / max(row["exposed_mean"], 1e-9)
        rows.append(row)
    tab = pd.DataFrame(rows)
    utils.save_table(tab, cfg["profile"], "intervention_effects")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    labels = [f"{r.intervention} {r.level}" for r in tab.itertuples()]
    x = np.arange(len(tab))
    panels = [("adopters", "total adopters", "Spread"),
              ("adopters_satisfied_frac", "share of adopters with true pref >= 4", "Satisfaction of adopters"),
              ("unwanted_exposures", "exposures to users with true pref < 3", "Unwanted exposures")]
    for a_, (col, ylab, title) in zip(ax, panels):
        a_.bar(x, tab[col + "_mean"], 0.7, yerr=tab[col + "_std"], color=PALETTE[0], capsize=2)
        a_.set(xticks=x, ylabel=ylab, title=title)
        a_.set_xticklabels(labels, rotation=75, fontsize=6)
    fig.tight_layout()
    fig.savefig(utils.figure_path(cfg["profile"], "fig9_intervention"))
    plt.close(fig)
    return {"intervention": tab[["intervention", "level", "adopters_mean", "adopter_pref_mean", "unwanted_exposure_rate"]].to_dict("records")}

