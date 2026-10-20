# SNTP Assignment 2 — Recommendation & Information Diffusion in a Social Network

| | |
|---|---|
| **Name** | Muhammad Mahad Bashir |
| **Roll number** | 23F-3011 |
| **Course** | SNTP (Social Network Theory & Practice) — Assignment 2 |
| **Dataset** | Epinions trust network + product ratings (Massa & Avesani), the dataset supplied with the assignment |

The system learns user preferences from Epinions ratings with two recommenders written from scratch (user-kNN and matrix factorisation), adds trust information to one of them (SocialMF), lets recommended items spread through the trust network (Independent Cascade and SIR), chooses whom to promote to under a budget (CELF lazy greedy with uncertainty estimates), recommends to groups (average / least misery / most pleasure), and evaluates recommendation quality, diffusion and their interaction, as well as interventions and computational cost.

All code is plain Python functions; it uses no classes.

---

## 1. Folder map

```
23F3011_SNTP_Assignment/
├── README.md                  <- this file
├── requirements.txt           <- exact dependency versions
├── config/config.json         <- every experiment setting, random seed and grid ('demo' and 'full' profiles)
├── data/
│   ├── raw/ratings_data.txt   <- original Epinions ratings (user item rating), supplied with the assignment
│   ├── raw/trust_data.txt     <- original Epinions trust statements (truster trustee 1)
│   └── processed/<profile>/   <- prepared samples (demo, full): ratings.csv, trust.csv, users.csv, items.csv, meta.json
│                                 (included; regenerated identically by data_prep.py)
├── src/
│   ├── run_all.py             <- entry point: prepares data and runs every experiment
│   ├── data_prep.py           <- Req 1: loading, filtering, snowball sampling, train/test splits
│   ├── recommenders.py        <- Req 2/3: popularity, user-kNN, biased MF (ALS), SocialMF
│   ├── metrics.py             <- Req 7: RMSE/MAE, accuracy, precision/recall/NDCG, coverage, diversity, novelty
│   ├── diffusion.py           <- Req 4: influence graph, edge strengths, IC and SIR simulators, null graphs
│   ├── influence_max.py       <- Req 5: live-edge worlds, CELF / naive greedy, baselines, overlap measures
│   ├── group_rec.py           <- Req 6: group formation, conflict, aggregation strategies, satisfaction
│   ├── experiments.py         <- experiments for Req 1-8 (tables + figures)
│   ├── cost_analysis.py       <- Req 8: runtime/memory scaling, bottleneck, CELF & SCC optimisations
│   └── utils.py               <- config loading, saving, timing/memory measurement, plotting style
├── results/
│   ├── full/                  <- results of the full run reported in the report
│   │   ├── summary.json       <- headline numbers + runtime of every experiment
│   │   ├── tuned_hyperparameters.json
│   │   ├── tables/*.csv       <- every result table
│   │   └── figures/*.png      <- every figure used in the report
│   └── demo/                  <- results of the small demonstration run
└── report/
    └── 23F3011_SNTP_Assignment_Report.pdf   <- written report (searchable PDF)
```

## 2. Software environment

* Python **3.11** (tested with 3.11.4 on Windows 11; any OS with Python ≥ 3.10 works)
* `numpy 1.26.4`, `scipy 1.15.3`, `pandas 2.2.3`, `matplotlib 3.10.0` (see `requirements.txt`)
* No GPU and no internet connection needed; the raw data are already in `data/raw/`.

Run these commands **from this folder** (the outer submission folder):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Preparing the data

```bash
python src/data_prep.py --profile demo     # 600-user sample  -> data/processed/demo/
python src/data_prep.py --profile full     # 3000-user sample -> data/processed/full/
```

`run_all.py` also runs this step automatically. Data preparation reads `ratings_data.txt` and `trust_data.txt` from `data/raw/`. It drops self-trust, keeps items with ≥ 20 ratings and users with ≥ 3 ratings who appear in the trust network, and then takes a snowball (BFS) sample of `max_users` users starting from a random user (seed 42). Finally it keeps items with ≥ 5 ratings inside the sample. If the raw files are lost, the original dataset is the *Epinions* dataset by Massa & Avesani ("Trust-aware recommender systems", RecSys 2007). It is distributed on Paolo Massa's trustlet.org pages and on many mirrors, such as Kaggle's "Epinions dataset" (files `ratings_data.txt` and `trust_data.txt`). Put the two text files into `data/raw/`.

## 4. Running

| What | Command (from this folder) | Runtime* | Memory |
|---|---|---|---|
| Small demonstration (600 users, 1 split) | `python src/run_all.py --profile demo` | ≈ 30 s | < 1 GB |
| Full experiments (3000 users, 3 splits, all scaling runs) | `python src/run_all.py --profile full` | ≈ 6–7 min | ≈ 2 GB peak |
| A subset of the experiments | `python src/run_all.py --profile full --only rec,social` | — | — |

\*Measured on a 12-thread laptop CPU with 16 GB RAM. The full run is dominated by the cost-scaling experiment (networks up to 8000 users) and by influence maximisation in the viral regime.

Experiment keys for `--only`: `dataset, rec, social, diffusion, im, groups, interaction, intervention, cost`.
All outputs go to `results/<profile>/` (`tables/*.csv`, `figures/*.png`, `summary.json`). Every random number generator is seeded from `config.json` (`seed = 42`, split seeds `0, 1, 2`), so a rerun reproduces the tables. The only exception is the runtime and memory measurements, which depend on the machine.

## 5. Requirement map

| # | Requirement | Source (functions) | Results | Report section |
|---|---|---|---|---|
| 1 | Social environment | `data_prep.py` (`core_filter`, `bfs_sample`, `build_sample`, `make_split`); `diffusion.edge_agreement`; `experiments.exp_dataset` | `dataset_statistics.csv`, `observed_social_correlation.csv`, fig1, fig1b | §2 |
| 2 | Two recommenders + behaviour on sparse users/items/rating styles | `recommenders.fit_user_knn`, `knn_rows`, `rank_rows`, `fit_mf`, `_ridge_solve`; `experiments.exp_recommendation`, `_bucket_analysis`, `_cold_start_simulation`, `exp_tuning` | `rec_summary.csv`, `rec_rmse_by_bucket.csv`, `rec_cold_start_simulation.csv`, `tuning_validation.csv`, fig2, fig3 | §3 |
| 3 | Social extension, friends vs similar strangers, adjustable social weight | `recommenders.fit_mf(social=T, beta=β)`, `knn_from_neighbours`; `experiments.exp_social`, `_friends_vs_similar` | `social_beta_sweep.csv`, `social_friends_vs_similar.csv`, `social_effect_by_agreement_history.csv`, fig4 | §4 |
| 4 | Receive / expose / adopt; IC + SIR; preferences, structure, influence strength | `diffusion.simulate`, `run_many`, `adoption_prob`, `attention_prob`, `rewire_degree_preserving`, `random_graph`; `experiments.exp_diffusion` | `diffusion_experiments.csv`, `diffusion_network_structure.csv`, fig5 | §5 |
| 5 | Budgeted seeding with marginal gain + uncertainty | `influence_max.sample_worlds`, `initial_gains`, `celf`, `degree_seeds`, `random_seeds`, `top_individual_seeds`, `audience_overlap`; `experiments.exp_influence_max` | `im_budget_comparison.csv`, `im_celf_steps.csv`, `im_uncertainty_worlds.csv`, fig6 | §6 |
| 6 | Group recommendation and spread through the group | `group_rec.py`; `experiments.exp_groups` | `group_summary.csv`, `group_per_group.csv`, `group_candidate_spread.csv`, fig7 | §7 |
| 7 | Evaluation of the complete system | `metrics.py`; `data_prep.split_per_user`; `experiments.exp_recommendation`, `exp_interaction`, `_campaign`, `_hybrid_scores` | `rec_per_seed.csv`, `interaction_summary.csv`, fig8 | §8 |
| 8 | Intervention + runtime/memory + optimisation | `experiments.exp_intervention`; `cost_analysis.exp_cost`; `influence_max.celf` vs `naive_greedy`; `initial_gains(use_scc)` | `intervention_effects.csv`, `cost_*.csv`, fig9, fig10 | §9 |
| — | Final discussion, limitations | — | `summary.json` | §10, §11 |

## 6. Known limitations

* **Sample, not the full graph.** The full experiments use a 3000-user snowball sample, which is denser than Epinions as a whole. A snowball sample over-represents well-connected users, so absolute spread sizes should not be extrapolated to the whole platform.
* **The simulator's "true preferences" are inferred.** Where a rating is not observed, it is a blend of kNN and MF predictions fitted on all observed ratings. Diffusion, group and intervention outcomes are therefore what-if results under the model. Because the oracle comes from the same model families, it tends to agree with the kNN/MF recommenders, which inflates their simulated satisfaction (see report §8).
* **Edge strengths are inferred** from rating agreement on training data. Epinions contains no observed propagation events, so influence probabilities cannot be validated against real cascades. λ is a scenario parameter, not an estimate.
* **Rating-trained MF is a weak top-N ranker** on the full catalogue (missing-not-at-random data). Validation selected strong regularisation, so plain MF behaves almost like the bias baseline.
* **Small effect sizes.** SocialMF improves RMSE by about 0.005–0.011. The improvement is consistent over 3 splits, but it is small in absolute terms.
* Runtime and memory figures depend on the machine. Memory is measured with `tracemalloc` (Python/numpy allocations only).
