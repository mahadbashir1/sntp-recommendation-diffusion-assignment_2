"""Entry point: prepares data (if needed) and runs every experiment.

    python src/run_all.py --profile demo            # small sample, ~2-4 minutes
    python src/run_all.py --profile full            # full experiments
    python src/run_all.py --profile full --only rec,social

Experiment keys: dataset, rec, social, diffusion, im, groups, interaction,
intervention, cost (requirement numbers are listed in README.md).
"""
import argparse
import json
import time

import cost_analysis
import data_prep
import experiments as ex
import utils

STEPS = [("dataset", ex.exp_dataset), ("rec", ex.exp_recommendation), ("social", ex.exp_social),
         ("diffusion", ex.exp_diffusion), ("im", ex.exp_influence_max), ("groups", ex.exp_groups),
         ("interaction", ex.exp_interaction), ("intervention", ex.exp_intervention), ("cost", cost_analysis.exp_cost)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="demo", choices=["demo", "full"])
    ap.add_argument("--only", default="", help="comma separated experiment keys")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    t_start = time.time()
    data_prep.prepare(args.profile)
    ctx = ex.build_context(args.profile)
    summary, timings = {"hyperparameters": ctx["hp"]}, {}
    for key, fn in STEPS:
        if only and key not in only:
            continue
        utils.log(f"=== experiment '{key}' ===")
        t0 = time.time()
        summary[key] = fn(ctx)
        timings[key] = round(time.time() - t0, 1)
        utils.log(f"'{key}' finished in {timings[key]} s")
    summary["runtime_seconds"] = timings
    summary["total_seconds"] = round(time.time() - t_start, 1)
    name = "summary" if not only else "summary_" + "_".join(sorted(only))
    utils.save_json(summary, args.profile, name)
    utils.log(f"all done in {summary['total_seconds']} s - results in {utils.results_dir(args.profile)}")
    print(json.dumps(timings, indent=1))


if __name__ == "__main__":
    main()
