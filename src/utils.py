"""Shared helpers: configuration, paths, saving results, timing and plotting.

Everything is written as plain functions (no classes) so that each step of the
pipeline can be read top-to-bottom.
"""
import json
import os
import time
import tracemalloc

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "config.json")


def load_config(profile):
    """Return the settings of one profile ('demo' or 'full') merged over 'common'."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    merged = json.loads(json.dumps(cfg["common"]))
    for section, values in cfg["profiles"][profile].items():
        if isinstance(values, dict):
            merged.setdefault(section, {}).update(values)
        else:
            merged[section] = values
    merged["profile"] = profile
    return merged


def processed_dir(profile):
    return os.path.join(ROOT, "data", "processed", profile)


def results_dir(profile):
    return os.path.join(ROOT, "results", profile)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_table(df, profile, name):
    path = os.path.join(ensure_dir(os.path.join(results_dir(profile), "tables")), name + ".csv")
    df.to_csv(path, index=False, float_format="%.5f")
    return path


def save_json(obj, profile, name):
    path = os.path.join(ensure_dir(results_dir(profile)), name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_str_keys(obj), f, indent=2, default=_json_default)
    return path


def _str_keys(obj):
    if isinstance(obj, dict):
        return {(k if isinstance(k, (str, int, float, bool)) else str(k)): _str_keys(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_str_keys(v) for v in obj]
    return obj


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(str(type(x)))


def figure_path(profile, name):
    return os.path.join(ensure_dir(os.path.join(results_dir(profile), "figures")), name + ".png")


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def measure(fn, *args, track_memory=False, **kwargs):
    """Run fn(*args, **kwargs); return (result, seconds, peak_MB).

    Peak memory is measured with tracemalloc (numpy allocations are reported to
    it). Tracing slows Python down, so runtime is only trusted when
    track_memory=False.
    """
    if track_memory:
        tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    secs = time.perf_counter() - t0
    peak = 0.0
    if track_memory:
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
    return out, secs, peak


def mean_ci(values, z=1.96):
    """Mean, standard deviation and 95% normal-approximation CI half-width."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return np.nan, np.nan, np.nan
    sd = v.std(ddof=1) if v.size > 1 else 0.0
    return v.mean(), sd, z * sd / np.sqrt(v.size)


def summarise_runs(df, group_cols, value_cols):
    """Aggregate repeated trials into mean / std columns."""
    agg = df.groupby(group_cols)[value_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 140, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False,
    })
    return plt
