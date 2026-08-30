#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-series forecast between Series 1 and Series 2 (weekly resolution).

A CNN ensemble is trained on AAFT surrogates from one series and applied,
without retraining, to forecast the other. Renders a 3x2 panel: index and
trend, out-of-sample SDML probability, and ROC curves against the classical
indicators. Trained ensembles are cached under output/ and reused.

Usage:
    python src/make_cross_forecast.py [--span 0.5] [--burnin short|span]

Note: with --burnin span, a burn-in proportional to the detrending span can
coincide with the 50/50 label split and leave a single neutral evaluation
point; the default --burnin short starts scoring once max(L, w) observations
have accrued.
"""
import os, sys, time, pickle
import numpy as np
import pandas as pd

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SRC_DIR)
FIG_DIR  = os.path.join(REPO_DIR, "figures")
OUT_DIR  = os.path.join(REPO_DIR, "output")
sys.path.insert(0, SRC_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sdml_pipeline_weekly as base
import tensorflow as tf


def _argval(flag, default, cast=str):
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return cast(sys.argv[i + 1])
    return default

SPAN   = _argval("--span", 0.5, float)
BURNIN = _argval("--burnin", "short", str).lower()
TAG    = f"span{int(round(SPAN * 100)):03d}"

base.LOWESS_FRAC = SPAN
METHOD = "AAFT"
L      = base.L


def halves(cfg):
    dates, x = base.load_series(cfg)
    resid = x - base.lowess_smooth(x, base.LOWESS_FRAC)
    cut = int(round((1.0 - base.PRE_FRAC) * len(x)))
    return dates, x, resid, cut


def build_training(cfg):
    _, _, resid, cut = halves(cfg)
    surr_neu = base.generate_surrogates(resid[:cut], base.SURR_N_PER_CLASS, METHOD, seed=0)
    surr_pre = base.generate_surrogates(resid[cut:], base.SURR_N_PER_CLASS, METHOD, seed=1)
    X_neu = base.extract_windows_lastL(surr_neu, L, normalize=True)
    X_pre = base.extract_windows_lastL(surr_pre, L, normalize=True)
    X = np.concatenate([X_neu, X_pre], axis=0)[:, :, np.newaxis]
    y = np.concatenate([np.zeros(len(X_neu)), np.ones(len(X_pre))])
    return X, y


def get_ensemble(train_id):
    """Load the cached ensemble for this span/train series, or train it."""
    mdir = os.path.join(OUT_DIR, f"models_cross_{TAG}_trainS{train_id}")
    paths = [os.path.join(mdir, f"model_{s}.keras") for s in range(base.N_MODELS)]
    if all(os.path.exists(p) for p in paths):
        print(f"[train S{train_id}] loading cached ensemble ...")
        return [tf.keras.models.load_model(p) for p in paths]
    print(f"[train S{train_id}] span={SPAN}: surrogates + training ...")
    os.makedirs(mdir, exist_ok=True)
    X, y = build_training(base.CFG_LIST[train_id - 1])
    return base.train_ensemble(X, y, L, base.N_MODELS, mdir)


def predict_and_eval(cfg, trained, train_name, train_id):
    name = cfg["name"]
    dates, x_raw, resid, cut_ix = halves(cfg)
    n_pre   = len(x_raw)
    t_num   = mdates.date2num(pd.to_datetime(dates))
    t_trans = mdates.date2num(pd.Timestamp(cfg["TRANSITION_DATE"]))
    smooth  = x_raw - resid

    w = max(10, int(base.EWS_WINDOW_FRAC * n_pre))
    if BURNIN == "span":
        warmup = max(20, int(base.LOWESS_FRAC * n_pre))
        i_start = min(max(max(L, w, warmup) - 1, 1), n_pre - 1)
    else:
        i_start = max(L, w) - 1
    i_stop = n_pre - 1

    pos, mu, sd = base.sdml_predict_incremental(resid, L, trained, base.INC, i_start, i_stop)

    mu_v, sd_v = mu.copy(), sd.copy()
    if base.SMOOTH_SDML:
        mu_v = base.ema_causal(mu_v, base.SDML_EMA_ALPHA)
        if base.SDML_TRIANG_WIN > 0:
            mu_v = base.causal_triangular(mu_v, base.SDML_TRIANG_WIN)
        sd_v = base.ema_causal(sd_v, base.SDML_EMA_ALPHA)

    var_r, ac1_r = base.rolling_var_ac1(resid, w)

    bnd_s = max(1, min(len(pos) - 1, int(np.searchsorted(pos, cut_ix, side="left"))))
    yt_s  = np.zeros(len(pos), int); yt_s[bnd_s:] = 1
    fpr_s, tpr_s, auc_s = base.roc_safe(yt_s, mu)

    valid = np.where(np.isfinite(var_r) & np.isfinite(ac1_r))[0]
    bnd_e = max(1, min(len(valid) - 1, int(np.searchsorted(valid, cut_ix, side="left"))))
    yt_e  = np.zeros(len(valid), int); yt_e[bnd_e:] = 1
    fpr_v, tpr_v, auc_v = base.roc_safe(yt_e, var_r[valid])
    fpr_a, tpr_a, auc_a = base.roc_safe(yt_e, ac1_r[valid])

    print(f"  train[{train_name}] -> {name:<10}  AUC SDML={auc_s:.3f}  Var={auc_v:.3f}  "
          f"AC1={auc_a:.3f}  (neg SDML={int((yt_s == 0).sum())})")

    return dict(
        name=f"{name}  (train: {train_name})",
        t_num=t_num, x=x_raw, x_smooth=smooth, t_trans_num=t_trans,
        pos=pos, mu_raw=mu, sd_raw=sd, mu_v=mu_v, sd_v=sd_v,
        fpr_sdml=fpr_s, tpr_sdml=tpr_s, auc_sdml=auc_s,
        fpr_var=fpr_v, tpr_var=tpr_v, auc_var=auc_v,
        fpr_ac1=fpr_a, tpr_ac1=tpr_a, auc_ac1=auc_a,
        surrogate_method=f"{METHOD} (train S{train_id}, span {SPAN})",
        _auc_sdml=auc_s, _test_name=name,
    )


t0 = time.time()
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

results_by_train = {}
for train_id, test_id in [(1, 2), (2, 1)]:
    train_name = base.CFG_LIST[train_id - 1]["name"]
    trained = get_ensemble(train_id)
    res = [predict_and_eval(base.CFG_LIST[test_id - 1], trained, train_name, train_id)]
    results_by_train[train_id] = res
    with open(os.path.join(OUT_DIR, f"results_cross_{TAG}_trainS{train_id}.pkl"), "wb") as f:
        pickle.dump(res, f)

# ── Figure ──
INDEX_NAME = "Critical Transition Index"
COLS = [
    dict(res=results_by_train[1][0], title="Train: Series 1  →  Forecast: Series 2"),
    dict(res=results_by_train[2][0], title="Train: Series 2  →  Forecast: Series 1"),
]

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10})
fig, ax = plt.subplots(3, 2, figsize=(12.5, 10.5), constrained_layout=True)
LETTERS = [["A", "B"], ["C", "D"], ["E", "F"]]

for c, col in enumerate(COLS):
    r      = col["res"]
    t_num  = np.asarray(r["t_num"], float)
    pos    = np.asarray(r["pos"], int)
    t_tr   = r["t_trans_num"]

    a0 = ax[0, c]
    a0.plot(t_num, r["x"],        color="0.55", lw=0.8, label=INDEX_NAME)
    a0.plot(t_num, r["x_smooth"], color="k",    lw=1.5, label=f"LOWESS trend (span {SPAN})")
    a0.axvline(t_tr, color="crimson", ls="--", lw=1.3, label="critical transition")
    a0.set_title(col["title"], fontweight="bold", pad=8)
    a0.set_ylabel(INDEX_NAME)
    if c == 0:
        a0.legend(loc="upper left", framealpha=0.9)

    a1 = ax[1, c]
    tp = t_num[pos]
    a1.fill_between(tp, r["mu_v"] - r["sd_v"], r["mu_v"] + r["sd_v"], color="C0", alpha=0.18)
    a1.plot(tp, r["mu_v"], color="C0", lw=1.9, label="SDML probability")
    a1.axvline(t_tr, color="crimson", ls="--", lw=1.3)
    a1.set_ylim(-0.02, 1.02)
    a1.set_ylabel("P(approaching transition)")
    a1.text(0.97, 0.07, f"AUC = {r['auc_sdml']:.3f}", transform=a1.transAxes,
            fontsize=11, fontweight="bold", color="C0", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="C0", alpha=0.9))
    if c == 0:
        a1.legend(loc="upper left", framealpha=0.9)

    for a in (a0, a1):
        a.xaxis.set_major_locator(mdates.YearLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    a2 = ax[2, c]
    a2.plot([0, 1], [0, 1], color="0.7", ls=":", lw=1)
    a2.plot(r["fpr_sdml"], r["tpr_sdml"], color="C0", lw=2.2,
            label=f"SDML (AUC={r['auc_sdml']:.2f})")
    a2.plot(r["fpr_var"],  r["tpr_var"],  color="0.35", lw=1.6,
            label=f"Variance (AUC={r['auc_var']:.2f})")
    a2.plot(r["fpr_ac1"],  r["tpr_ac1"],  color="darkorange", lw=1.6,
            label=f"Lag-1 autocorr. (AUC={r['auc_ac1']:.2f})")
    a2.set_xlim(-0.02, 1.02); a2.set_ylim(-0.02, 1.02)
    a2.set_xlabel("False positive rate"); a2.set_ylabel("True positive rate")
    a2.set_aspect("equal", adjustable="box")
    # Left column evaluates Series 2, whose variance ROC has a step near the
    # lower-right corner; the legend moves to the free mid band there.
    anchor = (0.97, 0.26) if c == 0 else (0.97, 0.06)
    a2.legend(loc="lower right", frameon=False, bbox_to_anchor=anchor)

    # Panel letters outside the plotting area so legends never cover them.
    for rrow in range(3):
        ax[rrow, c].text(-0.09, 1.02, LETTERS[rrow][c], transform=ax[rrow, c].transAxes,
                         fontsize=14, fontweight="bold", va="bottom", ha="left", clip_on=False)

out = os.path.join(FIG_DIR, f"cross_forecast_S1S2_weekly_{TAG}.pdf")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("\nsaved ->", out)
print(f"Total: {time.time() - t0:.0f}s")
