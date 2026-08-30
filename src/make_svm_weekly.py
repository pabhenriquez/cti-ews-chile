#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM variant of the weekly SDML analysis. Same data conventions, detrending,
surrogate split, and evaluation as sdml_pipeline_weekly.py, with an RBF SVM
ensemble instead of the multi-head CNN.

Usage:
    python src/make_svm_weekly.py [--method AAFT|FT|IAAFT|RP]
"""
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kendalltau

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys
SURROGATE_METHOD = "AAFT"
for _i, _a in enumerate(sys.argv):
    if _a == "--method" and _i + 1 < len(sys.argv):
        SURROGATE_METHOD = sys.argv[_i + 1].upper()

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")
FIG_DIR  = os.path.join(REPO_DIR, "figures")
OUT_DIR  = os.path.join(REPO_DIR, "output")

CFG_LIST = [
    dict(
        name="Series 1",
        DATA_PATH=os.path.join(DATA_DIR, "serie1.csv"),
        DATE_COL="fecha",
        VALUE_COL="indice_cti",
        METHOD_COL=None,
        METHOD_FILTER=None,
        TEMPORALIDAD_COL="Temporalidad",
        TEMPORALIDAD_FILTER="Semanal",
        START_DATE_USED="2014-01-01",
        TRANSITION_DATE="2020-03-31",
    ),
    dict(
        name="Series 2",
        DATA_PATH=os.path.join(DATA_DIR, "serie2.csv"),
        DATE_COL="time",
        VALUE_COL="Value",
        METHOD_COL="Method",
        METHOD_FILTER="Original",
        TEMPORALIDAD_COL="Frequency",
        TEMPORALIDAD_FILTER="Weekly",
        START_DATE_USED="2014-01-01",
        TRANSITION_DATE="2020-03-31",
    ),
    dict(
        name="Series 3",
        DATA_PATH=os.path.join(DATA_DIR, "serie3.csv"),
        DATE_COL="time",
        VALUE_COL="Value",
        METHOD_COL="Method",
        METHOD_FILTER="Original",
        TEMPORALIDAD_COL="Frequency",
        TEMPORALIDAD_FILTER="Weekly",
        START_DATE_USED="2014-01-01",
        TRANSITION_DATE="2021-12-31",
    ),
]

DETREND_METHOD  = "lowess"
GAUSS_SIGMA     = 56
LOWESS_FRAC     = 0.40

EWS_WINDOW_PRE_FRAC = 0.30
SMOOTH_EWS          = True
EWS_MOV_WIN         = 5

PRE_FRAC            = 0.50
L                  = 100
INC                = 3
SURR_N_PER_CLASS   = 800
N_MODELS           = 10
SMOOTH_SDML        = True
SDML_EMA_ALPHA     = 0.1
SDML_TRIANG_WIN    = 9

FONTSIZE_LABELS     = 13
FONTSIZE_TICKS      = 12
FONTSIZE_LEGEND_ROC = 11
COLOR_GREY = "0.40"
COLOR_BLACK= "0.05"

ROW_LETTERS = ["A", "B", "C", "D", "E"]

def add_row_letter(ax, letter):
    ax.text(
        -0.25, 0.5, letter,
        transform=ax.transAxes,
        ha="right", va="center",
        fontsize=FONTSIZE_LABELS + 3,
        fontweight="bold",
        clip_on=False
    )

def gaussian_smooth(x, sigma):
    return gaussian_filter1d(np.asarray(x, float), sigma=float(sigma), mode="nearest") if sigma > 0 else np.asarray(x, float)

def residuals_gaussian_full(x, sigma):
    return np.asarray(x, float) - gaussian_smooth(x, sigma)

def lowess_smooth(x, frac):
    x = np.asarray(x, float)
    t = np.arange(len(x), dtype=float)
    return lowess(x, t, frac=float(frac), return_sorted=False)

def residuals_lowess_full(x, frac):
    return np.asarray(x, float) - lowess_smooth(x, frac)

def rolling_var_ac1_pre(resid, tcut, w):
    n = len(resid)
    var = np.full(n, np.nan)
    ac1 = np.full(n, np.nan)
    if w < 2 or tcut < w - 1:
        return var, ac1
    for i in range(w - 1, min(tcut, n - 1) + 1):
        seg = resid[i - w + 1:i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) < 2:
            continue
        v = np.var(seg, ddof=1) if len(seg) > 1 else np.nan
        var[i] = v
        x = seg - np.mean(seg)
        denom = np.dot(x, x)
        ac1[i] = (np.dot(x[1:], x[:-1]) / denom) if denom > 0 else np.nan
    return var, ac1

def ema_causal_nan(y, alpha=0.25):
    y = np.asarray(y, float)
    s = np.full_like(y, np.nan)
    idx = np.where(np.isfinite(y))[0]
    if len(idx) == 0:
        return s
    s[idx[0]] = y[idx[0]]
    for k in idx[1:]:
        prev = s[k - 1] if np.isfinite(s[k - 1]) else s[idx[0]]
        s[k] = alpha * y[k] + (1 - alpha) * prev
    return s

def movmean_causal_nan(y, win=5):
    y = pd.Series(y, dtype=float)
    m = y.rolling(window=max(1, int(win)), min_periods=1, center=False).mean()
    first = np.where(~y.isna())[0]
    if len(first) == 0:
        return y.to_numpy()
    m.iloc[:first[0]] = np.nan
    return m.to_numpy()

def causal_triangular(y, w=5):
    if w < 2:
        return np.asarray(y, float)
    mm  = pd.Series(y).rolling(w, min_periods=1).mean().to_numpy()
    tri = pd.Series(mm).rolling(w, min_periods=1).mean().to_numpy()
    return tri

def normalize_window(wv, eps=1e-3):
    wv = np.asarray(wv, float)
    m = np.nanmean(np.abs(wv))
    s = np.nanstd(wv)
    scale = max(m, 0.5 * s, eps)
    return wv / scale

def windows_lastL(arr, L_):
    arr = np.asarray(arr, float)
    if len(arr) >= L_:
        return arr[-L_:]
    out = np.zeros(L_, float)
    out[-len(arr):] = arr
    return out

def split_neutral_pre_by_frac(rpre, pre_frac=PRE_FRAC):
    rpre = np.asarray(rpre, float)
    cut  = int(round((1.0 - pre_frac) * len(rpre)))
    cut  = min(max(cut, 1), len(rpre) - 1)
    return rpre[:cut].copy(), rpre[cut:].copy(), cut

from numpy.fft import rfft, irfft

def surrogate_rp(x, n=1, rng=None):
    rng = np.random.default_rng(rng)
    x = np.asarray(x, float)
    return np.array([rng.permutation(x) for _ in range(n)])

def surrogate_ft(x, n=1, rng=None):
    rng = np.random.default_rng(rng)
    x = np.asarray(x, float)
    N = len(x)
    mag = np.abs(rfft(x))
    X = []
    for _ in range(n):
        phase = rng.uniform(0, 2 * np.pi, mag.shape)
        phase[0] = 0.0
        if N % 2 == 0:
            phase[-1] = 0.0
        y = irfft(mag * np.exp(1j * phase), n=N)
        X.append(y)
    return np.array(X)

def erfinv(y):
    a = 0.147
    y = np.asarray(y, float)
    ln = np.log(1 - y**2)
    return np.sign(y) * np.sqrt(np.sqrt((2 / (np.pi * a) + ln / 2.0)**2 - ln / a) - (2 / (np.pi * a) + ln / 2.0))

def surrogate_aaft(x, n=1, rng=None):
    rng = np.random.default_rng(rng)
    x = np.asarray(x, float)
    X = []
    xr = pd.Series(x).rank(method="average").to_numpy() / (len(x) + 1.0)
    z  = np.sqrt(2.0) * erfinv(2 * xr - 1.0)
    for _ in range(n):
        y = surrogate_ft(z, n=1, rng=rng)[0]
        r = pd.Series(y).rank(method="first").astype(int).to_numpy() - 1
        xs = np.sort(x)
        X.append(xs[np.clip(r, 0, len(xs) - 1)])
    return np.array(X)

def surrogate_iaaft(x, n=1, iters=200, rng=None):
    rng = np.random.default_rng(rng)
    x = np.asarray(x, float)
    N = len(x)
    Xf_mag = np.abs(rfft(x))
    xs = np.sort(x)
    X = []
    for _ in range(n):
        y = rng.permutation(x)
        for _ in range(iters):
            Yf = rfft(y)
            y  = irfft(Xf_mag * np.exp(1j * np.angle(Yf)), n=N)
            r  = pd.Series(y).rank(method="first").astype(int).to_numpy() - 1
            y  = xs[np.clip(r, 0, N - 1)]
        X.append(y)
    return np.array(X)

def get_surrogate_generator(method):
    m = str(method).upper()
    if m == "AAFT":
        return surrogate_aaft
    if m == "FT":
        return surrogate_ft
    if m == "RP":
        return surrogate_rp
    if m == "IAAFT":
        return lambda x, n, rng=None: surrogate_iaaft(x, n=n, iters=200, rng=rng)
    raise ValueError("SURROGATE_METHOD must be 'AAFT', 'FT', 'IAAFT' or 'RP'.")

def build_svm_classifier(seed=0):
    svm = SVC(
        C=3.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=seed
    )
    return make_pipeline(StandardScaler(with_mean=True, with_std=True), svm)

def sdml_predict_incremental_pre_fixed_svm(t_num, resid_pre_all, L, models, inc, start_idx, stop_idx):
    means, stds, tt, ipos = [], [], [], []
    for i in range(max(L, start_idx) + 1, stop_idx + 1, inc):
        xr = np.asarray(resid_pre_all[:i], float)
        win = normalize_window(windows_lastL(xr, L))[None, :]
        probs = [float(m.predict_proba(win)[0, 1]) for m in models]
        means.append(np.mean(probs))
        stds.append(np.std(probs))
        tt.append(t_num[i - 1])
        ipos.append(i - 1)
    return np.array(tt), np.array(ipos), np.array(means), np.array(stds)

def roc_safe(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, float)
    if len(y_true) == 0 or len(scores) == 0 or len(np.unique(y_true)) < 2:
        return np.array([0, 1]), np.array([0, 1]), np.nan
    fpr, tpr, _ = roc_curve(y_true, scores)
    return fpr, tpr, roc_auc_score(y_true, scores)

def process_dataset(cfg, surrogate_method=SURROGATE_METHOD):
    # Exact-match filters, no date deduplication, series truncated at
    # START <= date <= TRANSITION (the transition is the last point).
    df = pd.read_csv(cfg["DATA_PATH"])

    if cfg.get("METHOD_COL") and cfg.get("METHOD_FILTER") is not None:
        df = df[df[cfg["METHOD_COL"]] == cfg["METHOD_FILTER"]]

    df = df[df[cfg["TEMPORALIDAD_COL"]] == cfg["TEMPORALIDAD_FILTER"]]

    date_col = cfg["DATE_COL"]
    value_col = cfg["VALUE_COL"]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    start_used = pd.Timestamp(cfg["START_DATE_USED"])
    trans_date = pd.Timestamp(cfg["TRANSITION_DATE"])
    df = df[(df[date_col] >= start_used) & (df[date_col] <= trans_date)].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"[{cfg['name']}] No data after filtering.")

    vals = df[value_col].values.astype(float)
    ok   = np.isfinite(vals)
    t_num = mdates.date2num(df[date_col].values.astype("datetime64[ns]")[ok])
    x     = vals[ok]
    n     = len(x)

    ix_trans = n - 1
    n_pre = n
    L_auto = max(128, min(500, int(0.8 * n_pre)))
    L_eff  = min(L, L_auto)

    if DETREND_METHOD.lower() == "gauss":
        resid_pre = residuals_gaussian_full(x[:n_pre], GAUSS_SIGMA)
        x_smooth  = gaussian_smooth(x, GAUSS_SIGMA)
    else:
        resid_pre = residuals_lowess_full(x[:n_pre], LOWESS_FRAC)
        x_smooth  = lowess_smooth(x, LOWESS_FRAC)

    resid = np.full(n, np.nan)
    resid[:n_pre] = resid_pre

    w = max(10, int(EWS_WINDOW_PRE_FRAC * n_pre))
    var_resid, ac1_resid = rolling_var_ac1_pre(resid, tcut=ix_trans, w=w)

    mask_tau = np.arange(n) <= ix_trans
    mask_tau[:w-1] = False
    idx_tau = np.where(mask_tau & np.isfinite(var_resid) & np.isfinite(ac1_resid))[0]
    tau_var = np.nan
    tau_ac1 = np.nan
    if len(idx_tau) > 3:
        t_seq = idx_tau.astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tau_var = kendalltau(t_seq, var_resid[idx_tau]).statistic
            tau_ac1 = kendalltau(t_seq, ac1_resid[idx_tau]).statistic

    gen = get_surrogate_generator(surrogate_method)

    x_neu_all, x_pre_all, cut_ix = split_neutral_pre_by_frac(resid_pre, PRE_FRAC)

    WARMUP_EXTRA = (3 * GAUSS_SIGMA) if DETREND_METHOD.lower() == "gauss" else max(20, int(LOWESS_FRAC * n_pre))
    i_start = max(L_eff, w, WARMUP_EXTRA) - 1
    i_stop  = ix_trans
    i_start = min(max(i_start, 1), i_stop)

    X_neu_s = gen(x_neu_all, n=SURR_N_PER_CLASS, rng=0)
    X_pre_s = gen(x_pre_all, n=SURR_N_PER_CLASS, rng=1)

    X_neu = np.stack([normalize_window(windows_lastL(s, L_eff)) for s in X_neu_s], axis=0)
    X_pre = np.stack([normalize_window(windows_lastL(s, L_eff)) for s in X_pre_s], axis=0)
    X_sdml = np.concatenate([X_neu, X_pre], axis=0)
    y_sdml = np.concatenate([np.zeros(len(X_neu)), np.ones(len(X_pre))]).astype(int)

    models_sdml, f1_list = [], []
    for seed in range(N_MODELS):
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_sdml, y_sdml, test_size=0.4, stratify=y_sdml, random_state=seed)
        X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)

        model = build_svm_classifier(seed=seed)
        model.fit(X_tr, y_tr)

        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        f1_list.append(f1_score(y_te, y_pred))
        models_sdml.append(model)

    print(f"[{cfg['name']}] SDML-SVM-{surrogate_method} F1 test: {np.mean(f1_list):.3f} ± {np.std(f1_list):.3f}")

    tt, ipos, mu, sd = sdml_predict_incremental_pre_fixed_svm(
        t_num=t_num, resid_pre_all=resid_pre, L=L_eff, models=models_sdml,
        inc=INC, start_idx=i_start, stop_idx=i_stop
    )

    boundary_sdml = int(np.searchsorted(ipos, cut_ix, side="left"))
    if boundary_sdml <= 0 or boundary_sdml >= len(ipos):
        boundary_sdml = max(1, min(len(ipos) - 1, len(ipos) // 2))
    y_true_sdml = np.zeros(len(ipos), dtype=int)
    y_true_sdml[boundary_sdml:] = 1

    fpr_sdml, tpr_sdml, auc_sdml = roc_safe(y_true_sdml, mu)

    valid_ews_idx = np.where((~np.isnan(var_resid)) & (~np.isnan(ac1_resid)) & (np.arange(n) <= ix_trans))[0]
    boundary_k = int(np.searchsorted(valid_ews_idx, cut_ix, side="left"))
    if boundary_k <= 0 or boundary_k >= len(valid_ews_idx):
        boundary_k = max(1, min(len(valid_ews_idx) - 1, len(valid_ews_idx) // 2))
    y_true_ews = np.zeros(len(valid_ews_idx), dtype=int)
    y_true_ews[boundary_k:] = 1

    scores_var = var_resid[valid_ews_idx]
    scores_ac1 = ac1_resid[valid_ews_idx]
    fpr_var,  tpr_var,  auc_var  = roc_safe(y_true_ews, scores_var)
    fpr_ac1,  tpr_ac1,  auc_ac1  = roc_safe(y_true_ews, scores_ac1)

    ac1_plot = movmean_causal_nan(ac1_resid, win=EWS_MOV_WIN) if SMOOTH_EWS else ac1_resid
    var_plot = movmean_causal_nan(var_resid, win=EWS_MOV_WIN) if SMOOTH_EWS else var_resid

    mu_v = mu.copy()
    if SMOOTH_SDML:
        mu_v = ema_causal_nan(mu_v, alpha=SDML_EMA_ALPHA)
        if SDML_TRIANG_WIN and SDML_TRIANG_WIN > 0:
            mu_v = causal_triangular(mu_v, w=SDML_TRIANG_WIN)
    sd_v = ema_causal_nan(sd, alpha=SDML_EMA_ALPHA) if SMOOTH_SDML else sd

    return dict(
        name=f"{cfg['name']}",
        ml_algo="SVM",
        surrogate_method=surrogate_method,
        t_num=t_num, x=x, x_smooth=x_smooth,
        ix_trans=ix_trans, t_trans_num=float(mdates.date2num(trans_date)),
        ac1_plot=ac1_plot, var_plot=var_plot,
        tau_ac1=tau_ac1, tau_var=tau_var,
        tt=tt, mu_v=mu_v, sd_v=sd_v,
        fpr_ac1=fpr_ac1, tpr_ac1=tpr_ac1, auc_ac1=auc_ac1,
        fpr_var=fpr_var, tpr_var=tpr_var, auc_var=auc_var,
        fpr_sdml=fpr_sdml, tpr_sdml=tpr_sdml, auc_sdml=auc_sdml
    )

results = [process_dataset(cfg, SURROGATE_METHOD) for cfg in CFG_LIST]

import pickle
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, f"results_svm_{SURROGATE_METHOD}_weekly.pkl"), "wb") as f:
    pickle.dump(results, f)

fig, axes = plt.subplots(5, 3, figsize=(18, 16), constrained_layout=True, sharex=False)

PAD_ROC = 0.02
def _clip_for_display(fpr, tpr, pad=PAD_ROC):
    return np.clip(fpr, pad, 1.0 - pad), np.clip(tpr, pad, 1.0 - pad)

for c, res in enumerate(results):
    t_num = res["t_num"]
    tmin = float(mdates.date2num(pd.Timestamp("2014-01-01")))
    tmax = float(mdates.date2num(pd.Timestamp("2024-01-01")))
    t_trans_num = res["t_trans_num"]

    ax = axes[0, c]
    ax.plot(t_num, res["x"], color=COLOR_GREY, lw=1.0)
    ax.plot(t_num, res["x_smooth"], color=COLOR_BLACK, lw=1.5)
    ax.axvline(t_trans_num, ls="--", color=COLOR_GREY, lw=1.0)
    ax.set_xlim(tmin, tmax)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='both', labelsize=FONTSIZE_TICKS)
    ax.set_title(res["name"], fontsize=FONTSIZE_LABELS)
    if c == 0:
        ax.set_ylabel("Critical transition index", fontsize=FONTSIZE_LABELS)
        add_row_letter(ax, ROW_LETTERS[0])

    ax = axes[1, c]
    ax.plot(t_num, res["ac1_plot"], color=COLOR_GREY, lw=1.25)
    ax.axvline(t_trans_num, ls="--", color=COLOR_GREY, lw=1.0)
    ax.set_xlim(tmin, tmax)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='both', labelsize=FONTSIZE_TICKS)
    if np.isfinite(res["tau_ac1"]):
        ax.text(0.98, 0.05, rf"$\tau$={res['tau_ac1']:.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=FONTSIZE_TICKS, color="black")
    if c == 0:
        ax.set_ylabel("Lag-1 AC", fontsize=FONTSIZE_LABELS)
        add_row_letter(ax, ROW_LETTERS[1])

    ax = axes[2, c]
    ax.plot(t_num, res["var_plot"], color=COLOR_GREY, lw=1.25)
    ax.axvline(t_trans_num, ls="--", color=COLOR_GREY, lw=1.0)
    ax.set_xlim(tmin, tmax)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='both', labelsize=FONTSIZE_TICKS)
    if np.isfinite(res["tau_var"]):
        ax.text(0.98, 0.05, rf"$\tau$={res['tau_var']:.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=FONTSIZE_TICKS, color="black")
    if c == 0:
        ax.set_ylabel("Variance", fontsize=FONTSIZE_LABELS)
        add_row_letter(ax, ROW_LETTERS[2])

    ax = axes[3, c]
    ax.plot(res["tt"], res["mu_v"], color=COLOR_GREY, lw=1.25)
    ax.fill_between(res["tt"],
                    np.clip(res["mu_v"] - res["sd_v"], 0, 1),
                    np.clip(res["mu_v"] + res["sd_v"], 0, 1),
                    color=COLOR_GREY, alpha=0.20)
    ax.axvline(t_trans_num, ls="--", color=COLOR_GREY, lw=1.0)
    ax.set_ylim(0, 1)
    ax.set_xlim(tmin, tmax)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='both', labelsize=FONTSIZE_TICKS)
    if c == 0:
        ax.set_ylabel("DL probability", fontsize=FONTSIZE_LABELS)
        add_row_letter(ax, ROW_LETTERS[3])
    ax.set_xlabel("Time", fontsize=FONTSIZE_LABELS)

    ax = axes[4, c]
    fpr, tpr = _clip_for_display(res["fpr_ac1"], res["tpr_ac1"])
    ax.plot(fpr, tpr, lw=1.8, linestyle="--", label=f"AC(1) (AUC={res['auc_ac1']:.3f})")
    fpr, tpr = _clip_for_display(res["fpr_var"], res["tpr_var"])
    ax.plot(fpr, tpr, lw=1.8, linestyle="-.", label=f"Variance (AUC={res['auc_var']:.3f})")
    fpr, tpr = _clip_for_display(res["fpr_sdml"], res["tpr_sdml"])
    ax.plot(fpr, tpr, lw=1.8, label=f"SDML-{res['surrogate_method']} ({res['ml_algo']}, AUC={res['auc_sdml']:.3f})")
    d = np.linspace(PAD_ROC, 1.0 - PAD_ROC, 200)
    ax.plot(d, d, ls=":", lw=1.0, color="0.6")
    ax.set_xlim(0.0 - PAD_ROC, 1.0 + PAD_ROC)
    ax.set_ylim(0.0 - PAD_ROC, 1.0 + PAD_ROC)
    ax.tick_params(axis='both', labelsize=FONTSIZE_TICKS)
    if c == 0:
        ax.set_xlabel("False Positive Rate", fontsize=FONTSIZE_LABELS)
        ax.set_ylabel("True Positive Rate", fontsize=FONTSIZE_LABELS)
        add_row_letter(ax, ROW_LETTERS[4])
    # Series 2 legend moves up to avoid the variance ROC step near the corner.
    legend_anchor = (0.96, 0.26) if c == 1 else (0.96, 0.06)
    ax.legend(frameon=False, fontsize=FONTSIZE_LEGEND_ROC, loc="lower right",
              bbox_to_anchor=legend_anchor)

os.makedirs(FIG_DIR, exist_ok=True)
PDF_PATH = os.path.join(FIG_DIR, f"SVM_{SURROGATE_METHOD}_weekly.pdf")
fig.savefig(PDF_PATH, bbox_inches="tight", transparent=False)
print(f"[OK] Figure saved: {PDF_PATH}")
for _r in results:
    print(f"AUC {_r['name']}: SDML={_r['auc_sdml']:.3f} Var={_r['auc_var']:.3f} AC1={_r['auc_ac1']:.3f}")
