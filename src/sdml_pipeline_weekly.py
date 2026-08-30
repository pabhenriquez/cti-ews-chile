#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDML pipeline for the weekly Critical Transition Index (CTI) series.

Trains an ensemble of multi-head CNNs on surrogate data generated from the
neutral / pre-transition halves of each series, computes incremental
transition probabilities, classical early-warning indicators (rolling
variance and lag-1 autocorrelation), and pseudo-ROC curves, and renders the
five-row figure panel.

Usage:
    python src/sdml_pipeline_weekly.py [--method AAFT|FT|IAAFT|RP] [--plot-only]
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata, kendalltau
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════
SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_DIR  = os.path.dirname(SRC_DIR)
DATA_DIR  = os.path.join(REPO_DIR, "data")
FIG_DIR   = os.path.join(REPO_DIR, "figures")
OUT_DIR   = os.path.join(REPO_DIR, "output")

CFG_LIST = [
    dict(
        name="Series 1",
        DATA_PATH=os.path.join(DATA_DIR, "serie1.csv"),
        DATE_COL="fecha",  VALUE_COL="indice_cti",
        METHOD_COL=None,   METHOD_FILTER=None,
        TEMPORALIDAD_COL="Temporalidad", TEMPORALIDAD_FILTER="Semanal",
        START_DATE="2014-01-01",  TRANSITION_DATE="2020-03-31",
    ),
    dict(
        name="Series 2",
        DATA_PATH=os.path.join(DATA_DIR, "serie2.csv"),
        DATE_COL="time",   VALUE_COL="Value",
        METHOD_COL="Method", METHOD_FILTER="Original",
        TEMPORALIDAD_COL="Frequency", TEMPORALIDAD_FILTER="Weekly",
        START_DATE="2014-01-01",  TRANSITION_DATE="2020-03-31",
    ),
    dict(
        name="Series 3",
        DATA_PATH=os.path.join(DATA_DIR, "serie3.csv"),
        DATE_COL="time",   VALUE_COL="Value",
        METHOD_COL="Method", METHOD_FILTER="Original",
        TEMPORALIDAD_COL="Frequency", TEMPORALIDAD_FILTER="Weekly",
        START_DATE="2014-01-01",  TRANSITION_DATE="2021-12-31",
    ),
]

SURROGATE_METHOD = "AAFT"
for _i, _a in enumerate(sys.argv):
    if _a == "--method" and _i + 1 < len(sys.argv):
        SURROGATE_METHOD = sys.argv[_i + 1].upper()
        break
SURR_N_PER_CLASS = 800
PRE_FRAC         = 0.50     # neutral / pre-transition label split
L                = 100      # classifier input window length
INC              = 3        # incremental prediction step
N_MODELS         = 10       # ensemble size

LOWESS_FRAC = 0.40          # detrending span

EPOCHS     = 100
BATCH_SIZE = 64
PATIENCE   = 10
ES_MONITOR = "val_loss"

EWS_WINDOW_FRAC = 0.30      # rolling window for variance / lag-1 AC
SMOOTH_EWS      = True
EWS_MOV_WIN     = 5

SMOOTH_SDML     = True
SDML_EMA_ALPHA  = 0.1
SDML_TRIANG_WIN = 9

FONTSIZE_LABELS = 13
FONTSIZE_TICKS  = 12
FONTSIZE_LEGEND = 11
COLOR_GREY  = "0.40"
COLOR_BLACK = "0.05"
ROW_LETTERS = ["A", "B", "C", "D", "E"]
PAD_ROC     = 0.02


# ════════════════════════════════════════════════════════════════
# SURROGATE GENERATION
# ════════════════════════════════════════════════════════════════

def surrogate_rp(x, rng):
    return rng.permutation(x).copy()

def surrogate_ft(x, rng):
    n  = len(x)
    ft = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, len(ft))
    ph[0] = 0
    if n % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(ft * np.exp(1j * ph), n=n)

def surrogate_aaft(x, rng):
    n = len(x)
    sorted_x = np.sort(x)
    ranks    = rankdata(x, method="ordinal").astype(int) - 1
    g_sorted = np.sort(rng.standard_normal(n))
    x_gauss  = g_sorted[ranks]
    ft = np.fft.rfft(x_gauss)
    ph = rng.uniform(0, 2 * np.pi, len(ft))
    ph[0] = 0
    if n % 2 == 0:
        ph[-1] = 0
    gs = np.fft.irfft(ft * np.exp(1j * ph), n=n)
    return sorted_x[rankdata(gs, method="ordinal").astype(int) - 1]

def surrogate_iaaft(x, rng, n_iter=200):
    n = len(x)
    sorted_x = np.sort(x)
    mag      = np.abs(np.fft.rfft(x))
    surr     = rng.permutation(x).copy()
    for _ in range(n_iter):
        ft   = np.fft.rfft(surr)
        surr = np.fft.irfft(mag * np.exp(1j * np.angle(ft)), n=n)
        surr = sorted_x[rankdata(surr, method="ordinal").astype(int) - 1]
    return surr

_SURR_FN = {
    "RP":    surrogate_rp,
    "FT":    surrogate_ft,
    "AAFT":  surrogate_aaft,
    "IAAFT": surrogate_iaaft,
}

def generate_surrogates(x, n_surr, method="AAFT", seed=0):
    rng = np.random.default_rng(seed)
    fn  = _SURR_FN[method]
    return np.array([fn(x, rng) for _ in range(n_surr)])


# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

def load_series(cfg):
    """Weekly values from START_DATE up to and including TRANSITION_DATE."""
    df = pd.read_csv(cfg["DATA_PATH"])
    if cfg["METHOD_COL"] and cfg["METHOD_FILTER"]:
        df = df[df[cfg["METHOD_COL"]] == cfg["METHOD_FILTER"]]
    df = df[df[cfg["TEMPORALIDAD_COL"]] == cfg["TEMPORALIDAD_FILTER"]]
    df[cfg["DATE_COL"]] = pd.to_datetime(df[cfg["DATE_COL"]])
    df = df.sort_values(cfg["DATE_COL"]).reset_index(drop=True)
    start = pd.Timestamp(cfg["START_DATE"])
    trans = pd.Timestamp(cfg["TRANSITION_DATE"])
    df = df[(df[cfg["DATE_COL"]] >= start) &
            (df[cfg["DATE_COL"]] <= trans)].reset_index(drop=True)
    dates = df[cfg["DATE_COL"]].values
    vals  = df[cfg["VALUE_COL"]].values.astype(float)
    ok = np.isfinite(vals)
    return dates[ok], vals[ok]


# ════════════════════════════════════════════════════════════════
# DETRENDING & INDICATORS
# ════════════════════════════════════════════════════════════════

def lowess_smooth(x, frac):
    return sm_lowess(x, np.arange(len(x)), frac=frac, return_sorted=False)

def rolling_var_ac1(resid, w):
    n   = len(resid)
    var = np.full(n, np.nan)
    ac1 = np.full(n, np.nan)
    for i in range(w - 1, n):
        seg = resid[i - w + 1 : i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) < 2:
            continue
        var[i] = np.var(seg, ddof=1)
        c = seg - np.mean(seg)
        d = np.dot(c, c)
        ac1[i] = np.dot(c[1:], c[:-1]) / d if d > 0 else np.nan
    return var, ac1

def movmean_causal(y, win=5):
    return pd.Series(y).rolling(win, min_periods=1).mean().to_numpy()

def ema_causal(y, alpha=0.25):
    y = np.asarray(y, float)
    s = np.full_like(y, np.nan)
    idx = np.where(np.isfinite(y))[0]
    if len(idx) == 0:
        return s
    s[idx[0]] = y[idx[0]]
    for k in idx[1:]:
        s[k] = alpha * y[k] + (1 - alpha) * s[k - 1]
    return s

def causal_triangular(y, w=5):
    if w < 2:
        return np.asarray(y, float)
    mm = pd.Series(y).rolling(w, min_periods=1).mean().to_numpy()
    return pd.Series(mm).rolling(w, min_periods=1).mean().to_numpy()


# ════════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════════

def build_multihead_cnn(input_len):
    inp = layers.Input(shape=(input_len, 1))

    def head(k, f=32):
        h = layers.Conv1D(f, k, padding="same", activation="relu")(inp)
        h = LayerNormalization()(h)
        h = layers.Dropout(0.30)(h)
        h = layers.MaxPooling1D(2)(h)
        h = layers.Flatten()(h)
        return h

    merged = layers.Concatenate()(
        [head(3, 32), head(5, 32), head(11, 64), head(21, 64)]
    )
    h   = layers.Dense(128, activation="relu")(merged)
    out = layers.Dense(1, activation="sigmoid")(h)
    m   = models.Model(inp, out)
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


# ════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════

def normalize_window(wv, eps=1e-3):
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

def extract_windows_lastL(surrogates, target_L, normalize=True):
    out = np.zeros((len(surrogates), target_L))
    for i, s in enumerate(surrogates):
        out[i] = windows_lastL(s, target_L)
        if normalize:
            out[i] = normalize_window(out[i])
    return out

def train_ensemble(X, y, input_len, n_models, model_dir):
    es_mode = "min" if "loss" in ES_MONITOR else "max"
    trained = []
    for seed in range(n_models):
        print(f"    Model {seed+1}/{n_models} ...", end=" ", flush=True)
        tf.keras.utils.set_random_seed(seed)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.40, random_state=seed
        )
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_tr, y_tr, test_size=0.50, random_state=seed
        )
        model = build_multihead_cnn(input_len)
        fpath = os.path.join(model_dir, f"model_{seed}.keras")
        ckpt = ModelCheckpoint(
            fpath, monitor=ES_MONITOR, save_best_only=True, mode=es_mode, verbose=0
        )
        es = EarlyStopping(
            monitor=ES_MONITOR, patience=PATIENCE, mode=es_mode,
            restore_best_weights=True, verbose=0,
        )
        model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=[ckpt, es], verbose=0,
        )
        best = tf.keras.models.load_model(fpath)
        _, acc = best.evaluate(X_te, y_te, verbose=0)
        print(f"test_acc={acc:.3f}")
        trained.append(best)
    return trained


# ════════════════════════════════════════════════════════════════
# PREDICTION
# ════════════════════════════════════════════════════════════════

def sdml_predict_incremental(resid, L_eff, trained_models, inc, start_idx, stop_idx):
    """Causal expanding prediction: score the last L_eff residuals every inc steps."""
    positions, means, stds = [], [], []
    for i in range(max(L_eff, start_idx) + 1, stop_idx + 1, inc):
        xr  = np.asarray(resid[:i], float)
        win = normalize_window(windows_lastL(xr, L_eff))
        win_3d = win[np.newaxis, :, np.newaxis]
        probs = [float(m.predict(win_3d, verbose=0)[0, 0]) for m in trained_models]
        positions.append(i - 1)
        means.append(np.mean(probs))
        stds.append(np.std(probs))
    return np.array(positions), np.array(means), np.array(stds)


# ════════════════════════════════════════════════════════════════
# ROC
# ════════════════════════════════════════════════════════════════

def roc_safe(y_true, scores):
    y_true = np.asarray(y_true, int)
    scores = np.asarray(scores, float)
    if len(np.unique(y_true)) < 2:
        return np.array([0, 1]), np.array([0, 1]), np.nan
    fpr, tpr, _ = roc_curve(y_true, scores)
    return fpr, tpr, roc_auc_score(y_true, scores)


# ════════════════════════════════════════════════════════════════
# PROCESS ONE SERIES
# ════════════════════════════════════════════════════════════════

def process_series(cfg, model_dir):
    name = cfg["name"]
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    dates, x_raw = load_series(cfg)
    n_pre = len(x_raw)
    t_num   = mdates.date2num(pd.to_datetime(dates))
    t_trans = mdates.date2num(pd.Timestamp(cfg["TRANSITION_DATE"]))
    print(f"  Loaded: {n_pre} points")

    smooth = lowess_smooth(x_raw, LOWESS_FRAC)
    resid  = x_raw - smooth

    L_auto = max(128, min(500, int(0.8 * n_pre)))
    L_eff  = min(L, L_auto)
    print(f"  L={L}, L_auto={L_auto}, L_eff={L_eff}")

    cut_ix     = int(round((1.0 - PRE_FRAC) * n_pre))
    r_neutral  = resid[:cut_ix]
    r_pretrans = resid[cut_ix:]
    print(f"  Split: neutral={len(r_neutral)}, pre-trans={len(r_pretrans)}")

    # Burn-in covers the input window, the indicator window, and the left
    # boundary region of the lowess fit.
    w = max(10, int(EWS_WINDOW_FRAC * n_pre))
    WARMUP_EXTRA = max(20, int(LOWESS_FRAC * n_pre))
    i_start = max(L_eff, w, WARMUP_EXTRA) - 1
    i_stop  = n_pre - 1
    i_start = min(max(i_start, 1), i_stop)
    print(f"  w={w}, WARMUP={WARMUP_EXTRA}, i_start={i_start}, i_stop={i_stop}")

    print(f"  Generating {SURR_N_PER_CLASS} {SURROGATE_METHOD} surrogates/class ...")
    t0 = time.time()
    surr_neu = generate_surrogates(r_neutral,  SURR_N_PER_CLASS, SURROGATE_METHOD, seed=0)
    surr_pre = generate_surrogates(r_pretrans, SURR_N_PER_CLASS, SURROGATE_METHOD, seed=1)
    print(f"  Done in {time.time() - t0:.1f}s  "
          f"(shapes: {surr_neu.shape}, {surr_pre.shape})")

    X_neu = extract_windows_lastL(surr_neu, L_eff, normalize=True)
    X_pre = extract_windows_lastL(surr_pre, L_eff, normalize=True)
    X_all = np.concatenate([X_neu, X_pre], axis=0)
    y_all = np.concatenate([np.zeros(len(X_neu)), np.ones(len(X_pre))])
    X_all = X_all[:, :, np.newaxis]
    print(f"  Training data: {X_all.shape}, balance: {y_all.mean():.2f}")

    print(f"  Training {N_MODELS} CNN models (patience={PATIENCE}) ...")
    trained = train_ensemble(X_all, y_all, L_eff, N_MODELS, model_dir)

    print(f"  Expanding window prediction (L_eff={L_eff}, INC={INC}, "
          f"i_start={i_start}, i_stop={i_stop}) ...")
    pos, mu, sd = sdml_predict_incremental(resid, L_eff, trained, INC, i_start, i_stop)

    mu_v = mu.copy()
    sd_v = sd.copy()
    if SMOOTH_SDML:
        mu_v = ema_causal(mu_v, SDML_EMA_ALPHA)
        if SDML_TRIANG_WIN > 0:
            mu_v = causal_triangular(mu_v, SDML_TRIANG_WIN)
        sd_v = ema_causal(sd_v, SDML_EMA_ALPHA)

    var_r, ac1_r = rolling_var_ac1(resid, w)
    var_plot = movmean_causal(var_r, EWS_MOV_WIN) if SMOOTH_EWS else var_r
    ac1_plot = movmean_causal(ac1_r, EWS_MOV_WIN) if SMOOTH_EWS else ac1_r

    mask    = np.arange(n_pre) >= w - 1
    idx_tau = np.where(mask & np.isfinite(var_r) & np.isfinite(ac1_r))[0]
    tau_var = tau_ac1 = np.nan
    if len(idx_tau) > 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tau_var = kendalltau(idx_tau.astype(float), var_r[idx_tau]).statistic
            tau_ac1 = kendalltau(idx_tau.astype(float), ac1_r[idx_tau]).statistic

    # Pseudo-ROC: labels from the 50/50 split, unsmoothed scores.
    bnd_s = int(np.searchsorted(pos, cut_ix, side="left"))
    bnd_s = max(1, min(len(pos) - 1, bnd_s))
    yt_s  = np.zeros(len(pos), int);  yt_s[bnd_s:] = 1
    fpr_s, tpr_s, auc_s = roc_safe(yt_s, mu)

    valid = np.where(np.isfinite(var_r) & np.isfinite(ac1_r))[0]
    bnd_e = int(np.searchsorted(valid, cut_ix, side="left"))
    bnd_e = max(1, min(len(valid) - 1, bnd_e))
    yt_e  = np.zeros(len(valid), int);  yt_e[bnd_e:] = 1
    fpr_v, tpr_v, auc_v = roc_safe(yt_e, var_r[valid])
    fpr_a, tpr_a, auc_a = roc_safe(yt_e, ac1_r[valid])

    print(f"  AUC  SDML={auc_s:.3f}  Var={auc_v:.3f}  AC1={auc_a:.3f}")
    print(f"  Tau  Var={tau_var:.3f}  AC1={tau_ac1:.3f}")

    return dict(
        name=name, t_num=t_num, x=x_raw, x_smooth=smooth,
        t_trans_num=t_trans,
        ac1_plot=ac1_plot, var_plot=var_plot,
        tau_ac1=tau_ac1, tau_var=tau_var,
        pos=pos, mu_raw=mu, sd_raw=sd, mu_v=mu_v, sd_v=sd_v,
        fpr_sdml=fpr_s, tpr_sdml=tpr_s, auc_sdml=auc_s,
        fpr_var=fpr_v, tpr_var=tpr_v, auc_var=auc_v,
        fpr_ac1=fpr_a, tpr_ac1=tpr_a, auc_ac1=auc_a,
        surrogate_method=SURROGATE_METHOD,
    )


# ════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════

def plot_panel(results, output_path):
    fig, axes = plt.subplots(5, 3, figsize=(18, 16), constrained_layout=True)

    def clip_roc(fpr, tpr):
        return (np.clip(fpr, PAD_ROC, 1 - PAD_ROC),
                np.clip(tpr, PAD_ROC, 1 - PAD_ROC))

    def add_letter(ax, letter):
        ax.text(-0.25, 0.5, letter, transform=ax.transAxes,
                ha="right", va="center",
                fontsize=FONTSIZE_LABELS + 3, fontweight="bold", clip_on=False)

    tmin_fixed = mdates.date2num(pd.Timestamp("2014-01-01"))
    tmax_fixed = mdates.date2num(pd.Timestamp("2024-01-01"))

    for c, res in enumerate(results):
        t    = res["t_num"]
        tmin = tmin_fixed
        tmax = tmax_fixed
        tt   = t[res["pos"]]

        ax = axes[0, c]
        ax.plot(t, res["x"],        color=COLOR_GREY,  lw=1.0)
        ax.plot(t, res["x_smooth"], color=COLOR_BLACK, lw=1.5)
        ax.axvline(res["t_trans_num"], ls="--", color=COLOR_GREY, lw=1.0)
        ax.set_xlim(tmin, tmax);  ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(labelsize=FONTSIZE_TICKS)
        ax.set_title(res["name"], fontsize=FONTSIZE_LABELS)
        if c == 0:
            ax.set_ylabel("Critical transition index", fontsize=FONTSIZE_LABELS)
            add_letter(ax, ROW_LETTERS[0])

        ax = axes[1, c]
        ax.plot(t, res["ac1_plot"], color=COLOR_GREY, lw=1.0)
        ax.axvline(res["t_trans_num"], ls="--", color=COLOR_GREY, lw=1.0)
        ax.set_xlim(tmin, tmax);  ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(labelsize=FONTSIZE_TICKS)
        if np.isfinite(res["tau_ac1"]):
            ax.text(0.98, 0.05, rf"$\tau$={res['tau_ac1']:.2f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=FONTSIZE_TICKS, color="black")
        if c == 0:
            ax.set_ylabel("Lag-1 AC", fontsize=FONTSIZE_LABELS)
            add_letter(ax, ROW_LETTERS[1])

        ax = axes[2, c]
        ax.plot(t, res["var_plot"], color=COLOR_GREY, lw=1.0)
        ax.axvline(res["t_trans_num"], ls="--", color=COLOR_GREY, lw=1.0)
        ax.set_xlim(tmin, tmax);  ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(labelsize=FONTSIZE_TICKS)
        if np.isfinite(res["tau_var"]):
            ax.text(0.98, 0.05, rf"$\tau$={res['tau_var']:.2f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=FONTSIZE_TICKS, color="black")
        if c == 0:
            ax.set_ylabel("Variance", fontsize=FONTSIZE_LABELS)
            add_letter(ax, ROW_LETTERS[2])

        ax = axes[3, c]
        ax.plot(tt, res["mu_v"], color=COLOR_GREY, lw=1.0)
        ax.fill_between(
            tt,
            np.clip(res["mu_v"] - res["sd_v"], 0, 1),
            np.clip(res["mu_v"] + res["sd_v"], 0, 1),
            color=COLOR_GREY, alpha=0.20,
        )
        ax.axvline(res["t_trans_num"], ls="--", color=COLOR_GREY, lw=1.0)
        ax.set_ylim(0, 1);  ax.set_xlim(tmin, tmax);  ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(labelsize=FONTSIZE_TICKS)
        if c == 0:
            ax.set_ylabel("DL probability", fontsize=FONTSIZE_LABELS)
            add_letter(ax, ROW_LETTERS[3])
        ax.set_xlabel("Time", fontsize=FONTSIZE_LABELS)

        ax = axes[4, c]
        f, tp = clip_roc(res["fpr_ac1"], res["tpr_ac1"])
        ax.plot(f, tp, lw=1.8, ls="--",
                label=f"AC(1) (AUC={res['auc_ac1']:.3f})")
        f, tp = clip_roc(res["fpr_var"], res["tpr_var"])
        ax.plot(f, tp, lw=1.8, ls="-.",
                label=f"Variance (AUC={res['auc_var']:.3f})")
        f, tp = clip_roc(res["fpr_sdml"], res["tpr_sdml"])
        ax.plot(f, tp, lw=1.8,
                label=f"SDML-{res['surrogate_method']} (AUC={res['auc_sdml']:.3f})")
        d = np.linspace(PAD_ROC, 1 - PAD_ROC, 200)
        ax.plot(d, d, ls=":", lw=1.0, color="0.6")
        ax.set_xlim(-PAD_ROC, 1 + PAD_ROC)
        ax.set_ylim(-PAD_ROC, 1 + PAD_ROC)
        ax.tick_params(labelsize=FONTSIZE_TICKS)
        # Series 2 (c==1): the variance ROC has a step near the lower-right
        # corner, so the legend moves to the free mid band.
        anchor = (0.96, 0.26) if c == 1 else (0.96, 0.06)
        ax.legend(frameon=False, fontsize=FONTSIZE_LEGEND, loc="lower right",
                  bbox_to_anchor=anchor)
        if c == 0:
            ax.set_xlabel("False Positive Rate", fontsize=FONTSIZE_LABELS)
            ax.set_ylabel("True Positive Rate", fontsize=FONTSIZE_LABELS)
            add_letter(ax, ROW_LETTERS[4])

    fig.savefig(output_path, bbox_inches="tight", transparent=False)
    print(f"\nFigure saved -> {output_path}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pickle

    t_global = time.time()

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    output_pdf  = os.path.join(FIG_DIR, f"CNN_{SURROGATE_METHOD}_weekly.pdf")
    results_pkl = os.path.join(OUT_DIR, f"results_{SURROGATE_METHOD}_weekly.pkl")

    plot_only = "--plot-only" in sys.argv

    if plot_only and os.path.exists(results_pkl):
        print("Loading cached results ...")
        with open(results_pkl, "rb") as f:
            results = pickle.load(f)
        for res in results:
            if "mu_raw" in res:
                mu_v = res["mu_raw"].copy()
                sd_v = res["sd_raw"].copy()
                if SMOOTH_SDML:
                    mu_v = ema_causal(mu_v, SDML_EMA_ALPHA)
                    if SDML_TRIANG_WIN > 0:
                        mu_v = causal_triangular(mu_v, SDML_TRIANG_WIN)
                    sd_v = ema_causal(sd_v, SDML_EMA_ALPHA)
                res["mu_v"] = mu_v
                res["sd_v"] = sd_v
    else:
        results = []
        for s_i, cfg in enumerate(CFG_LIST, start=1):
            model_dir = os.path.join(
                OUT_DIR, f"models_weekly_{SURROGATE_METHOD}", f"S{s_i}")
            os.makedirs(model_dir, exist_ok=True)
            results.append(process_series(cfg, model_dir))
        with open(results_pkl, "wb") as f:
            pickle.dump(results, f)

    plot_panel(results, output_pdf)

    print(f"\nTotal time: {time.time() - t_global:.0f}s")
