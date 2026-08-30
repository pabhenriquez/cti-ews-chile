#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity of the SDML probability to the lowess detrending span.

For each series and each span in {0.10, 0.20, 0.30, 0.40, 0.50}, the full
SDML pipeline (surrogate generation, CNN ensemble training, causal
incremental prediction) is re-run and the ensemble-mean probability is
rendered as a time x span heatmap. Scoring starts once max(L, w)
observations have accrued, uniformly across spans. Long run (~150 CNN fits).

Usage:
    python src/make_lowess_heatmap.py
"""
import os
import pickle
import numpy as np

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")
FIG_DIR  = os.path.join(REPO_DIR, "figures")
OUT_DIR  = os.path.join(REPO_DIR, "output")
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import time

from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kendalltau
from numpy.fft import rfft, irfft

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import LayerNormalization

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, roc_auc_score

SURROGATE_METHOD = "AAFT"  # "IAAFT" | "FT" | "AAFT" | "RP"

LOWESS_FRAC_GRID = [0.10, 0.20, 0.30, 0.40, 0.50]

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

EWS_WINDOW_PRE_FRAC = 0.30
SMOOTH_EWS          = True
EWS_MOV_WIN         = 5

PRE_FRAC            = 0.50
L                  = 100
INC                = 3
SURR_N_PER_CLASS   = 800
N_MODELS           = 10
EPOCHS             = 100
BATCH_SIZE         = 64
PATIENCE           = 10

SMOOTH_SDML        = True
SDML_EMA_ALPHA     = 0.1
SDML_TRIANG_WIN    = 9

FONTSIZE_LABELS     = 13
FONTSIZE_TICKS      = 12

PDF_PATH = os.path.join(FIG_DIR, "LOWESS_heatmap_DLprob.pdf")

def gaussian_smooth(x, sigma):
    return gaussian_filter1d(np.asarray(x, float), sigma=float(sigma), mode="nearest") if sigma > 0 else np.asarray(x, float)

def lowess_smooth(x, frac):
    x = np.asarray(x, float)
    t = np.arange(len(x), dtype=float)
    return lowess(x, t, frac=float(frac), return_sorted=False)

def residuals_gaussian_full(x, sigma):
    return np.asarray(x, float) - gaussian_smooth(x, sigma)

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
        prev = s[k - 1]
        s[k] = alpha * y[k] + (1 - alpha) * prev
    return s

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

def build_multihead_cnn_binary(L=256):
    inp = layers.Input(shape=(L, 1))
    def head(k, f=32):
        h = layers.Conv1D(f, k, padding="same", activation="relu")(inp)
        h = LayerNormalization()(h)
        h = layers.Dropout(0.30)(h)
        h = layers.MaxPooling1D(2)(h)
        h = layers.Flatten()(h)
        return h
    h = layers.Concatenate()([head(3, 32), head(5, 32), head(11, 64), head(21, 64)])
    h = layers.Dense(128, activation="relu")(h)
    out = layers.Dense(1, activation="sigmoid")(h)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def sdml_predict_incremental_pre_fixed(t_num, resid_pre_all, L, models_list, inc, start_idx, stop_idx):
    means, stds, tt, ipos = [], [], [], []
    for i in range(max(L, start_idx) + 1, stop_idx + 1, inc):
        xr = np.asarray(resid_pre_all[:i], float)
        win = normalize_window(windows_lastL(xr, L))[None, :, None]
        probs = [float(m.predict(win, verbose=0)[0, 0]) for m in models_list]
        means.append(np.mean(probs))
        stds.append(np.std(probs))
        tt.append(t_num[i - 1])
        ipos.append(i - 1)
    return np.array(tt), np.array(ipos), np.array(means), np.array(stds)

# Exact-match filters; series truncated at START <= date <= TRANSITION.
def load_and_filter(cfg):
    df = pd.read_csv(cfg["DATA_PATH"])

    if cfg.get("METHOD_COL") and cfg.get("METHOD_FILTER") is not None:
        df = df[df[cfg["METHOD_COL"]] == cfg["METHOD_FILTER"]]

    df = df[df[cfg["TEMPORALIDAD_COL"]] == cfg["TEMPORALIDAD_FILTER"]]

    date_col = cfg["DATE_COL"]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    start_used = pd.Timestamp(cfg["START_DATE_USED"])
    trans_date = pd.Timestamp(cfg["TRANSITION_DATE"])
    df = df[(df[date_col] >= start_used) & (df[date_col] <= trans_date)].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"[{cfg['name']}] No data after filtering.")
    return df

def compute_dl_prob_curve(cfg, lowess_frac, surrogate_method=SURROGATE_METHOD):
    df = load_and_filter(cfg)

    date_col = cfg["DATE_COL"]
    value_col = cfg["VALUE_COL"]
    trans_date = pd.Timestamp(cfg["TRANSITION_DATE"])

    vals = df[value_col].values.astype(float)
    ok   = np.isfinite(vals)
    t_num = mdates.date2num(df[date_col].values.astype("datetime64[ns]")[ok])
    x     = vals[ok]
    n     = len(x)

    ix_trans = n - 1
    n_pre = n
    L_auto = max(64, min(500, int(0.8 * n_pre)))
    L_eff  = min(L, L_auto)

    if DETREND_METHOD.lower() == "gauss":
        resid_pre = residuals_gaussian_full(x[:n_pre], GAUSS_SIGMA)
    else:
        resid_pre = residuals_lowess_full(x[:n_pre], lowess_frac)

    w = max(10, int(EWS_WINDOW_PRE_FRAC * n_pre))
    # Uniform burn-in max(L, w) so every span row covers the same period.
    i_start = max(L_eff, w) - 1
    i_stop  = ix_trans
    i_start = min(max(i_start, 1), i_stop)

    x_neu_all, x_pre_all, cut_ix = split_neutral_pre_by_frac(resid_pre, PRE_FRAC)

    gen = get_surrogate_generator(surrogate_method)
    X_neu_s = gen(x_neu_all, n=SURR_N_PER_CLASS, rng=0)
    X_pre_s = gen(x_pre_all, n=SURR_N_PER_CLASS, rng=1)

    X_neu = np.stack([normalize_window(windows_lastL(s, L_eff)) for s in X_neu_s], axis=0)[..., None]
    X_pre = np.stack([normalize_window(windows_lastL(s, L_eff)) for s in X_pre_s], axis=0)[..., None]
    X_sdml = np.concatenate([X_neu, X_pre], axis=0)
    y_sdml = np.concatenate([np.zeros(len(X_neu)), np.ones(len(X_pre))]).astype(int)

    models_sdml = []
    f1_list = []

    for seed in range(N_MODELS):
        t0 = time.time()
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X_sdml, y_sdml, test_size=0.4, stratify=y_sdml, random_state=seed
        )
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed
        )

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)

        model = build_multihead_cnn_binary(L=L_eff)
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=PATIENCE, restore_best_weights=True
        )
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=[es]
        )

        y_prob = model.predict(X_te, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        f1_list.append(f1_score(y_te, y_pred))
        models_sdml.append(model)

        dt = time.time() - t0
        print(f"[{cfg['name']}] lowess={lowess_frac:.2f} | model {seed+1}/{N_MODELS} | {dt:.1f}s")

    print(f"[{cfg['name']}] lowess={lowess_frac:.2f} | F1={np.mean(f1_list):.3f} ± {np.std(f1_list):.3f}")

    tt, ipos, mu, sd = sdml_predict_incremental_pre_fixed(
        t_num=t_num, resid_pre_all=resid_pre, L=L_eff, models_list=models_sdml,
        inc=INC, start_idx=i_start, stop_idx=i_stop
    )

    mu_v = mu.copy()
    if SMOOTH_SDML:
        mu_v = ema_causal_nan(mu_v, alpha=SDML_EMA_ALPHA)
        if SDML_TRIANG_WIN and SDML_TRIANG_WIN > 0:
            mu_v = causal_triangular(mu_v, w=SDML_TRIANG_WIN)

    return dict(
        tt=np.asarray(tt, float),
        mu=np.asarray(mu_v, float),
        t_trans=float(mdates.date2num(trans_date))
    )

def edges_from_centers(x):
    x = np.asarray(x, float)
    if len(x) == 1:
        dx = 1.0
        return np.array([x[0] - dx/2, x[0] + dx/2], float)
    mid = (x[1:] + x[:-1]) / 2.0
    first = x[0] - (mid[0] - x[0])
    last = x[-1] + (x[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])

def align_curves_to_matrix(curves):
    # Prediction timestamps differ across rows; interpolate each curve onto
    # the common grid within its valid range to avoid interleaved gaps.
    all_tt = np.unique(np.round(np.concatenate([c["tt"] for c in curves]), 6))
    all_tt = np.sort(all_tt)
    F = len(curves)
    MU = np.full((F, len(all_tt)), np.nan, float)

    for r, c in enumerate(curves):
        tt = np.asarray(c["tt"], float)
        mu = np.asarray(c["mu"], float)
        inside = (all_tt >= tt.min() - 1e-9) & (all_tt <= tt.max() + 1e-9)
        MU[r, inside] = np.interp(all_tt[inside], tt, mu)

    return all_tt, MU

def run_all_series_heatmap(cfg_list, lowess_grid, surrogate_method=SURROGATE_METHOD):
    out = []
    for cfg in cfg_list:
        print(f"\n===== Processing {cfg['name']} =====")
        curves = []
        t_trans_ref = None
        for k, frac in enumerate(lowess_grid, 1):
            print(f"\n[{cfg['name']}] LOWESS {k}/{len(lowess_grid)}: frac={frac:.2f}")
            res = compute_dl_prob_curve(cfg, lowess_frac=frac, surrogate_method=surrogate_method)
            curves.append(res)
            t_trans_ref = res["t_trans"]
        tt_union, MU = align_curves_to_matrix(curves)
        out.append(dict(name=cfg["name"], tt=tt_union, MU=MU, t_trans=t_trans_ref))
    return out

def plot_heatmaps(agg_full, lowess_fracs, pdf_path=PDF_PATH):
    n = len(agg_full)
    fig, axes = plt.subplots(1, n, figsize=(18, 5), constrained_layout=True, sharey=True)
    if n == 1:
        axes = [axes]

    im_last = None
    y_centers = np.asarray(lowess_fracs, float)
    y_edges = edges_from_centers(y_centers)

    for ax, res in zip(axes, agg_full):
        tt = np.asarray(res["tt"], float)
        MU = np.asarray(res["MU"], float)
        # Crop to the time range covered by all rows.
        keep = np.isfinite(MU).all(axis=0)
        tt, MU = tt[keep], MU[:, keep]
        x_edges = edges_from_centers(tt)

        MU_mask = np.ma.masked_invalid(MU)
        im = ax.pcolormesh(
            x_edges, y_edges, MU_mask,
            shading="auto",
            vmin=0.0, vmax=1.0
        )
        im_last = im

        ax.axvline(res["t_trans"], ls="--", lw=1.0, color="0.4")
        ax.set_xlim(x_edges[0], res["t_trans"])
        ax.set_title(res["name"], fontsize=FONTSIZE_LABELS)
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis="both", labelsize=FONTSIZE_TICKS)
        ax.set_xlabel("Time", fontsize=FONTSIZE_LABELS)

    axes[0].set_ylabel("Lowess", fontsize=FONTSIZE_LABELS)
    cbar = fig.colorbar(im_last, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("DL probability", fontsize=FONTSIZE_LABELS)

    fig.savefig(pdf_path, bbox_inches="tight", transparent=False)
    
    print(f"[OK] Saved: {pdf_path}")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
agg = run_all_series_heatmap(CFG_LIST, LOWESS_FRAC_GRID, surrogate_method=SURROGATE_METHOD)
with open(os.path.join(OUT_DIR, "results_lowess_heatmap.pkl"), "wb") as f:
    pickle.dump(dict(agg=agg, lowess_grid=LOWESS_FRAC_GRID), f)
plot_heatmaps(agg, LOWESS_FRAC_GRID, pdf_path=PDF_PATH)
