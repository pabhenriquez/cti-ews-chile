#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grad-CAM interpretation of the SDML classifier on the daily CTI series.

For each surrogate family (IAAFT, AAFT, FT, RP), an ensemble of ten
multi-head CNNs is trained on surrogate windows and a 1D Grad-CAM map is
computed on the last causal window ending at the transition, averaged over
convolutional branches and ensemble members, and overlaid on the residual
signal. Gradients are taken with respect to the sigmoid output (the
predicted probability of the pre-transition class).

Usage:
    python src/make_gradcam.py [--series 1|2|3]   # default: all three

Long run: 4 families x 10 CNN fits per series on daily data.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.ndimage import gaussian_filter1d
from numpy.fft import rfft, irfft
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import LayerNormalization

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")
FIG_DIR  = os.path.join(REPO_DIR, "figures")

CFG_LIST = [
    dict(
        SERIES=1,
        DATA_CSV=os.path.join(DATA_DIR, "serie1.csv"),
        DATE_COL="fecha", VALUE_COL="indice_cti",
        METHOD_COL=None, METHOD_FILTER=None,
        TEMP_COL="Temporalidad", TEMP_FILTER="Diario",
        TRANSITION_DATE="2020-03-31",
    ),
    dict(
        SERIES=2,
        DATA_CSV=os.path.join(DATA_DIR, "serie2.csv"),
        DATE_COL="time", VALUE_COL="Value",
        METHOD_COL="Method", METHOD_FILTER="Original",
        TEMP_COL="Frequency", TEMP_FILTER="Daily",
        TRANSITION_DATE="2020-03-31",
    ),
    dict(
        SERIES=3,
        DATA_CSV=os.path.join(DATA_DIR, "serie3.csv"),
        DATE_COL="time", VALUE_COL="Value",
        METHOD_COL="Method", METHOD_FILTER="Original",
        TEMP_COL="Frequency", TEMP_FILTER="Daily",
        TRANSITION_DATE="2021-12-31",
    ),
]

START_DATE_USED   = "2014-01-01"
DETREND_METHOD    = "lowess"
GAUSS_SIGMA       = 56
LOWESS_FRAC       = 0.40
PRE_FRAC          = 0.50

L                  = 256
SURR_METHODS       = ["IAAFT", "AAFT", "FT", "RP"]
SURR_N_PER_CLASS   = 800
N_MODELS           = 10
EPOCHS             = 80
BATCH_SIZE         = 64
PATIENCE           = 10

COLOR_SIGNAL       = "0.35"
CMAP_HEAT          = "Reds"
POST_MARGIN_DAYS   = 21
TICK_MONTH_INTERVAL= 1
FONTSIZE_LABELS    = 12
FONTSIZE_TICKS     = 11
XROT_DEG           = 45


def lowess_smooth(x, frac):
    x = np.asarray(x, float); t = np.arange(len(x), dtype=float)
    return lowess(x, t, frac=float(frac), it=0, return_sorted=False)

def gaussian_smooth(x, sigma):
    return gaussian_filter1d(np.asarray(x, float), sigma=float(sigma), mode="nearest") if sigma > 0 else np.asarray(x, float)

def residuals_lowess_full(x, frac):
    return np.asarray(x, float) - lowess_smooth(x, frac)

def residuals_gaussian_full(x, s):
    return np.asarray(x, float) - gaussian_smooth(x, s)

def normalize_window(wv, eps=1e-3):
    wv = np.asarray(wv, float)
    m = np.nanmean(np.abs(wv)); s = np.nanstd(wv)
    return wv / max(m, 0.5*s, eps)

def windows_lastL(arr, L_):
    arr = np.asarray(arr, float)
    if len(arr) >= L_:
        return arr[-L_:]
    out = np.zeros(L_, float)
    out[-len(arr):] = arr
    return out

def split_neutral_pre_by_frac(rpre, pre_frac):
    rpre = np.asarray(rpre, float)
    cut  = int(round((1.0 - pre_frac) * len(rpre)))
    cut  = min(max(cut, 1), len(rpre)-1)
    return rpre[:cut].copy(), rpre[cut:].copy(), cut

def surrogate_rp(x, n=1, rng=None):
    rng = np.random.default_rng(rng); x = np.asarray(x, float)
    return np.array([rng.permutation(x) for _ in range(n)])

def surrogate_ft(x, n=1, rng=None):
    rng = np.random.default_rng(rng); x = np.asarray(x, float); N = len(x)
    mag = np.abs(rfft(x)); X = []
    for _ in range(n):
        phase = rng.uniform(0, 2*np.pi, mag.shape)
        phase[0] = 0.0
        if N % 2 == 0:
            phase[-1] = 0.0
        y = irfft(mag * np.exp(1j*phase), n=N)
        X.append(y)
    return np.array(X)

def erfinv(y):
    a = 0.147; y = np.asarray(y, float)
    ln = np.log(1 - y**2)
    return np.sign(y) * np.sqrt(np.sqrt((2/(np.pi*a) + ln/2.0)**2 - ln/a) - (2/(np.pi*a) + ln/2.0))

def surrogate_aaft(x, n=1, rng=None):
    rng = np.random.default_rng(rng); x = np.asarray(x, float); X = []
    xr = pd.Series(x).rank(method="average").to_numpy() / (len(x)+1.0)
    z  = np.sqrt(2.0) * erfinv(2*xr - 1.0)
    for _ in range(n):
        y = surrogate_ft(z, n=1, rng=rng)[0]
        r = pd.Series(y).rank(method="first").astype(int).to_numpy() - 1
        xs = np.sort(x)
        X.append(xs[np.clip(r, 0, len(xs)-1)])
    return np.array(X)

def surrogate_iaaft(x, n=1, iters=200, rng=None):
    rng = np.random.default_rng(rng); x = np.asarray(x, float); N = len(x)
    Xf_mag = np.abs(rfft(x)); xs = np.sort(x); X = []
    for _ in range(n):
        y = rng.permutation(x)
        for _ in range(iters):
            Yf = rfft(y)
            y  = irfft(Xf_mag * np.exp(1j*np.angle(Yf)), n=N)
            r  = pd.Series(y).rank(method="first").astype(int).to_numpy() - 1
            y  = xs[np.clip(r, 0, N-1)]
        X.append(y)
    return np.array(X)

def get_surrogate_generator(name):
    m = name.upper()
    if m == "IAAFT":
        return lambda x, n, rng=None: surrogate_iaaft(x, n=n, iters=200, rng=rng)
    if m == "FT":
        return surrogate_ft
    if m == "AAFT":
        return surrogate_aaft
    if m == "RP":
        return surrogate_rp
    raise ValueError("SURR_METHOD must be 'IAAFT', 'FT', 'AAFT' or 'RP'.")

def build_multihead_cnn_binary(L=256):
    inp = layers.Input(shape=(L,1))
    def head(kernel, filters, name_prefix):
        h = layers.Conv1D(filters, kernel, padding="same", activation="relu", name=f"{name_prefix}_conv")(inp)
        h = LayerNormalization(name=f"{name_prefix}_ln")(h)
        h = layers.Dropout(0.30, name=f"{name_prefix}_drop")(h)
        h = layers.MaxPooling1D(2, name=f"{name_prefix}_pool")(h)
        h = layers.Flatten(name=f"{name_prefix}_flat")(h)
        return h
    h3  = head(3,  32, "h3")
    h5  = head(5,  32, "h5")
    h11 = head(11, 64, "h11")
    h21 = head(21, 64, "h21")
    h = layers.Concatenate(name="concat")([h3, h5, h11, h21])
    h = layers.Dense(128, activation="relu", name="dense")(h)
    out = layers.Dense(1, activation="sigmoid", name="out")(h)
    model = models.Model(inp, out, name="sdml_multikernel_cnn")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

CAM_LAYER_NAMES = ["h3_conv", "h5_conv", "h11_conv", "h21_conv"]

def _normalize_cam(cam, eps=1e-8):
    cam = np.maximum(cam, 0.0)
    m, M = cam.min(), cam.max()
    return (cam - m) / (M - m + eps) if np.isfinite(M) and (M - m) > 0 else np.zeros_like(cam)

def grad_cam_1d(model, x_win_1d, layer_names=CAM_LAYER_NAMES, class_index=0):
    x_tf = tf.convert_to_tensor(x_win_1d[None, :, None], dtype=tf.float32)
    conv_layers = [model.get_layer(n).output for n in layer_names]
    cam_model = tf.keras.Model(model.inputs, conv_layers + [model.output])
    with tf.GradientTape() as tape:
        outs = cam_model(x_tf, training=False)
        conv_acts = outs[:-1]
        y_pred = outs[-1][:, class_index]
    grads = tape.gradient(y_pred, conv_acts)
    cams = []
    for A, G in zip(conv_acts, grads):
        weights = tf.reduce_mean(G, axis=1)
        cam = tf.nn.relu(tf.reduce_sum(A * weights[:, None, :], -1))
        cams.append(cam[0].numpy())
    cam_mean = np.mean(np.stack(cams, axis=0), axis=0)
    return _normalize_cam(cam_mean)

def grad_cam_ensemble(models_list, x_win_1d, layer_names=CAM_LAYER_NAMES, class_index=0):
    cams = [grad_cam_1d(m, x_win_1d, layer_names, class_index) for m in models_list]
    return _normalize_cam(np.mean(np.stack(cams, axis=0), axis=0))


def run_series(cfg):
    df = pd.read_csv(cfg["DATA_CSV"])
    DATE_COL, VALUE_COL = cfg["DATE_COL"], cfg["VALUE_COL"]

    if cfg["METHOD_COL"] is not None and cfg["METHOD_FILTER"] is not None:
        if cfg["METHOD_COL"] not in df.columns:
            raise ValueError(f"Missing column '{cfg['METHOD_COL']}' in the CSV.")
        df = df[df[cfg["METHOD_COL"]].astype(str).str.strip().str.lower()
                == str(cfg["METHOD_FILTER"]).strip().lower()].copy()

    if cfg["TEMP_COL"] is not None and cfg["TEMP_FILTER"] is not None:
        if cfg["TEMP_COL"] not in df.columns:
            raise ValueError(f"Missing column '{cfg['TEMP_COL']}' in the CSV.")
        df = df[df[cfg["TEMP_COL"]].astype(str).str.strip().str.lower()
                == str(cfg["TEMP_FILTER"]).strip().lower()].copy()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, VALUE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    if df.empty:
        raise ValueError("Empty dataframe after filters.")

    if df[DATE_COL].duplicated().any():
        df = (df.groupby(DATE_COL, as_index=False)[VALUE_COL]
                .mean()
                .sort_values(DATE_COL)
                .reset_index(drop=True))

    t_num_full = mdates.date2num(df[DATE_COL].values.astype("datetime64[ns]"))
    x_full     = df[VALUE_COL].astype(float).to_numpy()

    start_used = pd.Timestamp(START_DATE_USED)
    trans_date = pd.Timestamp(cfg["TRANSITION_DATE"])
    idx        = pd.DatetimeIndex(df[DATE_COL])

    if not (df[DATE_COL].iloc[0] <= trans_date <= df[DATE_COL].iloc[-1]):
        raise ValueError("TRANSITION_DATE out of range.")

    ix_start = int(idx.get_indexer([start_used], method="nearest")[0])
    ix_trans = int(idx.get_indexer([trans_date],  method="nearest")[0])

    t_num = t_num_full[ix_start:]
    x     = x_full[ix_start:]
    n     = len(x)

    ix_trans = ix_trans - ix_start
    if ix_trans <= 0 or ix_trans >= n:
        raise ValueError("Transition outside the segment used.")
    n_pre = ix_trans + 1

    if DETREND_METHOD.lower() == "gauss":
        resid_pre = residuals_gaussian_full(x[:n_pre], GAUSS_SIGMA)
    else:
        resid_pre = residuals_lowess_full(x[:n_pre], LOWESS_FRAC)

    x_neu_all, x_pre_all, cut_ix = split_neutral_pre_by_frac(resid_pre, PRE_FRAC)

    L_auto = max(64, min(500, int(0.8 * n_pre)))
    L_eff  = min(L, L_auto)
    print(f"[INFO] series {cfg['SERIES']}: n={n}, n_pre={n_pre}, L_eff={L_eff}, cut_ix(pre)={cut_ix}")

    i_last = ix_trans
    xr_full = np.asarray(resid_pre[:i_last+1], float)

    if i_last + 1 >= L_eff:
        win_raw = windows_lastL(xr_full, L_eff)
        win_raw_times = t_num[i_last-L_eff+1:i_last+1].astype(float)
    else:
        step = float(np.median(np.diff(t_num[:min(len(t_num)-1, 10)]))) if len(t_num) > 1 else 1.0
        t_end = float(t_num[i_last])
        win_raw = windows_lastL(xr_full, L_eff)
        win_raw_times = t_end + step * (np.arange(L_eff) - (L_eff-1))

    t_cut_num    = float(t_num[min(max(cut_ix, 0), len(t_num)-1)])
    t_trans_num  = float(t_num[ix_trans])

    results = []
    for meth in SURR_METHODS:
        print(f"\n[Surrogate] Training and computing CAM: {meth}")
        gen = get_surrogate_generator(meth)

        X_neu_s = gen(x_neu_all, n=SURR_N_PER_CLASS, rng=0)
        X_pre_s = gen(x_pre_all, n=SURR_N_PER_CLASS, rng=1)

        X_neu = np.stack([normalize_window(windows_lastL(s, L_eff)) for s in X_neu_s], axis=0)[..., None]
        X_pre = np.stack([normalize_window(windows_lastL(s, L_eff)) for s in X_pre_s], axis=0)[..., None]
        X_sdml = np.concatenate([X_neu, X_pre], axis=0)
        y_sdml = np.concatenate([np.zeros(len(X_neu)), np.ones(len(X_pre))]).astype(int)

        models_sdml = []
        for seed in range(N_MODELS):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            m = build_multihead_cnn_binary(L=L_eff)

            n_tot = len(X_sdml)
            idxs = np.arange(n_tot)
            rng = np.random.default_rng(seed)
            rng.shuffle(idxs)
            cut = int(0.7 * n_tot)
            tr, va = idxs[:cut], idxs[cut:]

            es = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=PATIENCE, restore_best_weights=True
            )

            m.fit(
                X_sdml[tr], y_sdml[tr],
                validation_data=(X_sdml[va], y_sdml[va]),
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                verbose=0, callbacks=[es]
            )
            models_sdml.append(m)

        win_inp = normalize_window(win_raw)
        cam01   = grad_cam_ensemble(models_sdml, win_inp, class_index=0)
        results.append(dict(name=meth, win_signal=win_raw.copy(), cam01=cam01.copy()))

    M = len(results)
    fig_h = max(2.8*M, 6.0)
    fig, axes = plt.subplots(M, 1, figsize=(8, fig_h), constrained_layout=True, sharex=False)
    if M == 1:
        axes = np.array([axes])

    xlim_left  = float(win_raw_times[0])
    xlim_right = float(t_trans_num + POST_MARGIN_DAYS)

    for i, (ax, res) in enumerate(zip(axes, results)):
        sig  = res["win_signal"]
        cam  = res["cam01"]

        ax.set_xlim(xlim_left, xlim_right)
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=TICK_MONTH_INTERVAL))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', labelrotation=XROT_DEG, labelsize=FONTSIZE_TICKS)
        ax.tick_params(axis='y', labelsize=FONTSIZE_TICKS)

        t0, t1 = float(win_raw_times[0]), float(win_raw_times[-1])

        if t0 < t_cut_num < t1:
            ax.axvspan(t0, t_cut_num, color="#4C78A8", alpha=0.10, lw=0, zorder=0)
            ax.axvspan(t_cut_num, t1, color="#F58518", alpha=0.10, lw=0, zorder=0)
        else:
            ax.axvspan(t0, t1, color="#F58518", alpha=0.06, lw=0, zorder=0)

        ax.axvspan(t_trans_num, xlim_right, color="0.85", alpha=0.5, lw=0, zorder=0)

        ax.plot(win_raw_times, sig, color=COLOR_SIGNAL, lw=1.2, zorder=2)
        ymin, ymax = np.nanmin(sig), np.nanmax(sig)
        if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
            ymin, ymax = -1.0, 1.0

        ax.imshow(
            cam[None, :],
            extent=[win_raw_times[0], win_raw_times[-1], ymin, ymax],
            aspect="auto", cmap=CMAP_HEAT, alpha=0.35, origin="lower", zorder=1
        )

        ax.axvline(t_trans_num, ls="--", color="0.5", lw=1.0, zorder=3)

        ax.set_ylabel(f"{res['name']}\nResidual", fontsize=FONTSIZE_LABELS)
        if i == M-1:
            ax.set_xlabel("Time", fontsize=FONTSIZE_LABELS)

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"gradcam_serie{cfg['SERIES']}.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figure saved: {out}")


if __name__ == "__main__":
    wanted = None
    for i, a in enumerate(sys.argv):
        if a == "--series" and i + 1 < len(sys.argv):
            wanted = int(sys.argv[i + 1])
    for cfg in CFG_LIST:
        if wanted is None or cfg["SERIES"] == wanted:
            run_series(cfg)
