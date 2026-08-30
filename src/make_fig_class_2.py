#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily CTI series with class probabilities from a pretrained multiclass
classifier (fold / hopf / transcritical / no-transition), applied without
retraining through the ewstools interface on fixed 1,500-point windows.
Same as make_fig_class_1.py with the estimation window extended to mid-2023.

Requires the pretrained classifier files under pretrained/ (see README).

Usage:
    python src/make_fig_class_2.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
np.random.seed(0)
import pandas as pd

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")
FIG_DIR  = os.path.join(REPO_DIR, "figures")
PRETRAINED_DIR = os.path.join(REPO_DIR, "pretrained")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import ewstools
from tensorflow.keras.models import load_model

FONTSIZE_LABELS = 14
FONTSIZE_TICKS  = 14
LEGEND_FS       = 12

CLASS_NAMES = ["fold", "hopf", "transcritical", "no-transition"]

COLOR_GREY  = "0.4"
COLOR_BLACK = "0.0"

CLASS_COLORS = {
    "fold": "red",
    "hopf": "green",
    "transcritical": "blue",
    "no-transition": "0.5",
}

YLABEL_X = -0.07

CFG_1 = dict(
    title="a) Series 1: Security, education, health, work, and politics",
    CSV_PATH=os.path.join(DATA_DIR, "serie1.csv"),
    DATE_COL="fecha",
    VALUE_COL="indice_cti",
    START_DATE="2014-01-01",
    TRANSITION_DATE="2023-06-30",
    lowess_span=0.4,
    INC=20,
    TEMPORALIDAD="Diario",
    MODEL_PATH=os.path.join(PRETRAINED_DIR, "best_model_1_2_len1500.keras"),
)

CFG_3 = dict(
    title="b) Series 2: Crime, students, hospital, wages, and government",
    CSV_PATH=os.path.join(DATA_DIR, "serie2.csv"),
    DATE_COL="time",
    VALUE_COL="Value",
    START_DATE="2014-01-01",
    TRANSITION_DATE="2023-06-30",
    lowess_span=0.4,
    INC=20,
    METHOD_COL="Method",
    METHOD_FILTER="Original",
    TEMPORALIDAD_COL="Frequency",
    TEMPORALIDAD_FILTER="Daily",
    MODEL_PATH=os.path.join(PRETRAINED_DIR, "best_model_1_1_len1500.keras"),
)

CFG_5 = dict(
    title="c) Series 3: Lagos, Bachelet, Boric, Kast, and Piñera",
    CSV_PATH=os.path.join(DATA_DIR, "serie3.csv"),
    DATE_COL="time",
    VALUE_COL="Value",
    START_DATE="2014-01-01",
    TRANSITION_DATE="2023-06-30",
    lowess_span=0.4,
    INC=20,
    METHOD_COL="Method",
    METHOD_FILTER="Original",
    TEMPORALIDAD_COL="Frequency",
    TEMPORALIDAD_FILTER="Daily",
    MODEL_PATH=os.path.join(PRETRAINED_DIR, "best_model_1_1_len1500.keras"),
)

def _align_names(df_dl):
    if df_dl is None or df_dl.empty:
        return df_dl
    rename_map = {}
    for j, name in enumerate(CLASS_NAMES):
        if j in df_dl.columns:            rename_map[j] = name
        if str(j) in df_dl.columns:       rename_map[str(j)] = name
        if f"c{j}" in df_dl.columns:      rename_map[f"c{j}"] = name
    return df_dl.rename(columns=rename_map)

def enforce_dl_labels(df_dl):
    if df_dl is None or df_dl.empty:
        return df_dl
    keep = ["time"] + [c for c in CLASS_NAMES if c in df_dl.columns]
    return df_dl[keep]

def pad_left_reflect(y, need):
    if need <= 0:
        return y, 0
    if len(y) == 0:
        raise ValueError("Empty series; cannot pad.")
    k = min(len(y), need)
    base = y[:k][::-1]
    rep  = int(np.ceil(need / len(base)))
    ref  = np.tile(base, rep)[:need]
    return np.concatenate([ref, y]), need

def make_ts_from_series(y_vals, x_index_num, transition_num, lowess_span):
    s = pd.Series(np.asarray(y_vals, dtype=float), index=np.asarray(x_index_num, dtype=float), name="state")
    if not s.index.is_unique:
        s = s.groupby(level=0).mean().sort_index()
    ts = ewstools.TimeSeries(s, transition=float(transition_num))
    ts.detrend(method="Lowess", span=lowess_span)
    return ts

def run_dl_fixed_len(ts_focus, L_target, model_path, inc, lowess_span, allow_padding=False):
    if not Path(model_path).exists():
        print(f"[DL] Model not found: {model_path}")
        return None, None

    pre_mask = ts_focus.state.index <= ts_focus.transition
    y_pre = ts_focus.state.loc[pre_mask, "state"].to_numpy()
    t_pre = ts_focus.state.index[pre_mask].to_numpy()
    n_pre = len(y_pre)
    if n_pre == 0:
        print("[DL] No pre-transition data.")
        return None, None

    if n_pre >= L_target:
        y_use = y_pre[-L_target:]
        t_use = t_pre[-L_target:]
        ts_dl = make_ts_from_series(y_use, t_use, ts_focus.transition, lowess_span)
        cut_time_min = float(t_use.min())
        pad_added = 0
    else:
        if not allow_padding:
            print(f"[DL] Pre-transition length ({n_pre}) below L={L_target} and padding disabled.")
            return None, None
        need = L_target - n_pre
        y_ext, added = pad_left_reflect(y_pre, need)
        t0 = float(t_pre.min())
        t_ext = np.arange(len(y_ext), dtype=float)
        t_ext = t_ext + (t0 - len(y_ext))
        shift = float(t_pre.max()) - float(t_ext[-1])
        t_ext = t_ext + shift
        ts_dl = make_ts_from_series(y_ext, t_ext, ts_focus.transition, lowess_span)
        cut_time_min = float(t_pre.min())
        pad_added = added

    try:
        clf = load_model(model_path)
    except Exception as e:
        print(f"[DL] Could not load {model_path}: {e}")
        return None, None

    tmin = float(ts_dl.state.index.min())
    tmax = float(ts_dl.transition)
    ts_dl.apply_classifier(clf, tmin=tmin, tmax=tmax, verbose=0)
    if hasattr(ts_dl, "clear_dl_preds"):
        ts_dl.clear_dl_preds()
    ts_dl.apply_classifier_inc(clf, inc=inc, verbose=0)

    df_dl = ts_dl.dl_preds.copy()
    df_dl = _align_names(df_dl)
    df_dl = enforce_dl_labels(df_dl)
    df_dl = df_dl[df_dl["time"] >= cut_time_min]

    print(f"[DL] {Path(model_path).name} | L={L_target} | pre_real={n_pre} | padding={pad_added}")
    return df_dl, L_target

def load_series(cfg):
    df = pd.read_csv(cfg["CSV_PATH"])

    method_col = cfg.get("METHOD_COL")
    method_filter = cfg.get("METHOD_FILTER")
    if method_col and method_col in df.columns and method_filter is not None:
        df = df[df[method_col] == method_filter].copy()

    temp_col = cfg.get("TEMPORALIDAD_COL")
    temp_filter = cfg.get("TEMPORALIDAD_FILTER")
    if temp_col and temp_filter is not None:
        if temp_col not in df.columns:
            raise ValueError(f"Missing column '{temp_col}' in the CSV.")
        want = str(temp_filter).strip().lower()
        df = df[df[temp_col].astype(str).str.strip().str.lower() == want].copy()

    temporalidad = cfg.get("TEMPORALIDAD", None)
    if temporalidad is not None:
        if "Temporalidad" not in df.columns:
            raise ValueError("Missing column 'Temporalidad' in the CSV.")
        want = str(temporalidad).strip().lower()
        df = df[df["Temporalidad"].astype(str).str.strip().str.lower() == want].copy()

    date_col = cfg["DATE_COL"]
    value_col = cfg["VALUE_COL"]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col])

    if cfg.get("START_DATE"):
        df = df[df[date_col] >= pd.to_datetime(cfg["START_DATE"])]

    if df.empty:
        raise ValueError("No data after filters/START_DATE.")

    df = df.sort_values(date_col).reset_index(drop=True)

    if df[date_col].duplicated().any():
        df = (df.groupby(date_col, as_index=False)[value_col]
                .mean()
                .sort_values(date_col)
                .reset_index(drop=True))

    x_num = mdates.date2num(df[date_col].values.astype("datetime64[ns]"))
    y_all = df[value_col].astype(float).to_numpy()

    t_target = pd.to_datetime(cfg["TRANSITION_DATE"])
    idx_star = int(((df[date_col] - t_target).abs().dt.total_seconds()).values.argmin())
    t_transition_num = float(mdates.date2num(df.loc[idx_star, date_col].to_datetime64()))
    return x_num, y_all, t_transition_num

def plot_series_with_lowess(ax, ts_full, title):
    st = ts_full.state
    ax.plot(st.index, st["state"].to_numpy(), color=COLOR_GREY, lw=1.0)
    if "smoothing" in st.columns:
        ax.plot(st.index, st["smoothing"].to_numpy(), color=COLOR_BLACK, lw=1.5)
    ax.axvline(ts_full.transition, ls="--", color=COLOR_GREY, lw=1.0)
    ax.set_ylabel("Critical transition index", fontsize=FONTSIZE_LABELS)
    ax.yaxis.set_label_coords(YLABEL_X, 0.5)
    ax.set_title(title, fontsize=FONTSIZE_LABELS, loc="left")

def plot_dl_probability_4classes(ax, df_dl, ts_ref):
    ax.set_ylabel("DL probability", fontsize=FONTSIZE_LABELS)
    ax.yaxis.set_label_coords(YLABEL_X, 0.5)
    ax.axvline(ts_ref.transition, ls="--", color=COLOR_GREY, lw=1.0)
    ax.set_ylim(0, 1)

    if df_dl is None or df_dl.empty:
        ax.text(0.5, 0.5, "DL unavailable", ha="center", va="center", transform=ax.transAxes)
        return

    t = df_dl["time"].to_numpy()
    styles = {
        "fold":          dict(ls="-",  lw=2.0, marker="o"),
        "hopf":          dict(ls="--", lw=2.0, marker="s"),
        "transcritical": dict(ls="-.", lw=2.0, marker="^"),
        "no-transition": dict(ls=":",  lw=2.2, marker="D"),
    }
    me = max(1, len(t) // 25)

    for cls in CLASS_NAMES:
        if cls in df_dl.columns:
            y = df_dl[cls].to_numpy()
            st = styles.get(cls, dict(ls="-", lw=2.0, marker=None))
            ax.plot(
                t, y,
                color=CLASS_COLORS.get(cls, COLOR_GREY),
                linestyle=st["ls"],
                linewidth=st["lw"],
                marker=st["marker"],
                markevery=me,
                mec="black", mfc="white", ms=4,
                label=cls
            )

def format_time_axis(ax, set_xlabel=False):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    if set_xlabel:
        ax.set_xlabel("Time", fontsize=FONTSIZE_LABELS)

def plot_panel(ax_ts, ax_dl, cfg):
    x_num, y_all, t_transition_num = load_series(cfg)
    ts_full = make_ts_from_series(y_all, x_num, t_transition_num, cfg["lowess_span"])

    df_dl, _ = run_dl_fixed_len(
        ts_focus=ts_full,
        L_target=1500,
        model_path=cfg["MODEL_PATH"],
        inc=cfg["INC"],
        lowess_span=cfg["lowess_span"],
        allow_padding=False
    )

    plot_series_with_lowess(ax_ts, ts_full, cfg["title"])
    plot_dl_probability_4classes(ax_dl, df_dl, ts_full)

    ax_ts.tick_params(axis="both", labelsize=FONTSIZE_TICKS)
    ax_dl.tick_params(axis="both", labelsize=FONTSIZE_TICKS)
    ax_ts.tick_params(labelbottom=False)

    format_time_axis(ax_ts, set_xlabel=False)
    format_time_axis(ax_dl, set_xlabel=True)

    xmin, xmax = float(ts_full.state.index.min()), float(ts_full.state.index.max())
    ax_ts.set_xlim(xmin, xmax)
    ax_dl.set_xlim(xmin, xmax)

fig = plt.figure(figsize=(20, 11))
gs_bottom = fig.add_gridspec(
    1, 2,
    left=0.06, right=0.94, bottom=0.08, top=0.46,
    wspace=0.13
)

gs_bl = gs_bottom[0, 0].subgridspec(2, 1, hspace=0.06)
ax2_ts = fig.add_subplot(gs_bl[0, 0])
ax2_dl = fig.add_subplot(gs_bl[1, 0], sharex=ax2_ts)

gs_br = gs_bottom[0, 1].subgridspec(2, 1, hspace=0.06)
ax3_ts = fig.add_subplot(gs_br[0, 0])
ax3_dl = fig.add_subplot(gs_br[1, 0], sharex=ax3_ts)

pos_bl = ax2_ts.get_position()
pos_br = ax3_ts.get_position()
panel_w = pos_bl.width
mid = ((pos_bl.x0 + pos_bl.x1) / 2 + (pos_br.x0 + pos_br.x1) / 2) / 2
x0_top = mid - panel_w / 2

y0_top_union = 0.58
y1_top_union = 0.96
h_union = y1_top_union - y0_top_union
gap = 0.06 * h_union
h_each = (h_union - gap) / 2

ax1_dl = fig.add_axes([x0_top, y0_top_union, panel_w, h_each])
ax1_ts = fig.add_axes([x0_top, y0_top_union + h_each + gap, panel_w, h_each], sharex=ax1_dl)

plot_panel(ax1_ts, ax1_dl, CFG_1)
plot_panel(ax2_ts, ax2_dl, CFG_3)
plot_panel(ax3_ts, ax3_dl, CFG_5)

for ax in (ax2_dl, ax3_dl):
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

h1, l1 = ax1_dl.get_legend_handles_labels()
p1 = ax1_dl.get_position()
x_top = p1.x1 + 0.01
y_top = (p1.y0 + p1.y1) / 2

if len(h1) > 0:
    fig.legend(h1, l1, frameon=False, fontsize=LEGEND_FS, loc="center left",
               bbox_to_anchor=(x_top, y_top), bbox_transform=fig.transFigure)

os.makedirs(FIG_DIR, exist_ok=True)
PDF_PATH = os.path.join(FIG_DIR, "fig_class_2.pdf")
fig.savefig(PDF_PATH, bbox_inches="tight", transparent=False)
print(f"[OK] Figure saved: {PDF_PATH}")
