# CTI Early-Warning Signals — Chile 2014–2024

Code and data to reproduce the machine-learning figures of the paper *"Deep
learning detection of early warning signals for critical transitions in
social systems: Chile, 2014–2024"*.

The analyses operate on a Critical Transition Index (CTI) built from Chilean
press data. Classifiers are trained on surrogate time series generated from
the neutral and pre-transition segments of each series (SDML: surrogate-data
machine learning) and compared against classical early-warning indicators
(rolling variance and lag-1 autocorrelation) via pseudo-ROC curves. A
pretrained multiclass classifier (fold / hopf / transcritical /
no-transition) is additionally applied to the daily series, and Grad-CAM
maps interpret the SDML classifier decisions.

## Structure

```
data/         input CSV series (serie1.csv, serie2.csv, serie3.csv)
pretrained/   pretrained multiclass classifier files
src/          index-construction and figure-generation scripts
networks/     semantic-network / giant-component analysis (own README)
figures/      output PDFs (the published versions are included)
output/       cached models and results (ignored by git)
```

## Setup

```
pip install -r requirements.txt
```

Python 3.11 was used. **Note that `ewstools` must be installed** (it is
included in `requirements.txt`, or install it directly):

```
pip install ewstools
```

It is required by `make_fig_class_1.py` / `make_fig_class_2.py`, which also
need the pretrained classifier files in `pretrained/` (see
`pretrained/README.md`).

## Usage

| Script | Output |
|---|---|
| `python src/sdml_pipeline_weekly.py --method AAFT` | `figures/CNN_AAFT_weekly.pdf` |
| `python src/make_svm_weekly.py --method AAFT` | `figures/SVM_AAFT_weekly.pdf` |
| `python src/make_lstm_weekly.py --method AAFT` | `figures/LSTM_AAFT_weekly.pdf` |
| `python src/make_cross_forecast.py --span 0.5` | `figures/cross_forecast_S1S2_weekly_span050.pdf` |
| `python src/make_lowess_heatmap.py` | `figures/LOWESS_heatmap_DLprob.pdf` |
| `python src/make_fig_class_1.py` | `figures/fig_class_1.pdf` |
| `python src/make_fig_class_2.py` | `figures/fig_class_2.pdf` |
| `python src/make_gradcam.py --series 1` | `figures/gradcam_serie1.pdf` |
| `Rscript src/make_cti_index.R` | `data/serie1.csv`, `serie2.csv`, `serie3.csv` |
| `python networks/run_analysis.py` | `networks/figures/fig_networks_w200.pdf`, `fig_gc_robustness.pdf` |

`--method` accepts `AAFT`, `IAAFT`, `FT`, or `RP` (surrogate-generation
procedure). `make_cross_forecast.py` also accepts `--span` (lowess span,
default 0.5) and `--burnin short|span` (default `short`); trained ensembles
are cached under `output/` and reused on subsequent runs. `make_gradcam.py`
runs all three series unless `--series` is given.

All random seeds are fixed, so repeated runs reproduce the same figures.
The `figures/` directory ships the published versions of all PDFs.

`src/make_cti_index.R` documents how the CTI series in `data/` are built
from the raw news corpus (`data/news.csv`, with columns `fecha`, `fuente`,
`cuerpo_limpio`); the raw corpus itself is not distributed. It requires R
with the packages listed at the top of the script.

`networks/` contains the semantic-network and giant-component analysis
(three period networks, threshold sweep, volume-matched resampling) with its
own README, requirements, and published outputs; it also builds from the raw
corpus, which is not distributed.
