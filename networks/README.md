# Semantic networks and the giant component

Replication code for the semantic-network analysis (Methods subsection
*"Semantic networks, giant component, and volume-matched resampling"*): the
three period networks, the giant-component threshold sweep, the six
volume-matched resampling schemes, the statistical tests, and both figures.

## Quick start

```bash
pip install -r requirements.txt
python run_analysis.py
```

The whole analysis is one commented script, `run_analysis.py`, which runs the
steps in order, prints progress, and fills:

```
outputs/    numerical results, one CSV per step
figures/    fig_networks_w200 (main text) and fig_gc_robustness
            (supplementary), as PDF and PNG
```

The final step verifies the outputs against the values reported in the paper
and prints OK or FAIL for each check.

**Options**

```bash
python run_analysis.py --corpus /path/to/news.csv   # corpus kept elsewhere
python run_analysis.py --reuse-windows              # skip corpus parsing,
                                                    # reuse data/windows.npz
python run_analysis.py --replicates 100             # more replicates per cell
python run_analysis.py --help
```

## Input

The raw press corpus (`data/news.csv`, one row per news item, with a
lowercased `cuerpo_limpio` text field) is copyrighted press material and is
not redistributed; see `data/README.md` for the expected columns. It is the
same corpus used by `../src/make_cti_index.R`. Intermediate files
(`articles.parquet`, `monthly_bigrams.parquet`, `data/windows.npz`) are
rebuilt on every run.

## Results reproduced

At the illustration threshold `w = 200`, without resampling:

| Window | N | E | S_GC |
|---|---|---|---|
| Neutral | 912 | 884 | 0.546 |
| Pre-transition | 1,216 | 1,309 | 0.632 |
| Post-transition | 1,837 | 2,430 | 0.749 |

Under resampling, the post-transition window holds the largest giant
component in 56 of 56 (scheme × threshold) cells with `w ≤ 400`; the paired
difference post − neutral at `w = 200` is +0.12 to +0.16 depending on the
scheme, with 95% intervals excluding zero; the mean-degree ratio
post/neutral is ≈ 1.09–1.15.
