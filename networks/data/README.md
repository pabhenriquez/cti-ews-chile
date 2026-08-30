# data/

This folder is empty in the distributed code: **no data is shipped**.

## What to put here

`news.csv` — the press corpus, one row per news item, with the columns

```
fecha,cuerpo,fuente,url,fecha_scraping,cuerpo_limpio
```

`cuerpo_limpio` (lowercased, punctuation removed) is the field the analysis
uses. It is copyrighted press material and is not redistributed with this code.
Place your copy here, or point the script at it:

```bash
python3 run_analysis.py --corpus /path/to/news.csv
```

The stopword list applied before bigrams are formed is written out inside
`run_analysis.py`, so the method is fully contained in the code.

## What appears here after a run

`articles.parquet`, `monthly_bigrams.parquet` and `windows.npz` are written by
steps 1-3 and rebuilt on every run. None of them is versioned.
