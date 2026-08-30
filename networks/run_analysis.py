#!/usr/bin/env python3
"""
Semantic networks and the giant component -- full analysis in one script.

    python3 run_analysis.py                  run the whole analysis
    python3 run_analysis.py --reuse-windows  skip steps 1-3, reuse an earlier run
    python3 run_analysis.py --help           all options

Replication material for the Methods subsection "Semantic networks, giant
component, and volume-matched resampling" of

    Henriquez, Rozas, Mascareno, Ruz & Rica, "Deep learning detection of early
    warning signals for critical transitions in social systems using press
    data: Chile, 2014-2024".

The script is self-contained: on first run it builds an isolated environment in
./.venv, installs the exact package versions the results were validated with,
and re-launches itself inside it. Nothing on the machine is modified, and an
existing Anaconda or system Python is never used to run the analysis.

Input:  data/news.csv -- the press corpus, one row per news item. It is not
        distributed with this code (copyrighted press material); place your
        copy there, or point at it with --corpus /path/to/news.csv.

Writes: outputs/*.csv  (numerical results)   figures/*.pdf, *.png  (both figures)
"""

# =============================================================================
# STEP 0 -- BOOTSTRAP THE ENVIRONMENT
# Standard library only. Everything above the re-launch must run on any Python.
# =============================================================================
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
IMPORTS = "import numpy, pandas, igraph, matplotlib"


def imports_cleanly(python):
    """Test the imports in a fresh process.

    Done out of process because an ABI mismatch -- typically NumPy 2.x next to
    extensions compiled for NumPy 1.x, the usual state of an Anaconda install
    that has been upgraded piecemeal -- leaves a half-initialised extension
    behind that cannot be undone within a running interpreter.
    """
    return subprocess.run([str(python), "-c", IMPORTS],
                          capture_output=True, text=True).returncode == 0


def bootstrap():
    """Build ./.venv if needed and re-launch this script inside it."""
    if os.environ.get("EWS_BOOTSTRAPPED"):        # already inside the venv
        return
    if imports_cleanly(sys.executable):           # this interpreter is fine
        return

    print("This Python cannot run the analysis; preparing an isolated "
          "environment.\n")
    if not VENV_PY.exists():
        print(f"  creating {VENV}")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)

    if not (VENV / ".installed").exists():
        print("  installing the pinned versions (about a minute) ...")
        subprocess.run([str(VENV_PY), "-m", "pip", "install", "-q",
                        "--upgrade", "pip"], check=False)
        pinned = subprocess.run([str(VENV_PY), "-m", "pip", "install", "-q",
                                 "-r", str(ROOT / "requirements.txt")])
        if pinned.returncode != 0:
            print("  pinned versions unavailable here; using minimum versions")
            subprocess.run([str(VENV_PY), "-m", "pip", "install", "-q", "-r",
                            str(ROOT / "requirements-minimum.txt")], check=True)
        if not imports_cleanly(VENV_PY):
            sys.exit("the environment was built but does not import cleanly; "
                     "delete the .venv folder and run this again")
        (VENV / ".installed").touch()

    print(f"  environment ready\n\n  re-launching inside {VENV.name}\n")
    os.environ["EWS_BOOTSTRAPPED"] = "1"
    os.execv(str(VENV_PY), [str(VENV_PY), str(Path(__file__).resolve()),
                            *sys.argv[1:]])


bootstrap()

# From here on the scientific stack is guaranteed to be importable.
import argparse
import csv
import gzip
import random
import re
import time
import unicodedata
from collections import Counter

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =============================================================================
# PARAMETERS -- everything reported in the Methods section, in one place
# =============================================================================
DATA, OUT, FIG = ROOT / "data", ROOT / "outputs", ROOT / "figures"

# The raw corpus. Either a plain .csv or a gzipped .csv.gz is accepted, and
# --corpus points the script at a copy kept anywhere else, so a 2.5 GB file
# need not be duplicated into this folder.
RAW_CORPUS_CANDIDATES = [DATA / "news.csv", DATA / "news.csv.gz"]


def find_corpus(explicit=None):
    """Locate the raw corpus, or explain where to put it."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            sys.exit(f"corpus not found: {path}")
        return path
    for candidate in RAW_CORPUS_CANDIDATES:
        if candidate.exists():
            return candidate
    sys.exit("no raw corpus found. Put news.csv (or news.csv.gz) in data/, or "
             "pass --corpus /path/to/news.csv")

DATE_MIN, DATE_MAX = "2014-01-01", "2024-12-31"

# Series 1 concepts, exactly as defined in the study: an item enters the
# networks when its cleaned text contains at least one of these five terms as a
# whole word, matched on the accent-folded text. This is the filter that makes
# the networks describe the same communicative domain as the CTI.
SERIES1_CONCEPTS = ["seguridad", "educacion", "salud", "trabajo", "politica"]

# ---------------------------------------------------------------------------
# The text-preparation dictionary
#
# Bigrams are built from cuerpo_limpio after removing digits and the words
# below. The list is written out here, rather than read from a file, so that
# the whole method is contained in the code and can be inspected and cited.
#
# It is assembled from three explicit groups:
#
#   1. FUNCTION_WORDS -- articles, prepositions, conjunctions, auxiliary verbs,
#      generic quantifiers and time expressions. Without this step pairs such as
#      "si bien", "puede ser" or "cada vez" are among the strongest edges of the
#      network, which says nothing about what the press was talking about.
#
#   2. KEEP_IN_TEXT -- words that look like function words but are part of
#      names or of domain concepts in this corpus, and are therefore put back.
#      Removing them would erase real nodes: "Nueva Mayoria" and "Chile Vamos"
#      are political coalitions, "primera vuelta" an electoral round, "cuenta
#      publica" the annual presidential address, "medio ambiente" a policy
#      domain, "ex presidente" a former officeholder, "nuevos casos" the
#      pandemic count.
#
#   3. WEB_BOILERPLATE -- markup left over from the scraping. These are not
#      Spanish words at all: "dfp topright" is an ad-server slot and, left in,
#      it ranks among the ten strongest pairs of the whole corpus.
# ---------------------------------------------------------------------------

FUNCTION_WORDS = frozenset([
    "0", "1", "10", "100", "18", "2", "20", "21", "3", "30", "35", "4", "400",
    "4d", "5", "50", "500", "5d", "6", "60", "600", "6to", "7", "8", "800",
    "8d", "9", "_", "a", "acuerdo", "acá", "adelante", "ademas", "además",
    "adrede", "afirmó", "agregó", "ahi", "ahora", "ahí", "al", "algo",
    "alguna", "algunas", "alguno", "algunos", "algún", "alli", "allí",
    "alrededor", "ambos", "ampleamos", "antano", "antaño", "ante", "anterior",
    "antes", "apenas", "aproximadamente", "aquel", "aquella", "aquellas",
    "aquello", "aquellos", "aqui", "aquél", "aquélla", "aquéllas", "aquéllos",
    "aquí", "arriba", "arribaabajo", "aseguró", "asi", "así", "atras", "aun",
    "aunque", "ayer", "añadió", "aún", "b", "bajo", "bastante", "bien",
    "breve", "c", "cada", "casi", "cerca", "cierta", "ciertas", "cierto",
    "ciertos", "cinco", "claro", "comentó", "como", "con", "conmigo",
    "conseguimos", "conseguir", "considera", "consideró", "consigo",
    "consigue", "consiguen", "consigues", "contigo", "contra", "cosas",
    "creo", "cual", "cuales", "cualquier", "cuando", "cuanta", "cuantas",
    "cuanto", "cuantos", "cuatro", "cuenta", "cuál", "cuáles", "cuándo",
    "cuánta", "cuántas", "cuánto", "cuántos", "cómo", "d", "da", "dado",
    "dan", "dar", "de", "dea", "debajo", "debido", "decir", "dejó", "del",
    "delante", "demasiado", "demás", "dentro", "deprisa", "desde", "despacio",
    "despues", "después", "detras", "detrás", "dia", "dias", "dice", "dicen",
    "dicho", "dieron", "dijeron", "dijo", "dio", "donde", "dos", "durante",
    "día", "días", "dónde", "e", "ejemplo", "el", "ella", "ellas", "ello",
    "ellos", "embargo", "empleais", "en", "encima", "encuentra", "enfrente",
    "enseguida", "entonces", "entre", "era", "erais", "eramos", "eran",
    "eras", "eres", "es", "esa", "esas", "ese", "eso", "esos", "esta",
    "estaba", "estabais", "estaban", "estabas", "estad", "estada", "estadas",
    "estados", "estais", "estamos", "estan", "estando", "estar", "estaremos",
    "estará", "estarán", "estarás", "estaré", "estaréis", "estaría",
    "estaríais", "estaríamos", "estarían", "estarías", "estas", "este",
    "estemos", "esto", "estos", "estoy", "estuve", "estuviera", "estuvierais",
    "estuvieran", "estuvieras", "estuvieron", "estuviese", "estuvieseis",
    "estuviesen", "estuvieses", "estuvimos", "estuviste", "estuvisteis",
    "estuviéramos", "estuviésemos", "estuvo", "está", "estábamos", "estáis",
    "están", "estás", "esté", "estéis", "estén", "estés", "ex", "excepto",
    "existe", "existen", "explicó", "expresó", "f", "fin", "final", "fu",
    "fue", "fuera", "fuerais", "fueran", "fueras", "fueron", "fuese",
    "fueseis", "fuesen", "fueses", "fui", "fuimos", "fuiste", "fuisteis",
    "fuéramos", "fuésemos", "g", "gran", "grandes", "gueno", "h", "h1", "h2",
    "h3", "h4", "h5", "ha", "haber", "habia", "habida", "habidas", "habido",
    "habidos", "habiendo", "habla", "hablan", "habremos", "habrá", "habrán",
    "habrás", "habré", "habréis", "habría", "habríais", "habríamos",
    "habrían", "habrías", "habéis", "había", "habíais", "habíamos", "habían",
    "habías", "hace", "haceis", "hacemos", "hacen", "hacer", "hacerlo",
    "haces", "hacia", "haciendo", "hago", "han", "has", "hasta", "hay",
    "haya", "hayamos", "hayan", "hayas", "hayáis", "he", "hecho", "hemos",
    "hicieron", "hizo", "horas", "hoy", "hube", "hubiera", "hubierais",
    "hubieran", "hubieras", "hubieron", "hubiese", "hubieseis", "hubiesen",
    "hubieses", "hubimos", "hubiste", "hubisteis", "hubiéramos", "hubiésemos",
    "hubo", "i", "igual", "incluso", "indicó", "informo", "informó",
    "intenta", "intentais", "intentamos", "intentan", "intentar", "intentas",
    "intento", "ir", "j", "junto", "k", "l", "la", "lado", "largo", "las",
    "le", "lejos", "les", "llegó", "lleva", "llevar", "lo", "los", "luego",
    "lugar", "m", "m1", "m2", "m3", "m4", "m5", "mal", "manera", "manifestó",
    "mas", "mayor", "me", "mediante", "medio", "mencionó", "menos", "menudo",
    "mi", "mia", "mias", "mientras", "mio", "mios", "mis", "misma", "mismas",
    "mismo", "mismos", "modo", "momento", "mucha", "muchas", "mucho",
    "muchos", "muy", "más", "mí", "mía", "mías", "mío", "míos", "n", "nada",
    "nadie", "ni", "ninguna", "ningunas", "ninguno", "ningunos", "ningún",
    "nos", "nosotras", "nosotros", "nuestra", "nuestras", "nuestro",
    "nuestros", "nueva", "nuevas", "nuevo", "nuevos", "nunca", "o", "ocho",
    "os", "otra", "otras", "otro", "otros", "p", "para", "parece", "parte",
    "partir", "pasa", "pasada", "pasado", "peor", "pero", "pesar", "poca",
    "pocas", "poco", "pocos", "podeis", "podemos", "poder", "podria",
    "podriais", "podriamos", "podrian", "podrias", "podrá", "podrán",
    "podría", "poner", "por", "por qué", "porque", "posible", "primer",
    "primera", "primero", "primeros", "principalmente", "pronto", "propia",
    "propias", "propio", "propios", "proximo", "próximo", "próximos", "pudo",
    "pueda", "puede", "pueden", "puedo", "pues", "q", "qeu", "que", "quedó",
    "queremos", "quien", "quienes", "quiere", "quiza", "quizas", "quizá",
    "quizás", "quién", "quiénes", "qué", "r", "raras", "realizado",
    "realizar", "realizó", "repente", "respecto", "rt", "s", "sabe", "sabeis",
    "sabemos", "saben", "saber", "sabes", "sal", "salvo", "se", "sea",
    "seamos", "sean", "seas", "segun", "segunda", "segundo", "según", "seis",
    "ser", "sera", "seremos", "será", "serán", "serás", "seré", "seréis",
    "sería", "seríais", "seríamos", "serían", "serías", "seáis", "señaló",
    "sido", "siempre", "siendo", "siete", "sigue", "siguiente", "sino",
    "sobre", "sois", "sola", "solamente", "solas", "solo", "solos", "somos",
    "son", "soy", "soyos", "su", "supuesto", "sus", "suya", "suyas", "suyo",
    "suyos", "sé", "sólo", "t", "tal", "tambien", "también", "tampoco", "tan",
    "tanto", "tarde", "te", "temprano", "tened", "teneis", "tenemos", "tener",
    "tenga", "tengamos", "tengan", "tengas", "tengo", "tengáis", "tenida",
    "tenidas", "tenido", "tenidos", "teniendo", "tenéis", "tenía", "teníais",
    "teníamos", "tenían", "tenías", "tercera", "ti", "tiene", "tienen",
    "tienes", "toda", "todas", "todavia", "todavía", "todo", "todos", "total",
    "tras", "trata", "través", "tres", "tu", "tus", "tuve", "tuviera",
    "tuvierais", "tuvieran", "tuvieras", "tuvieron", "tuviese", "tuvieseis",
    "tuviesen", "tuvieses", "tuvimos", "tuviste", "tuvisteis", "tuviéramos",
    "tuviésemos", "tuvo", "tuya", "tuyas", "tuyo", "tuyos", "tú", "u",
    "ultimo", "un", "una", "unas", "uno", "unos", "usa", "usais", "usamos",
    "usan", "usar", "usas", "uso", "usted", "ustedes", "v", "va", "vais",
    "vamos", "van", "varias", "varios", "vaya", "veces", "vez", "vosotras",
    "vosotros", "voy", "vuestra", "vuestras", "vuestro", "vuestros", "w",
    "word", "x", "y", "ya", "yo", "z", "él", "éramos", "ésa", "ésas", "ése",
    "ésos", "ésta", "éstas", "éste", "éstos", "última", "últimas", "último",
    "últimos",
])

# Put back into the text: function-like words that carry meaning here.
KEEP_IN_TEXT = frozenset([
    "nueva", "nuevas", "nuevo", "nuevos",   # nueva constitucion, Nueva Mayoria
    "ex",                                    # ex presidente, ex ministro
    "medio",                                 # medio ambiente
    "vamos",                                 # Chile Vamos
    "primera", "primer",                     # primera vuelta, primer ministro
    "cuenta",                                # cuenta publica, cuenta twitter
])

# Scraping residue: ad-server and CSS tokens, not language.
WEB_BOILERPLATE = frozenset([
    "dfp", "topright", "right", "left", "px", "fontsize",
    "padding", "margin", "width", "height",
])

STOPWORDS = (FUNCTION_WORDS - KEEP_IN_TEXT) | WEB_BOILERPLATE

MIN_MONTHLY_COUNT = 2       # drop pairs seen once in a calendar month

# The window length is set by the shortest period, the pre-transition period,
# which enters in full while the other two are truncated to that same length.
WINDOW_MONTHS = 27
WINDOWS = {
    "Neutral":         ("2015-10", "2017-12"),
    "Pre-transition":  ("2018-01", "2020-03"),
    "Post-transition": ("2020-04", "2022-06"),
}
WINDOW_ORDER = list(WINDOWS)

THRESHOLDS = [25, 50, 75, 100, 150, 200, 300, 400, 600, 800]
W_ILLUSTRATION = 200
PREFILTER_FRACTION = 0.6
PREFILTER_MIN_W = int(PREFILTER_FRACTION * min(THRESHOLDS))

MREF_FACTOR = 0.8           # target mass as a fraction of the smallest M_v
N_REPLICATES = 25           # B in the paper
CI_LEVEL = 95
RHO = 0.05                  # intra-class correlation of the beta-binomial
SEED = 20240101
LAYOUT_SEED = 42

COLORS = {"Neutral": "#2E75B6", "Pre-transition": "#7030A0",
          "Post-transition": "#E8262B"}
MARKERS = {"Neutral": "o", "Pre-transition": "s", "Post-transition": "^"}
GC_NODE_COLOR, OTHER_NODE_COLOR, EDGE_COLOR = "#FF6347", "#ADD8E6", "#999999"
WINDOW_LABELS = {"Neutral": "Neutral period",
                 "Pre-transition": "Pre-transition",
                 "Post-transition": "Post-transition"}
WINDOW_LEGEND = {"Neutral": "Neutral  (Oct 2015 - Dec 2017)",
                 "Pre-transition": "Pre-transition  (Jan 2018 - Mar 2020)",
                 "Post-transition": "Post-transition  (Apr 2020 - Jun 2022)"}

SCHEME_LABELS = {
    "raw":            "Raw",
    "binomial":       "Binomial thinning",
    "hypergeometric": "Hypergeometric rarefaction",
    "multinomial":    "Multinomial bootstrap",
    "betabinomial":   "Beta-binomial thinning",
    "block":          "Monthly block bootstrap",
    "binomial_full":  "Binomial thinning, unfiltered masses",
}
SCHEME_ORDER = list(SCHEME_LABELS)


def banner(text):
    print("\n" + "=" * 74 + f"\n{text}\n" + "=" * 74, flush=True)


# =============================================================================
# STEP 1 -- PARSE THE CORPUS AND KEEP THE SERIES 1 SUBSET
#
# The raw CSV cannot be read with pandas.read_csv: body texts contain
# unbalanced double quotes and embedded newlines, so a standard parser loses
# synchronisation a few thousand rows in. Two structural anchors are used
# instead -- a record begins at a line starting with an ISO date, and the
# source field is the one immediately followed by '","http'. This recovers
# 100% of the records (512,058 items, 0 malformed).
# =============================================================================
RECORD_START = re.compile(r"^\d{4}-\d{2}-\d{2},")


def parse_record(rec):
    """Split one raw record into (date, source, cleaned body), or None."""
    date = rec[:10]
    i = rec.find('","http')                     # start of the URL field
    if i < 0:
        return None
    prev = rec.rfind('","', 0, i)               # separator before the source
    if prev < 0:
        return None
    source = rec[prev + 3:i]
    j = rec.find('",', i + 3)                   # end of the URL field
    if j < 0:
        return None
    k = rec.find(',"', j + 2)                   # start of cuerpo_limpio
    if k < 0:
        return None
    clean = rec[k + 2:].rstrip()
    return clean[:-1] if clean.endswith('"') else clean, date, source


def iter_articles(path):
    """Yield (date, source, cleaned body) for every record of the corpus."""
    buffer = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        fh.readline()                           # header
        for line in fh:
            if RECORD_START.match(line):
                if buffer:
                    parsed = parse_record("".join(buffer))
                    if parsed:
                        clean, date, source = parsed
                        yield date, source, clean
                buffer = [line]
            else:
                buffer.append(line)             # multi-line body
        if buffer:
            parsed = parse_record("".join(buffer))
            if parsed:
                clean, date, source = parsed
                yield date, source, clean


def strip_accents(text):
    """Fold accents so 'educacion' and 'educación' match the same concept."""
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


def contains_series1(text):
    """Does the cleaned text contain a Series 1 concept?

    Matching is on whole tokens of the accent-folded text, so 'salud' matches
    but 'saludable' does not, and 'poltica'/'politica' are the same term.
    """
    tokens = set(strip_accents(text.lower()).split())
    return any(concept in tokens for concept in SERIES1_CONCEPTS)


DIGITS = re.compile(r"\d+")


def clean_tokens(text, stopwords):
    """Tokens used to build bigrams: lowercase, digits removed, stopwords out.

    Digits are removed as characters, not as whole tokens, so "US$ 10 millones"
    contributes the pair (us, millones) -- matching the text preparation of the
    original analysis.
    """
    return [t for t in DIGITS.sub(" ", text.lower()).split()
            if t and t not in stopwords]


def step1_parse_corpus(corpus_path):
    banner("STEP 1  parse the corpus and keep the Series 1 subset")
    print(f"  reading {corpus_path}")
    rows, total, dated, kept = [], 0, 0, 0
    for date, source, clean in iter_articles(corpus_path):
        total += 1
        if not (DATE_MIN <= date <= DATE_MAX) or not clean.strip():
            continue
        dated += 1
        if not contains_series1(clean):
            continue
        kept += 1
        rows.append((date, source, clean))
        if total % 100_000 == 0:
            print(f"  {total:,} records read", flush=True)

    articles = pd.DataFrame(rows, columns=["date", "source", "clean"])
    articles["month"] = articles["date"].str.slice(0, 7) + "-01"
    articles.to_parquet(DATA / "articles.parquet", index=False)
    print(f"  {total:,} records, {dated:,} in range, "
          f"{kept:,} with a Series 1 concept")
    return articles


# =============================================================================
# STEP 2 -- BIGRAM COUNTS BY CALENDAR MONTH
#
# Consecutive word pairs are counted per month; pairs seen fewer than twice in
# a month are dropped -- the long tail that dominates memory and never reaches
# the thresholds used. The monthly mass is recorded before and after that
# pre-filter, because resampling scheme (vi) is defined on the unfiltered one.
# =============================================================================
def step2_monthly_bigrams(articles=None):
    banner("STEP 2  bigram counts by calendar month")
    if articles is None:
        articles = pd.read_parquet(DATA / "articles.parquet")

    print(f"  {len(STOPWORDS)} stopwords removed before counting")

    frames, masses = [], []
    for month, group in articles.groupby("month", sort=True):
        counter = Counter()
        for text in group["clean"]:
            tokens = clean_tokens(text, STOPWORDS)
            counter.update(zip(tokens[:-1], tokens[1:]))
        unfiltered = sum(counter.values())
        kept = {pair: c for pair, c in counter.items() if c >= MIN_MONTHLY_COUNT}
        frames.append(pd.DataFrame({"mes": month,
                                    "word1": [p[0] for p in kept],
                                    "word2": [p[1] for p in kept],
                                    "n": list(kept.values())}))
        masses.append({"mes": month, "n_articles": len(group),
                       "mass_unfiltered": unfiltered,
                       "mass_filtered": sum(kept.values())})
        print(f"  {month}: {len(group):,} articles, "
              f"{sum(kept.values()):,} co-occurrences kept", flush=True)

    monthly = pd.concat(frames, ignore_index=True)
    monthly.to_parquet(DATA / "monthly_bigrams.parquet", index=False)
    pd.DataFrame(masses).to_csv(OUT / "02_monthly_mass.csv", index=False)
    print(f"  {len(monthly):,} (month, pair) rows written")


# =============================================================================
# STEP 3 -- THE THREE EQUAL-LENGTH WINDOWS
#
# The analytical periods last 48, 27 and 57 months, so every comparison uses
# three windows of identical length set by the shortest of them. The month by
# edge matrix is stored because the block bootstrap resamples calendar months.
# =============================================================================
def months_in_window(start, end):
    return [p.to_timestamp().strftime("%Y-%m-01")
            for p in pd.period_range(start=start, end=end, freq="M")]


def normalise_months(values):
    """Coerce a month column to 'YYYY-MM-01' strings (parquet round-trips can
    turn it into datetime.date, which compares unequal to the string keys)."""
    if values.dtype == object and len(values):
        if not isinstance(values.iloc[0], str):
            return pd.to_datetime(values).dt.strftime("%Y-%m-01")
    elif pd.api.types.is_datetime64_any_dtype(values):
        return values.dt.strftime("%Y-%m-01")
    return values.astype(str)


def step3_windows():
    banner("STEP 3  build the three equal-length windows")
    monthly = pd.read_parquet(DATA / "monthly_bigrams.parquet")
    monthly["mes"] = normalise_months(monthly["mes"])
    mass_by_month = pd.read_csv(OUT / "02_monthly_mass.csv")
    mass_by_month["mes"] = normalise_months(mass_by_month["mes"])

    store, rows = {}, []
    for window in WINDOW_ORDER:
        start, end = WINDOWS[window]
        months = months_in_window(start, end)
        assert len(months) == WINDOW_MONTHS, "windows must be equal length"

        sub_all = monthly[monthly["mes"].isin(months)]
        edges = (sub_all.groupby(["word1", "word2"], observed=True, sort=False)["n"]
                 .sum().reset_index().rename(columns={"n": "w"}))

        selected = mass_by_month["mes"].isin(months)
        mass = float(mass_by_month.loc[selected, "mass_filtered"].sum())
        mass_full = float(mass_by_month.loc[selected, "mass_unfiltered"].sum())

        # Storage pre-filter: thinning only reduces counts, so edges below this
        # value cannot cross the smallest reported threshold.
        edges = edges[edges["w"] >= PREFILTER_MIN_W].reset_index(drop=True)

        # Month by edge matrix, built with a join rather than a Python loop.
        edges["edge_id"] = np.arange(len(edges), dtype=np.int64)
        position = {m: j for j, m in enumerate(months)}
        joined = sub_all.merge(edges[["word1", "word2", "edge_id"]],
                               on=["word1", "word2"], how="inner")
        matrix = np.zeros((len(edges), len(months)), dtype=np.int32)
        matrix[joined["edge_id"].to_numpy(),
               joined["mes"].map(position).to_numpy()] = joined["n"].to_numpy()

        store[f"{window}__word1"] = edges["word1"].to_numpy().astype("U")
        store[f"{window}__word2"] = edges["word2"].to_numpy().astype("U")
        store[f"{window}__w"] = edges["w"].to_numpy().astype(np.int64)
        store[f"{window}__matrix"] = matrix
        store[f"{window}__mass"] = np.array([mass, mass_full])
        rows.append({"window": window, "start": start, "end": end,
                     "months": len(months), "edges_kept": len(edges),
                     "mass_filtered": mass, "mass_unfiltered": mass_full})
        print(f"  {window:16s} {len(edges):,} edges, M_v = {mass:,.0f}")

    np.savez_compressed(DATA / "windows.npz", **store)
    pd.DataFrame(rows).to_csv(OUT / "03_window_summary.csv", index=False)


def load_windows():
    """Load the aggregated counts of the three windows."""
    store = np.load(DATA / "windows.npz", allow_pickle=False)
    windows = {}
    for name in WINDOW_ORDER:
        mass, mass_full = store[f"{name}__mass"]
        windows[name] = {"word1": store[f"{name}__word1"],
                         "word2": store[f"{name}__word2"],
                         "n": store[f"{name}__w"],
                         "matrix": store[f"{name}__matrix"],
                         "mass": float(mass), "mass_full": float(mass_full)}
    return windows


# =============================================================================
# NETWORKS AND THE GIANT COMPONENT
#
# Two words are linked when they co-occur at least w times over the window. The
# structural integration of the network is S_GC = N_GC / N, the relative size of
# its largest connected component. Bigrams are ordered, so a pair may appear in
# both orders; in an undirected co-occurrence network these are one relation,
# hence simplify(). That leaves S_GC unchanged but makes E and <k> correct.
# =============================================================================
def giant_component_stats(g):
    n = g.vcount()
    if n == 0:
        return {"N": 0, "E": 0, "N_GC": 0, "S_GC": np.nan,
                "mean_k": np.nan, "n_components": 0}
    sizes = g.connected_components().sizes()
    n_gc = max(sizes)
    return {"N": n, "E": g.ecount(), "N_GC": n_gc, "S_GC": n_gc / n,
            "mean_k": 2.0 * g.ecount() / n, "n_components": len(sizes)}


def s_gc_from_counts(word1, word2, counts, threshold):
    """S_GC directly from arrays -- called thousands of times in step 5."""
    keep = counts >= threshold
    if not keep.any():
        return {"N": 0, "E": 0, "N_GC": 0, "S_GC": np.nan, "mean_k": np.nan}
    g = ig.Graph.TupleList(zip(word1[keep], word2[keep]), directed=False)
    g.simplify()
    return giant_component_stats(g)


# =============================================================================
# THE SIX RESAMPLING SCHEMES
#
# Press volume grows across the decade and the giant component grows
# mechanically with volume, so every window is thinned to a common mass
# M_ref = 0.8 * min_v M_v, with retention fraction f_v = M_ref / M_v. Six
# schemes with complementary assumptions are used, so the result can be shown
# not to depend on any single resampling rule.
# =============================================================================
def resample(scheme, *, n, matrix, mass, mass_full, m_ref, m_ref_full, rng):
    if scheme == "raw":
        return n
    if scheme == "binomial":
        # (i) each co-occurrence kept with probability f_v (Steutel & van Harn 1979)
        return rng.binomial(n.astype(np.int64), m_ref / mass)
    if scheme == "hypergeometric":
        # (ii) M_ref drawn without replacement from the observed pool
        #      (Sanders 1968; Hurlbert 1971; Gotelli & Colwell 2001)
        good = n.astype(np.int64)
        return rng.hypergeometric(good, np.maximum(int(mass) - good, 0), int(m_ref))
    if scheme == "multinomial":
        # (iii) nonparametric bootstrap over occurrences (Efron 1979). The
        #       residual cell represents pairs removed by the pre-filter, so
        #       the retained pairs keep their correct probabilities.
        p = n.astype(np.float64) / float(mass)
        probabilities = np.append(p, max(1.0 - p.sum(), 0.0))
        return rng.multinomial(int(m_ref), probabilities / probabilities.sum())[:-1]
    if scheme == "betabinomial":
        # (iv) overdispersed thinning, E[pi] = f_v and ICC = rho, which
        #      accommodates the burstiness of words (Skellam 1948; Williams 1975)
        f = m_ref / mass
        scale = 1.0 / RHO - 1.0
        pi = rng.beta(f * scale, (1.0 - f) * scale, size=n.shape[0])
        return rng.binomial(n.astype(np.int64), pi)
    if scheme == "block":
        # (v) months resampled with replacement, preserving within-month
        #     dependence (Kuensch 1989; Politis & Romano 1994; Lahiri 2003)
        n_months = matrix.shape[1]
        drawn = rng.integers(0, n_months, size=n_months)
        return rng.binomial(matrix[:, drawn].sum(axis=1).astype(np.int64), m_ref / mass)
    if scheme == "binomial_full":
        # (vi) as (i), with retention fractions from the masses prior to the
        #      monthly pre-filter, correcting its mild over-thinning
        return rng.binomial(n.astype(np.int64), m_ref_full / mass_full)
    raise ValueError(scheme)


def make_rng(scheme, window, replicate):
    """Independent, reproducible generator per (scheme, window, replicate).

    Replicate r of every window shares the design, so replicates are paired
    across windows and can be differenced directly in step 6.
    """
    return np.random.default_rng([SEED, abs(hash((scheme, window))) % 2**32,
                                  replicate])


# =============================================================================
# STEP 4 -- THRESHOLD SWEEP, WITHOUT RESAMPLING
#
# S_GC is the order parameter of a percolation process: at low thresholds every
# network saturates, at high thresholds every one fragments. It is therefore
# reported as a curve over the grid, never at a single cut.
# =============================================================================
def step4_threshold_sweep(windows):
    banner("STEP 4  threshold sweep, without resampling")
    rows = [{"window": name, "w": w,
             **s_gc_from_counts(d["word1"], d["word2"], d["n"], w)}
            for name, d in windows.items() for w in THRESHOLDS]
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "04_threshold_sweep.csv", index=False)
    print(sweep[sweep["w"] == W_ILLUSTRATION][
        ["window", "N", "E", "N_GC", "S_GC", "mean_k"]].round(3).to_string(index=False))
    return sweep


# =============================================================================
# STEP 5 -- VOLUME-MATCHED RESAMPLING: 6 SCHEMES x 10 THRESHOLDS x 3 WINDOWS
# =============================================================================
def step5_resampling(windows):
    banner("STEP 5  volume-matched resampling")
    m_ref = MREF_FACTOR * min(d["mass"] for d in windows.values())
    m_ref_full = MREF_FACTOR * min(d["mass_full"] for d in windows.values())
    print(f"  M_ref = {m_ref:,.0f}")
    for name, d in windows.items():
        print(f"    f_{name:16s} = {m_ref / d['mass']:.3f}")

    rows, t0 = [], time.time()
    for scheme in SCHEME_ORDER:
        replicates = 1 if scheme == "raw" else N_REPLICATES
        for window in WINDOW_ORDER:
            d = windows[window]
            for r in range(replicates):
                counts = resample(scheme, n=d["n"], matrix=d["matrix"],
                                  mass=d["mass"], mass_full=d["mass_full"],
                                  m_ref=m_ref, m_ref_full=m_ref_full,
                                  rng=make_rng(scheme, window, r))
                for w in THRESHOLDS:
                    stats = s_gc_from_counts(d["word1"], d["word2"], counts, w)
                    rows.append({"scheme": scheme, "window": window,
                                 "replicate": r, "w": w, "S_GC": stats["S_GC"],
                                 "N": stats["N"], "mean_k": stats["mean_k"]})
        print(f"  {scheme:16s} done ({time.time() - t0:.0f}s)", flush=True)

    replicates_table = pd.DataFrame(rows)
    replicates_table.to_csv(OUT / "05_resampling_replicates.csv", index=False)
    low, high = (100 - CI_LEVEL) / 2, 100 - (100 - CI_LEVEL) / 2
    summary = (replicates_table.groupby(["scheme", "window", "w"], as_index=False)
               .agg(S_GC=("S_GC", "mean"),
                    lo=("S_GC", lambda s: np.percentile(s, low)),
                    hi=("S_GC", lambda s: np.percentile(s, high)),
                    mean_k=("mean_k", "mean"), N=("N", "mean")))
    summary.to_csv(OUT / "05_resampling_summary.csv", index=False)
    print(f"  {len(replicates_table):,} replicate rows written")
    return replicates_table, summary


# =============================================================================
# STEP 6 -- STATISTICAL TESTS
#
# (1) paired within-replicate differences, which cancel the shared Monte-Carlo
#     noise of the two draws; (2) whether the post-transition window holds the
#     largest giant component in every cell; (3) the mean degree, a
#     threshold-stable complement to S_GC.
# =============================================================================
def step6_tests(replicates_table):
    banner("STEP 6  statistical tests")
    low, high = (100 - CI_LEVEL) / 2, 100 - (100 - CI_LEVEL) / 2
    pairs = [("Post-transition", "Neutral"), ("Post-transition", "Pre-transition"),
             ("Pre-transition", "Neutral")]

    wide = replicates_table.pivot_table(index=["scheme", "w", "replicate"],
                                        columns="window", values="S_GC")
    rows = []
    for (scheme, w), block in wide.groupby(level=["scheme", "w"]):
        for a, b in pairs:
            d = (block[a] - block[b]).dropna().to_numpy()
            if d.size:
                rows.append({"scheme": scheme, "w": w, "contrast": f"{a} - {b}",
                             "n_replicates": d.size, "mean_difference": d.mean(),
                             "ci_lo": np.percentile(d, low) if d.size > 1 else np.nan,
                             "ci_hi": np.percentile(d, high) if d.size > 1 else np.nan,
                             "share_positive": float((d > 0).mean())})
    differences = pd.DataFrame(rows)
    differences.to_csv(OUT / "06_pairwise_differences.csv", index=False)

    means = (replicates_table.groupby(["scheme", "w", "window"], as_index=False)["S_GC"]
             .mean().pivot_table(index=["scheme", "w"], columns="window",
                                 values="S_GC").reset_index())
    means["post_is_highest"] = (means["Post-transition"]
                                >= means[["Neutral", "Pre-transition"]].max(axis=1))
    means["strictly_increasing"] = ((means["Neutral"] <= means["Pre-transition"])
                                    & (means["Pre-transition"] <= means["Post-transition"]))
    means.to_csv(OUT / "06_ordering_check.csv", index=False)

    degree = (replicates_table.groupby(["scheme", "w", "window"], as_index=False)["mean_k"]
              .mean().pivot_table(index=["scheme", "w"], columns="window",
                                  values="mean_k").reset_index())
    degree["ratio_post_over_neutral"] = degree["Post-transition"] / degree["Neutral"]
    degree.to_csv(OUT / "06_mean_degree_ratio.csv", index=False)

    informative = means[means["w"] <= 400]
    print(f"  post-transition highest in "
          f"{int(informative['post_is_highest'].sum())}/{len(informative)} "
          f"cells with w <= 400")
    focus = differences[(differences["w"] == W_ILLUSTRATION)
                        & (differences["contrast"] == "Post-transition - Neutral")]
    print(f"\n  paired difference post - neutral at w = {W_ILLUSTRATION}:")
    print(focus[["scheme", "mean_difference", "ci_lo", "ci_hi",
                 "share_positive"]].round(4).to_string(index=False))
    print("\n  mean-degree ratio post / neutral:")
    print(degree.groupby("scheme")["ratio_post_over_neutral"].mean().round(3).to_string())


# =============================================================================
# STEP 7 -- FIGURE: THE THREE SEMANTIC NETWORKS (main text)
#
# The three un-resampled networks at w = 200, drawn with the DrL force-directed
# layout (Martin et al. 2011). Fruchterman-Reingold, igraph's default, switches
# to a grid approximation above ~1000 nodes, collapsing the largest network into
# a blob and making the three panels incomparable.
# =============================================================================
def step7_network_figure(windows):
    banner("STEP 7  figure: the three semantic networks")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    rows = []
    for ax, window in zip(axes, WINDOW_ORDER):
        d = windows[window]
        keep = d["n"] >= W_ILLUSTRATION
        g = ig.Graph.TupleList(zip(d["word1"][keep], d["word2"][keep]),
                               directed=False)
        g.simplify()
        stats = giant_component_stats(g)
        components = g.connected_components()
        in_gc = np.asarray(components.membership) == int(np.argmax(components.sizes()))

        # igraph draws from a Python random.Random, so seeding it makes the
        # coordinates -- and hence the figure -- reproducible.
        ig.set_random_number_generator(random.Random(LAYOUT_SEED))
        layout = np.asarray(g.layout_drl().coords)

        segments = np.array([[layout[e.source], layout[e.target]] for e in g.es])
        if len(segments):
            ax.add_collection(LineCollection(segments, colors=EDGE_COLOR,
                                             linewidths=0.25, alpha=0.5, zorder=1))
        ax.scatter(layout[~in_gc, 0], layout[~in_gc, 1], s=14,
                   c=OTHER_NODE_COLOR, linewidths=0, zorder=2)
        ax.scatter(layout[in_gc, 0], layout[in_gc, 1], s=14,
                   c=GC_NODE_COLOR, linewidths=0, zorder=3)
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.autoscale_view()
        rows.append({"window": window, "w": W_ILLUSTRATION, **stats})
        print(f"  {window:16s} N={stats['N']:5d}  E={stats['E']:5d}  "
              f"S_GC={stats['S_GC']:.4f}")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.08, wspace=0.02)
    # Labels in figure coordinates so all three sit on the same baseline: an
    # equal aspect ratio gives each network its own bounding box.
    for ax, window in zip(axes, WINDOW_ORDER):
        box = ax.get_position()
        fig.text(box.x0 + box.width / 2, 0.035, WINDOW_LABELS[window],
                 ha="center", va="center", fontsize=15, color="#333333")
    fig.savefig(FIG / "fig_networks_w200.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_networks_w200.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "07_network_stats.csv", index=False)
    print("  figures/fig_networks_w200.pdf written")


# =============================================================================
# STEP 8 -- FIGURE: ROBUSTNESS PANEL (Supplementary Information)
#
# Seven panels -- raw plus one per scheme -- of S_GC against the threshold on a
# log axis, with 95% percentile bands over the replicates.
# =============================================================================
def step8_robustness_figure(summary):
    banner("STEP 8  figure: robustness panel")
    xticks = [25, 50, 100, 200, 400, 800]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5))
    flat = axes.ravel()
    for ax, scheme in zip(flat, SCHEME_ORDER):
        for window in WINDOW_ORDER:
            sub = summary[(summary["scheme"] == scheme)
                          & (summary["window"] == window)].sort_values("w")
            ax.plot(sub["w"], sub["S_GC"], marker=MARKERS[window],
                    color=COLORS[window], linewidth=1.8, markersize=6)
            if scheme != "raw":
                ax.fill_between(sub["w"], sub["lo"], sub["hi"],
                                color=COLORS[window], alpha=0.18, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks(xticks)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(0, 1.02)
        ax.set_title(SCHEME_LABELS[scheme], fontsize=15)
        ax.set_xlabel("Co-occurrence threshold $w$ (log scale)", fontsize=12)
        ax.set_ylabel("$S_1$ = GC / N", fontsize=13)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=11)

    legend_ax = flat[len(SCHEME_ORDER)]
    legend_ax.set_axis_off()
    handles = [Line2D([], [], color=COLORS[w], marker=MARKERS[w], linewidth=1.8,
                      markersize=7, label=WINDOW_LEGEND[w]) for w in WINDOW_ORDER]
    handles.append(Patch(facecolor="0.75", alpha=0.35,
                         label=f"Shaded band: {CI_LEVEL}% CI "
                               f"({N_REPLICATES} replicates)"))
    legend_ax.legend(handles=handles, loc="center", frameon=False, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG / "fig_gc_robustness.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_gc_robustness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  figures/fig_gc_robustness.pdf written")


# =============================================================================
# STEP 9 -- ARE THE RESULTS CORRECT?
#
# Each check compares a value just computed against the one reported in the
# paper. The w = 200 networks are deterministic and must match exactly; the
# resampling results are compared within a tolerance, being averages over 25
# random replicates.
# =============================================================================
def step9_verify():
    banner("STEP 9  are the results correct?")

    def read(path):
        with open(OUT / path, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    checks = []

    def check(name, ok, found, expected):
        checks.append(ok)
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name:50s} {found:>20s}   "
              f"expected {expected}")

    expected_at_200 = {"Neutral": (965, 0.589), "Pre-transition": (1276, 0.654),
                       "Post-transition": (1896, 0.765)}
    print("Networks at w = 200, without resampling\n")
    for row in [r for r in read("04_threshold_sweep.csv") if r["w"] == "200"]:
        nodes, s_gc = int(row["N"]), float(row["S_GC"])
        want_n, want_s = expected_at_200[row["window"]]
        check(f"{row['window']}: number of nodes", nodes == want_n,
              f"N = {nodes}", f"N = {want_n}")
        check(f"{row['window']}: giant component", abs(s_gc - want_s) < 0.001,
              f"S_GC = {s_gc:.3f}", f"S_GC = {want_s:.3f}")

    print("\nRobustness across the six resampling schemes\n")
    informative = [r for r in read("06_ordering_check.csv") if int(r["w"]) <= 400]
    highest = sum(1 for r in informative if r["post_is_highest"] in ("True", "true"))
    check("post-transition highest, w <= 400", highest == len(informative),
          f"{highest}/{len(informative)} cells", f"{len(informative)}/{len(informative)}")

    differences = [r for r in read("06_pairwise_differences.csv")
                   if r["w"] == "200" and r["scheme"] != "raw"
                   and r["contrast"] == "Post-transition - Neutral"]
    check("paired difference positive in every replicate",
          all(float(r["share_positive"]) == 1.0 for r in differences),
          f"{len(differences)} schemes", "all replicates")
    lowest = min(float(r["ci_lo"]) for r in differences)
    check("95% intervals exclude zero", lowest > 0,
          f"lowest bound {lowest:.3f}", "> 0")

    ratios = [float(r["ratio_post_over_neutral"])
              for r in read("06_mean_degree_ratio.csv") if r["scheme"] != "raw"]
    average = sum(ratios) / len(ratios)
    check("mean-degree ratio post / neutral", 1.05 <= average <= 1.20,
          f"{average:.3f}", "between 1.05 and 1.20")

    print("\n" + "=" * 74)
    print(f"ALL {len(checks)} CHECKS PASSED -- the outputs reproduce the "
          f"published results" if all(checks)
          else f"{checks.count(False)} of {len(checks)} checks failed")
    print("=" * 74)
    return all(checks)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Semantic networks and the giant component -- full analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reuse-windows", action="store_true",
                        help="skip steps 1-3 and reuse the aggregated counts in "
                             "data/windows.npz from an earlier run; without it "
                             "the analysis is rebuilt from the raw corpus")
    parser.add_argument("--no-stopwords", action="store_true",
                        help="build bigrams directly from cuerpo_limpio, without "
                             "removing function words (changes the results)")
    parser.add_argument("--corpus", metavar="PATH",
                        help="path to the raw corpus; by default data/news.csv "
                             "or data/news.csv.gz")
    parser.add_argument("--replicates", type=int, default=N_REPLICATES,
                        help=f"replicates per cell (default {N_REPLICATES})")
    arguments = parser.parse_args()

    globals()["N_REPLICATES"] = arguments.replicates
    if arguments.no_stopwords:
        globals()["STOPWORDS"] = frozenset()

    for folder in (DATA, OUT, FIG):
        folder.mkdir(exist_ok=True)

    print(f"running in {sys.executable}")
    print(f"numpy {np.__version__} | pandas {pd.__version__} | "
          f"igraph {ig.__version__} | matplotlib {matplotlib_version()}")

    started = time.time()
    aggregated = DATA / "windows.npz"
    if arguments.reuse_windows:
        if not aggregated.exists():
            sys.exit(f"{aggregated} not found -- run without --reuse-windows "
                     f"to rebuild it from the corpus")
        print(f"\nreusing {aggregated} (drop --reuse-windows to rebuild it)")
    else:
        articles = step1_parse_corpus(find_corpus(arguments.corpus))
        step2_monthly_bigrams(articles)
        step3_windows()

    windows = load_windows()
    step4_threshold_sweep(windows)
    replicates_table, summary = step5_resampling(windows)
    step6_tests(replicates_table)
    step7_network_figure(windows)
    step8_robustness_figure(summary)
    ok = step9_verify()

    banner(f"done in {time.time() - started:.0f}s")
    print(f"  outputs/   numerical results (CSV)")
    print(f"  figures/   fig_networks_w200 and fig_gc_robustness (PDF + PNG)")
    return 0 if ok else 1


def matplotlib_version():
    import matplotlib
    return matplotlib.__version__


if __name__ == "__main__":
    sys.exit(main())
