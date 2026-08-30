# Builds the Critical Transition Index (CTI) from the raw news corpus.
#
# Input:  data/news.csv with columns fecha, fuente, cuerpo_limpio
#         (the raw corpus is not distributed with this repository).
# Output: data/serie1.csv, data/serie2.csv, data/serie3.csv at daily and
#         weekly resolution, in the schema described in data/README.md.
#
# For each source and period, the share of articles containing at least one
# series keyword is z-scored within the source over the full sample, averaged
# across sources, and rescaled to a base-100 mean: CTI = 100 + mean(z).

# -------------------------------------------------------------------------
# 0. Libraries
# -------------------------------------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  dplyr,
  stringr,
  lubridate,
  rio,
  tidyr,
  EWSmethods
)

# -------------------------------------------------------------------------
# 1. Keyword series
# -------------------------------------------------------------------------
series_keywords <- list(
  Serie_1 = c("seguridad", "educación", "salud", "trabajo", "política"),
  Serie_2 = c("delincuencia", "estudiantes", "hospital", "sueldo", "gobierno"),
  Serie_3 = c("lagos", "bachelet", "boric", "kast", "piñera")
)

# -------------------------------------------------------------------------
# 2. Import data
# -------------------------------------------------------------------------
media_data_raw <- import("data/news.csv") %>%
  mutate(date = as.Date(fecha, format = "%Y-%m-%d"))

# -------------------------------------------------------------------------
# 3. Keyword counts per article
# -------------------------------------------------------------------------
media_data <- media_data_raw %>%
  mutate(
    n_keywords_serie1 = str_count(cuerpo_limpio, paste(series_keywords$Serie_1, collapse = "|")),
    n_keywords_serie2 = str_count(cuerpo_limpio, paste(series_keywords$Serie_2, collapse = "|")),
    n_keywords_serie3 = str_count(cuerpo_limpio, paste(series_keywords$Serie_3, collapse = "|"))
  )

# -------------------------------------------------------------------------
# 4. CTI: per-source z-scored shares, averaged, base 100
# -------------------------------------------------------------------------
calculate_cti <- function(data, freq_unit, keyword_col) {

  step1_counts <- data %>%
    mutate(period_date = floor_date(date, unit = freq_unit)) %>%
    group_by(period_date, fuente) %>%
    summarise(
      total_articles        = n(),
      articles_with_keyword = sum(.data[[keyword_col]] > 0),
      .groups = "drop"
    ) %>%
    mutate(ratio = articles_with_keyword / total_articles)

  step2_stats <- step1_counts %>%
    group_by(fuente) %>%
    summarise(
      mean_fuente = mean(ratio, na.rm = TRUE),
      sd_fuente   = sd(ratio, na.rm = TRUE),
      .groups = "drop"
    )

  final_index <- step1_counts %>%
    left_join(step2_stats, by = "fuente") %>%
    mutate(
      normalized_ratio = if_else(
        sd_fuente == 0 | is.na(sd_fuente),
        0,
        (ratio - mean_fuente) / sd_fuente
      )
    ) %>%
    group_by(period_date) %>%
    summarise(mean_normalized = mean(normalized_ratio, na.rm = TRUE), .groups = "drop") %>%
    mutate(CTI = 100 + mean_normalized) %>%
    rename(date = period_date)

  return(final_index)
}

# -------------------------------------------------------------------------
# 5. Compute CTI for all series (weekly and daily)
# -------------------------------------------------------------------------
weekly_serie1 <- calculate_cti(media_data, "week", "n_keywords_serie1")
daily_serie1  <- calculate_cti(media_data, "day",  "n_keywords_serie1")

weekly_serie2 <- calculate_cti(media_data, "week", "n_keywords_serie2")
daily_serie2  <- calculate_cti(media_data, "day",  "n_keywords_serie2")

weekly_serie3 <- calculate_cti(media_data, "week", "n_keywords_serie3")
daily_serie3  <- calculate_cti(media_data, "day",  "n_keywords_serie3")

# -------------------------------------------------------------------------
# 6. Series 1 output schema: fecha, promedio_normalizado, indice_cti,
#    Data, Temporalidad
# -------------------------------------------------------------------------
format_serie1 <- function(data, temporalidad) {
  data %>%
    select(fecha = date, promedio_normalizado = mean_normalized, indice_cti = CTI) %>%
    mutate(
      Data = "Medios",
      Temporalidad = temporalidad
    )
}

daily_serie1_out  <- format_serie1(daily_serie1, "Diario")
weekly_serie1_out <- format_serie1(weekly_serie1, "Semanal")

# -------------------------------------------------------------------------
# 7. Series 2 & 3 output schema: time, Method, Value, Data, Frequency.
#    Detrended variants via EWSmethods::detrend_ts (the analyses use the
#    "Original" rows).
# -------------------------------------------------------------------------
detrend_methods <- list(
  list(method = "linear"),
  list(method = "loess", span = 0.1),
  list(method = "loess", span = 0.25),
  list(method = "first.difference")
)

apply_detrending <- function(data, method, bandwidth = NULL, span = NULL, degree = 2) {
  detrend_ts(
    data      = data,
    method    = method,
    bandwidth = bandwidth,
    span      = span,
    degree    = degree
  )
}

format_series_2_3 <- function(data, frequency_label) {

  data_adjusted <- data.frame(
    time = as.Date(data$date),
    V1   = as.numeric(data$CTI),
    V2   = as.numeric(data$CTI)
  )

  results <- lapply(detrend_methods, function(params) {
    do.call(apply_detrending, c(list(data = data_adjusted), params))
  })
  method_names <- c("Linear", "Loess (0.1)", "Loess (0.25)", "First Difference")
  names(results) <- method_names

  combined <- data.frame(
    time = data_adjusted$time,
    Original = data_adjusted$V1
  )

  for (name in method_names) {
    combined[[name]] <- results[[name]]$V1
  }

  combined %>%
    pivot_longer(-time, names_to = "Method", values_to = "Value") %>%
    mutate(
      Data = "Media",
      Frequency = frequency_label
    ) %>%
    select(time, Method, Value, Data, Frequency)
}

daily_serie2_out  <- format_series_2_3(daily_serie2, "Daily")
weekly_serie2_out <- format_series_2_3(weekly_serie2, "Weekly")

daily_serie3_out  <- format_series_2_3(daily_serie3, "Daily")
weekly_serie3_out <- format_series_2_3(weekly_serie3, "Weekly")

# -------------------------------------------------------------------------
# 8. Export in the repository schema
# -------------------------------------------------------------------------
export(bind_rows(daily_serie1_out, weekly_serie1_out), "data/serie1.csv")
export(bind_rows(daily_serie2_out, weekly_serie2_out), "data/serie2.csv")
export(bind_rows(daily_serie3_out, weekly_serie3_out), "data/serie3.csv")

cat("\nExported: data/serie1.csv, data/serie2.csv, data/serie3.csv\n")
