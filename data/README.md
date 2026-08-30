# Input data

Critical Transition Index (CTI) series at daily and weekly resolution; the
scripts select the resolution through the filter columns.

| File | Date column | Value column | Filters used |
|---|---|---|---|
| `serie1.csv` | `fecha` | `indice_cti` | `Temporalidad` ∈ {`Semanal`, `Diario`} |
| `serie2.csv` | `time` | `Value` | `Method` = `Original`; `Frequency` ∈ {`Weekly`, `Daily`} |
| `serie3.csv` | `time` | `Value` | `Method` = `Original`; `Frequency` ∈ {`Weekly`, `Daily`} |

Series 1: security, education, health, work, and politics. Series 2: crime,
students, hospitals, wages, and government. Series 3: presidential figures.
