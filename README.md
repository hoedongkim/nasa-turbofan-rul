[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hoedongkim/nasa-turbofan-rul/HEAD?labpath=NASA_Turbofan_Jet_Engine_EDA_Predictive_Model.ipynb)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# NASA Turbofan RUL Prediction

**Predicting how many more flights a jet engine has left, from its sensor readings.**

*Solo project — NASA C-MAPSS FD001 dataset.*

[![Fleet health dashboard](assets/dashboard.png)](https://public.tableau.com/app/profile/hoedong.kim/viz/EngineHealthMaintenancePriority/Dashboard1)

*[Open the interactive dashboard →](https://public.tableau.com/app/profile/hoedong.kim/viz/EngineHealthMaintenancePriority/Dashboard1)*

---

## 🎯 Problem

Aircraft engines are serviced on fixed schedules. Service too early and you throw
away usable life; too late and the engine fails in the air. Either way the
schedule is a guess, because it is built around the *average* engine rather than
the one actually on the wing.

A better approach is to ask each engine directly: **how many more flights can you
do?** That number is called **RUL — Remaining Useful Life**, and predicting it
from sensor data is the goal of this project.

## 📊 Data

[NASA C-MAPSS](https://www.nasa.gov/intelligent-systems-division/), subset FD001 —
simulated run-to-failure records for a fleet of turbofan engines.

|  | Engines | Rows | Note |
|---|---|---|---|
| Training | 100 | 20,631 | Each engine flown until it fails |
| Test | 100 | 13,096 | Recording stops *before* failure |

Every row is one flight cycle: **21 sensor readings** (temperatures, pressures,
fan and core speeds, fuel and cooling flows) plus 3 operating settings.

**Target variable:** RUL — the number of flight cycles remaining before failure.
It is **capped at 125** for training, because very early in an engine's life the
sensors look identical whether 200 or 300 flights remain. Capping lets the model
spend its capacity on the wear-down phase, where the signal actually lives. Test
scores are measured against the true, uncapped RUL.

## 🔍 Approach

**1 — Narrow 21 sensors down to 12.** Seven columns turned out to be perfectly
constant, carrying no information at all. Three more (`P15`, `Nc`, `NRc`) were
removed after EDA flagged them (see *Key Insights*), along with two operating
settings that barely correlate with RUL.

**2 — Scale.** Sensor ranges differ by orders of magnitude (`T30` ≈ 1,590 vs
`BPR` ≈ 8.4), so all 12 were min-max scaled to [0, 1]. The scaler is fitted on
training data only and then applied to test — no peeking.

**3 — Compare five models**, chosen to answer one question: *does reading a
sequence beat reading a single snapshot?*

| Model | How it reads the data |
|---|---|
| **XGBoost** | One cycle at a time (baseline). Given 10-cycle rolling mean/std/max so it has some history |
| **LSTM** | 30-cycle window, remembers across the sequence |
| **GRU** | 30-cycle window, lighter recurrent variant |
| **1D CNN** | 30-cycle window, detects local patterns |
| **CNN-LSTM** | CNN extracts local patterns, LSTM tracks them over time |

**4 — Score on four metrics**, because no single one tells the whole story:

| Metric | Why it is here |
|---|---|
| **RMSE** | Punishes large misses heavily; the standard used in C-MAPSS literature, so results are comparable |
| **MAE** | Reads directly as "off by N flights on average" |
| **R²** | How much of the variation in RUL the model explains |
| **NASA Score** | The domain metric, **asymmetric on purpose**: claiming an engine has *more* life than it does is punished far harder than claiming it has less — one wastes a maintenance slot, the other puts a failing engine in the air |

**5 — Re-run everything under three random seeds** to check whether the ranking
is real or an accident of initialisation. This turned out to matter — see
*Takeaways*.

## 📈 Results

Mean ± standard deviation across seeds 42, 43 and 44:

| Model | RMSE | MAE | R² | NASA Score | Train time |
|---|---|---|---|---|---|
| XGBoost | 19.83 ± 0.00 | 14.73 ± 0.00 | 0.772 | 1305.1 | 1 s |
| LSTM | 15.08 ± 0.61 | 11.43 ± 0.57 | 0.868 | 410.8 | 102–162 s |
| GRU | 15.07 ± 0.04 | 11.30 ± 0.17 | 0.869 | 429.1 | 96–162 s |
| 1D CNN | 14.99 ± 0.20 | **11.13 ± 0.27** | 0.870 | **369.4** | 16–29 s |
| CNN-LSTM | **14.87 ± 0.10** | 11.37 ± 0.53 | **0.872** | 385.0 | 32–42 s |

*Bold marks the best mean in each column — no model leads on all four.*

![Model comparison](assets/model_comparison.png)

![Predicted vs actual](assets/predicted_vs_actual.png)

The best models land within roughly **15 flight cycles** of the truth. The four
sequence models beat the tabular baseline by about **5 RMSE**, and that gap holds
in every seed.

## 💡 Key Insights

**1 — The physics is visible in the raw sensors.**

![Sensor degradation](assets/sensor_degradation.png)

As engines wear out, **temperatures climb** (`T24`, `T30`, `T50`) while
**pressures and fuel flow fall** (`P30`, `phi`, `W31`, `W32`). That is exactly
what a degrading compressor does: it squeezes air less efficiently, so downstream
heat rises and pressure drops. The dataset simulates compressor wear, and the
simulation is faithful enough that the mechanism shows up plainly in the plots —
which is also why a model can learn it.

**2 — The most suspicious sensors were the ones that looked busiest.**

Core speed `Nc` and `NRc` — the two panels that fan out dramatically after ~150
cycles — look like the most dynamic signals on the page. They are traps. Three
independent checks agreed:

- **Correlation** — the two are 0.96 correlated with each other, so one is redundant
- **Stationarity** — flat until ~cycle 150, then the spread explodes for a handful of engines while the rest stay put
- **Outliers** — ~8% outlier rate, four times higher than any sensor that was kept

Meanwhile `P15` (top right) is a flat line: it moves between 21.60 and 21.61 for
the entire life of every engine. A sensor that never changes cannot tell you
anything about change. All three were dropped.

## 🧭 What This Would Mean in Practice

Only what the measured error supports — this is a public research dataset, so
there is no deployment and no real cost saving to report.

With **MAE ≈ 11 cycles** and **RMSE ≈ 15**, a maintenance window opened roughly
**25–30 flights ahead of the predicted failure** would cover the large majority of
engines while still recovering most of the remaining life. The NASA Score is the
metric that would set that margin in practice, since it already encodes the
asymmetry: an over-optimistic prediction is the expensive kind of wrong.

## 📌 Takeaways

**A single training run cannot rank models that finish this close.** The first
version of this project concluded CNN-LSTM was the winner, beating GRU by 0.06
RMSE. Re-running under three seeds showed that gap was noise: LSTM alone swings
±0.61 between seeds, and at seed 42 the 1D CNN comes first instead. The four
sequence models are a tie, and separating them would take roughly 15 seeds, not 3.

**What the experiment does establish is the comparison it was built for.**
Sequence models beat the tabular baseline by ~5 RMSE, in every seed. Reading the
*trend* in sensor readings, not just their level, is what predicts remaining life.

**When accuracy ties, cost decides.** The 1D CNN trains 4–6× faster than LSTM or
GRU for statistically indistinguishable accuracy — a far better basis for choosing
than the second decimal place of RMSE.

## 📁 Repository Structure

```
├── NASA_Turbofan_Jet_Engine_EDA_Predictive_Model.ipynb   Main analysis (outputs included)
├── cmapss_db.py          Data loader — Neon Postgres, or CMaps/*.txt as fallback
├── seed_study.py         Re-runs every model across seeds -> seed_study.csv
├── seed_study.csv        Per-model, per-seed metrics behind the error bars
├── upload_to_neon.py     One-off loader: raw text files -> Postgres
├── db_schema.sql         Database schema
├── export_predictions.py Trains the two dashboard models, writes predictions
├── dashboard_views.sql   Views the dashboard reads
├── dashboard/            Flattened predictions the Tableau workbook loads
├── CMaps/                Raw NASA C-MAPSS data
└── assets/               Figures used in this README
```

## 🖥 Fleet Health Dashboard

**[→ Open the live dashboard on Tableau Public](https://public.tableau.com/app/profile/hoedong.kim/viz/EngineHealthMaintenancePriority/Dashboard1)**

A maintenance view built on the same predictions — not model metrics, but which
engine needs attention and when. `export_predictions.py` trains CNN-LSTM and
1D CNN, scores every test engine at every cycle, and writes the results to
Postgres plus `dashboard/rul_predictions.csv`.

Engines are banded by predicted remaining life. The 30-cycle red line is set
from the measured error, not picked for looks: test RMSE is ~15 cycles, so it
leaves roughly two standard errors of margin.

| Band | Engines (CNN-LSTM) | Mean error |
|---|---|---|
| Critical (≤30 cycles) | 18 | 5.2 cycles |
| Warning (31–60) | 14 | 8.6 cycles |
| Healthy (>60) | 68 | 14.3 cycles |

Accuracy is highest in the band where a decision actually gets made — a
consequence of capping the training target at 125 cycles, which concentrates
learning on the degradation phase.

"Now" is each engine's last recorded cycle, which is exactly the point the model
was evaluated at, so the dashboard and the notebook report the same numbers.

**What it shows.** Four panels, none of them model metrics: fleet counts by risk
band, an alert list ranked by urgency with a plain-language action per engine
(*Ground now* / *Schedule maintenance*), a scatter placing all 100 engines by
flights flown against flights remaining, and a per-engine trajectory that appears
when you click any engine. The alert and warning thresholds are exposed as
controls, so a reader can move the 30-cycle line and watch the counts change.

The grey line on the trajectory chart is the ground truth, shown for validation —
in production only the prediction exists.

**On the data connection.** The pipeline is Postgres-backed end to end:
`export_predictions.py` writes predictions to Neon and `dashboard_views.sql`
defines the two views the dashboard consumes. Tableau Public cannot hold a live
database connection — it has nowhere to store credentials — so the workbook reads
a CSV exported from those views. Pointing Tableau Desktop or Looker Studio at
`cmapss.v_fleet_dashboard` gives the same dashboard over a live connection.

## 🚀 Run the Project

**Just read it** — the notebook is committed with all outputs, so every figure and
number renders on GitHub without running anything.

**In the browser** — click the Binder badge at the top. Binder takes 5–15 minutes
to build the first time. EDA runs quickly; training the four neural networks on
Binder's free tier is slow (20–30 min) and may run out of memory.

**Locally:**

```bash
git clone <this repo> && cd <this repo>
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab
```

No database or credentials needed — `cmapss_db.py` reads `CMaps/*.txt` when
`DATABASE_URL` is unset.

**With Postgres (optional).** The analysis was authored against a
[Neon](https://neon.tech) database. To reproduce that setup, put a connection
string in `.env` and load the data:

```bash
cp .env.example .env     # then paste your connection string
python upload_to_neon.py
```

`cmapss_db.py` switches to the database automatically. Both paths return identical
frames — verified value-for-value across all four C-MAPSS sub-datasets — so results
never depend on which source is active.

**Reproducing the seed study:**

```bash
python seed_study.py                  # seeds 42, 43, 44 -> seed_study.csv
python seed_study.py --seeds 42 43 44 45 46
```

Every model is seeded from the `SEED` constant in the notebook's setup cell, and
all metrics are collected programmatically — no number in the tables or charts is
transcribed by hand.

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?logo=xgboost&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=python&logoColor=white)

## License

[MIT](LICENSE)
