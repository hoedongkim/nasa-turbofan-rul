"""Train the dashboard models and publish RUL predictions.

Tableau cannot run a neural network, so predictions are computed once here and
written to two tables:

    cmapss.fleet_status  one row per engine x model — the current snapshot that
                         drives the summary cards and the alert list
    cmapss.rul_history   one row per engine x cycle x model — the trajectory
                         behind the per-engine drill-down

The two are then flattened into `dashboard/rul_predictions.csv`, which is what
the Tableau Public workbook actually loads. Tableau Public cannot connect to
PostgreSQL, and one wide file means the workbook needs no relationships: the
snapshot sheets filter on `is_latest`, the trajectory sheet does not.

**What "now" means.** C-MAPSS is a finished simulation, so there is no wall
clock. Each test engine's recording simply stops at a different cycle, and that
stopping point is treated as the present: engine 1 has flown 31 cycles, engine 7
has flown 160. That is exactly the point the model was evaluated at, so the
dashboard and the notebook report the same numbers. No calendar is invented.

Usage:
    python export_predictions.py
    python export_predictions.py --dataset FD001
"""

from __future__ import annotations

import argparse
import os
import pathlib

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import psycopg
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

import cmapss_db as db
# Architectures are imported rather than re-typed so the dashboard cannot drift
# away from the models the notebook and the seed study evaluated.
from seed_study import (DROP_COLS, RUL_CAP, SENSOR_COLS, WINDOW_SEQ,
                        build_cnn, build_cnn_lstm, create_sequences, evaluate)

SEED = 42
MODELS = {"CNN-LSTM": build_cnn_lstm, "1D CNN": build_cnn}

# Alert thresholds, in flight cycles of remaining life. Grounded in the measured
# error rather than picked for looks: test RMSE is ~15 cycles, so a 30-cycle red
# band leaves roughly two standard errors of margin before actual failure.
CRITICAL, WARNING = 30, 60

SCHEMA = """
CREATE TABLE IF NOT EXISTS cmapss.fleet_status (
    dataset       TEXT NOT NULL,
    model         TEXT NOT NULL,
    unit_number   INT  NOT NULL,
    current_cycle INT  NOT NULL,
    predicted_rul REAL NOT NULL,
    true_rul      INT  NOT NULL,
    abs_error     REAL NOT NULL,
    risk_band     TEXT NOT NULL,
    PRIMARY KEY (dataset, model, unit_number)
);

CREATE TABLE IF NOT EXISTS cmapss.rul_history (
    dataset       TEXT NOT NULL,
    model         TEXT NOT NULL,
    unit_number   INT  NOT NULL,
    time_cycles   INT  NOT NULL,
    predicted_rul REAL NOT NULL,
    true_rul      INT  NOT NULL,
    risk_band     TEXT NOT NULL,
    PRIMARY KEY (dataset, model, unit_number, time_cycles)
);
"""


def risk_band(rul: float) -> str:
    """Sorts correctly in Tableau because of the leading digit."""
    if rul <= CRITICAL:
        return "1 Critical"
    if rul <= WARNING:
        return "2 Warning"
    return "3 Healthy"


def prepare(dataset: str):
    """Notebook sections 3.1-4, reproduced: drop constants, scale, cap the target."""
    train, test, _ = db.load_notebook_frames(dataset)
    constant = [c for c in train.columns if train[c].std() < 1e-10]
    train = train.drop(columns=constant).drop(columns=DROP_COLS)
    test = test.drop(columns=constant).drop(columns=DROP_COLS)

    scaler = MinMaxScaler()
    train[SENSOR_COLS] = scaler.fit_transform(train[SENSOR_COLS])
    test[SENSOR_COLS] = scaler.transform(test[SENSOR_COLS])
    train["RUL"] = train["RUL"].clip(upper=RUL_CAP)
    return train, test


def every_cycle_windows(test: pd.DataFrame):
    """One full 30-cycle window per engine per cycle, for the drill-down curve.

    Cycles before the window is full are skipped rather than zero-padded. The
    notebook pads only for engines shorter than the window, which never happens
    in FD001 (the shortest test engine has 31 cycles), so padding here would
    feed the model rows of zeros it never saw in training. It shows: mean error
    below cycle 30 was 35.6 for 1D CNN and 14.9 for CNN-LSTM, against 11.2 and
    10.0 above it. A model that needs 30 cycles of history should say nothing
    until it has them.
    """
    X, keys = [], []
    for unit, grp in test.groupby("unit_number"):
        grp = grp.sort_values("time_in_cycles")
        data = grp[SENSOR_COLS].values
        for i, cycle in enumerate(grp["time_in_cycles"].values):
            if i + 1 < WINDOW_SEQ:          # not enough history yet
                continue
            X.append(data[i - WINDOW_SEQ + 1: i + 1])
            keys.append((unit, int(cycle)))
    return np.array(X), pd.DataFrame(keys, columns=["unit_number", "time_cycles"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="FD001")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    load_dotenv()
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL is not set — this script writes to Neon.")
        return 1

    train, test = prepare(args.dataset)
    X_train, y_train = create_sequences(train, SENSOR_COLS, WINDOW_SEQ)
    X_all, keys = every_cycle_windows(test)
    print(f"train windows {X_train.shape} | prediction points {X_all.shape}", flush=True)

    # "Now" for each engine is its last recorded cycle.
    last = test.groupby("unit_number")["time_in_cycles"].max().rename("current_cycle")
    final = db.load_rul(args.dataset).set_index("unit_number")["rul"].rename("final_rul")
    keys = keys.join(pd.concat([last, final], axis=1), on="unit_number")
    # True remaining life at this cycle = cycles left in the record + final RUL
    keys["true_rul"] = (keys.current_cycle - keys.time_cycles) + keys.final_rul

    history, status = [], []
    for name, builder in MODELS.items():
        tf.keras.utils.set_random_seed(SEED)
        model = builder(X_train.shape[1:])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        hist = model.fit(X_train, y_train, epochs=args.epochs, batch_size=64,
                         validation_split=0.2, verbose=0,
                         callbacks=[EarlyStopping(monitor="val_loss", patience=10,
                                                  restore_best_weights=True)])

        df = keys.copy()
        df["model"] = name
        df["predicted_rul"] = model.predict(X_all, verbose=0).flatten()
        df["risk_band"] = df.predicted_rul.map(risk_band)
        history.append(df)

        today = df[df.time_cycles == df.current_cycle].copy()
        today["abs_error"] = (today.predicted_rul - today.true_rul).abs()
        status.append(today)

        m = evaluate(today.true_rul.values, today.predicted_rul.values)
        print(f"{name:9s} {len(hist.history['loss']):3d} epochs | "
              f"RMSE {m['RMSE']:6.3f} MAE {m['MAE']:6.3f} | "
              f"critical {int((today.risk_band == '1 Critical').sum()):3d} engines", flush=True)
        tf.keras.backend.clear_session()

    history = pd.concat(history, ignore_index=True)
    status = pd.concat(status, ignore_index=True)
    history["dataset"] = status["dataset"] = args.dataset

    stat_cols = ["dataset", "model", "unit_number", "current_cycle",
                 "predicted_rul", "true_rul", "abs_error", "risk_band"]
    hist_cols = ["dataset", "model", "unit_number", "time_cycles",
                 "predicted_rul", "true_rul", "risk_band"]

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA)
            for table in ("fleet_status", "rul_history"):
                cur.execute(f"DELETE FROM cmapss.{table} WHERE dataset = %s", (args.dataset,))
        for table, df, cols in (("fleet_status", status, stat_cols),
                                ("rul_history", history, hist_cols)):
            with conn.cursor() as cur, cur.copy(
                    f"COPY cmapss.{table} ({', '.join(cols)}) FROM STDIN") as copy:
                for row in df[cols].itertuples(index=False, name=None):
                    copy.write_row(row)
            print(f"  cmapss.{table}: {len(df):,} rows", flush=True)
        conn.commit()

    write_dashboard_csv(args.dataset)
    return 0


def write_dashboard_csv(dataset: str) -> None:
    """Flatten both tables into the single file the Tableau workbook loads.

    Engine-level columns are suffixed `_now` and repeated on every cycle, so one
    file serves both the current-status sheets (filter `is_latest`) and the
    per-engine trajectory sheet (no filter).
    """
    df = db.query("""
        SELECT t.model, t.unit_number, t.time_cycles,
               t.predicted_rul, t.true_rul, t.risk_band,
               f.current_cycle, f.urgency_rank,
               f.predicted_rul AS predicted_rul_now,
               f.true_rul      AS true_rul_now,
               f.risk_band     AS risk_band_now,
               f.action        AS action_now,
               f.pct_life_used AS pct_life_used_now,
               f.abs_error     AS abs_error_now,
               t.is_latest
        FROM cmapss.v_rul_trajectory t
        JOIN cmapss.v_fleet_dashboard f USING (dataset, model, unit_number)
        WHERE t.dataset = :dataset
        ORDER BY t.model, t.unit_number, t.time_cycles
    """, {"dataset": dataset})

    # The stored values carry a sort prefix ("1 Critical"); strip it so Tableau
    # legends read cleanly. Ordering is handled by `urgency_rank` instead.
    for col in ("risk_band", "risk_band_now"):
        df[col] = df[col].str.slice(2)

    out = pathlib.Path(__file__).parent / "dashboard" / "rul_predictions.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  {out.relative_to(out.parent.parent)}: {len(df):,} rows x {df.shape[1]} cols "
          f"({out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
