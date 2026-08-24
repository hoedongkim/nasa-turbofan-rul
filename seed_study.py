"""Re-run every model across several random seeds and report mean +/- std.

The notebook trains each model once, at SEED = 42. The top three models finish
within ~0.4 RMSE of each other, which is small enough that the ranking could be
an artifact of one lucky initialisation. This script re-runs the same pipeline
under several seeds so the comparison table can carry error bars.

    python seed_study.py                  # seeds 42, 43, 44
    python seed_study.py --seeds 42 43 44 45 46
    python seed_study.py --epochs 5       # quick smoke test

Writes seed_study.csv (one row per model x seed); the notebook's section 6
reads that file. Keep the architectures here in sync with notebook sections
5.1-5.5 — they are duplicated on purpose so this script can run headless.
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    LSTM, GRU, Conv1D, Dense, Dropout, Flatten, MaxPooling1D,
)
from xgboost import XGBRegressor

import cmapss_db as db

DATASET = "FD001"
RUL_CAP = 125
WINDOW_XGB = 10
WINDOW_SEQ = 30
DROP_COLS = ["P15", "Nc", "NRc", "setting_1", "setting_2"]
SENSOR_COLS = ["T24", "T30", "T50", "P30", "Nf", "Ps30", "phi", "NRf",
               "BPR", "htBleed", "W31", "W32"]
OUT_CSV = "seed_study.csv"


# --------------------------------------------------------------------------
# Metrics (identical to the notebook's section 5.1)
# --------------------------------------------------------------------------
def nasa_score(y_true, y_pred):
    """PHM08 asymmetric score: late predictions are punished harder. Lower is better."""
    d = y_pred - y_true
    return np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))


def evaluate(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "NASA Score": float(nasa_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------
# Data pipeline (notebook sections 3.1, 4, 5.1, 5.2)
# --------------------------------------------------------------------------
def add_rolling_features(df, sensors, window):
    df = df.copy()
    for s in sensors:
        g = df.groupby("unit_number")[s]
        df[f"{s}_mean_{window}"] = g.transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f"{s}_std_{window}"] = g.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        df[f"{s}_max_{window}"] = g.transform(lambda x: x.rolling(window, min_periods=1).max())
    return df


def create_sequences(df, sensors, window, target="RUL"):
    X, y = [], []
    for unit in df["unit_number"].unique():
        u = df[df["unit_number"] == unit].sort_values("time_in_cycles")
        sd, td = u[sensors].values, u[target].values
        for i in range(len(sd) - window + 1):
            X.append(sd[i:i + window])
            y.append(td[i + window - 1])
    return np.array(X), np.array(y)


def create_test_sequences(df, sensors, window):
    X = []
    for unit in df["unit_number"].unique():
        u = df[df["unit_number"] == unit].sort_values("time_in_cycles")
        sd = u[sensors].values
        if len(sd) >= window:
            X.append(sd[-window:])
        else:
            X.append(np.vstack([np.zeros((window - len(sd), len(sensors))), sd]))
    return np.array(X)


def build_data():
    df_train, df_test, _ = db.load_notebook_frames(DATASET)

    constant = [c for c in df_train.columns if df_train[c].std() < 1e-10]
    train = df_train.drop(columns=constant).drop(columns=DROP_COLS)
    test = df_test.drop(columns=constant).drop(columns=DROP_COLS)

    scaler = MinMaxScaler()
    train[SENSOR_COLS] = scaler.fit_transform(train[SENSOR_COLS])
    test[SENSOR_COLS] = scaler.transform(test[SENSOR_COLS])
    train["RUL"] = train["RUL"].clip(upper=RUL_CAP)

    # XGBoost: rolling features, last cycle of each test engine
    tr_fe = add_rolling_features(train, SENSOR_COLS, WINDOW_XGB)
    te_fe = add_rolling_features(test, SENSOR_COLS, WINDOW_XGB)
    exclude = ["unit_number", "time_in_cycles", "RUL_Max", "RUL"]
    feats = [c for c in tr_fe.columns if c not in exclude]
    te_last = te_fe.groupby("unit_number").last().reset_index()

    # Sequence models: 30-cycle windows
    Xs, ys = create_sequences(train, SENSOR_COLS, WINDOW_SEQ)
    Xt = create_test_sequences(test, SENSOR_COLS, WINDOW_SEQ)
    yt = test.groupby("unit_number")["RUL_Max"].last().values

    return {
        "xgb": (tr_fe[feats], tr_fe["RUL"], te_last[feats], te_last["RUL_Max"]),
        "seq": (Xs, ys, Xt, yt),
    }


# --------------------------------------------------------------------------
# Architectures (notebook sections 5.2-5.5)
# --------------------------------------------------------------------------
def build_lstm(shape):
    i = tf.keras.Input(shape=shape)
    x = LSTM(64, return_sequences=True)(i)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation="relu")(x)
    return tf.keras.Model(i, Dense(1)(x))


def build_gru(shape):
    i = tf.keras.Input(shape=shape)
    x = GRU(64, return_sequences=True)(i)
    x = Dropout(0.2)(x)
    x = GRU(32)(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation="relu")(x)
    return tf.keras.Model(i, Dense(1)(x))


def build_cnn(shape):
    i = tf.keras.Input(shape=shape)
    x = Conv1D(64, 3, activation="relu", padding="same")(i)
    x = Conv1D(32, 3, activation="relu", padding="same")(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    return tf.keras.Model(i, Dense(1)(x))


def build_cnn_lstm(shape):
    i = tf.keras.Input(shape=shape)
    x = Conv1D(64, 3, activation="relu", padding="same")(i)
    x = Conv1D(32, 3, activation="relu", padding="same")(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation="relu")(x)
    return tf.keras.Model(i, Dense(1)(x))


BUILDERS = {
    "LSTM": build_lstm,
    "GRU": build_gru,
    "1D CNN": build_cnn,
    "CNN-LSTM": build_cnn_lstm,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    print(f"TensorFlow {tf.__version__} | seeds {args.seeds} | max epochs {args.epochs}",
          flush=True)
    data = build_data()
    X_tr, y_tr, X_te, y_te = data["xgb"]
    Xs, ys, Xt, yt = data["seq"]
    print(f"sequences {Xs.shape} -> test {Xt.shape}\n", flush=True)

    rows = []
    for seed in args.seeds:
        # XGBoost is deterministic given random_state, but re-run per seed so the
        # table is uniform and the seed genuinely varies the model.
        t0 = time.time()
        xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                           random_state=seed, verbosity=0)
        xgb.fit(X_tr, y_tr)
        m = evaluate(y_te, xgb.predict(X_te))
        rows.append({"Model": "XGBoost", "seed": seed, **m})
        print(f"seed {seed}  XGBoost   RMSE {m['RMSE']:7.4f}  ({time.time()-t0:.0f}s)", flush=True)

        for name, builder in BUILDERS.items():
            t0 = time.time()
            tf.keras.utils.set_random_seed(seed)     # weights, dropout, shuffling
            model = builder(Xs.shape[1:])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
            hist = model.fit(
                Xs, ys, epochs=args.epochs, batch_size=64, validation_split=0.2,
                callbacks=[EarlyStopping(monitor="val_loss", patience=10,
                                         restore_best_weights=True)],
                verbose=0,
            )
            m = evaluate(yt, model.predict(Xt, verbose=0).flatten())
            rows.append({"Model": name, "seed": seed, "epochs_run": len(hist.history["loss"]), **m})
            print(f"seed {seed}  {name:9s} RMSE {m['RMSE']:7.4f}  "
                  f"({len(hist.history['loss'])} epochs, {time.time()-t0:.0f}s)", flush=True)
            tf.keras.backend.clear_session()

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    order = ["XGBoost", "LSTM", "GRU", "1D CNN", "CNN-LSTM"]
    summary = df.groupby("Model")[["RMSE", "MAE", "R2", "NASA Score"]].agg(["mean", "std"])
    print(f"\n=== mean +/- std over {len(args.seeds)} seeds ===")
    print(summary.loc[order].round(4).to_string())
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
