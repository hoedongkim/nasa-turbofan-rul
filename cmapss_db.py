"""Load the NASA C-MAPSS dataset, from Neon Postgres or from the raw text files.

    import cmapss_db as db

    train = db.load_train("FD001")               # all training readings
    test  = db.load_test("FD001")
    rul   = db.load_rul("FD001")                 # ground-truth RUL per test unit
    train = db.load_train_with_rul("FD001")      # adds a per-row RUL target
    df    = db.query("SELECT * FROM cmapss.train LIMIT 5")   # database mode only

There are two data sources and the choice is automatic:

  * **Neon Postgres** when `DATABASE_URL` is set (in the environment or a local
    `.env`). This is the authoring setup — see `upload_to_neon.py`.
  * **The raw `CMaps/*.txt` files** otherwise, so a fresh clone runs with no
    database, no credentials and no setup.

Both paths return identical frames — verified value-for-value across all four
sub-datasets — so results do not depend on which one is in use. The database
packages are imported lazily, meaning the file path needs neither `sqlalchemy`
nor `psycopg` installed.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "CMaps"

SENSORS = [f"sensor_{i}" for i in range(1, 22)]
OP_SETTINGS = [f"op_setting_{i}" for i in range(1, 4)]
READING_COLUMNS = ["unit_number", "time_cycles"] + OP_SETTINGS + SENSORS

# Physical column names used by the NASA_Turbofan notebook and the PHM08 papers.
# Pass names="notebook" to any loader to get these instead of sensor_1..21.
NOTEBOOK_NAMES = {
    "time_cycles": "time_in_cycles",
    "op_setting_1": "setting_1", "op_setting_2": "setting_2", "op_setting_3": "TRA",
    "sensor_1": "T2",    "sensor_2": "T24",   "sensor_3": "T30",  "sensor_4": "T50",
    "sensor_5": "P2",    "sensor_6": "P15",   "sensor_7": "P30",  "sensor_8": "Nf",
    "sensor_9": "Nc",    "sensor_10": "epr",  "sensor_11": "Ps30", "sensor_12": "phi",
    "sensor_13": "NRf",  "sensor_14": "NRc",  "sensor_15": "BPR", "sensor_16": "farB",
    "sensor_17": "htBleed", "sensor_18": "Nf_dmd", "sensor_19": "PCNfR_dmd",
    "sensor_20": "W31",  "sensor_21": "W32",  "rul": "RUL",
}

DATASETS = ["FD001", "FD002", "FD003", "FD004"]


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------
def _dsn() -> str | None:
    """The Neon connection string, or None when no database is configured."""
    if not os.getenv("DATABASE_URL"):
        try:                                    # optional — absent on a bare clone
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
    return os.getenv("DATABASE_URL")


def using_database() -> bool:
    """True when reads go to Neon, False when they come from CMaps/*.txt."""
    return _dsn() is not None


def source_description() -> str:
    """One line naming the active data source — handy at the top of a notebook."""
    if using_database():
        return "Neon Postgres (schema `cmapss`)"
    return f"local files in {DATA_DIR.name}/"


@lru_cache(maxsize=1)
def get_engine():
    """SQLAlchemy engine for the Neon database. Requires DATABASE_URL."""
    dsn = _dsn()
    if not dsn:
        raise RuntimeError(
            "DATABASE_URL is not set, so there is no database to connect to. "
            "Either add it to .env, or just use the loaders — they fall back to "
            f"{DATA_DIR.name}/ automatically."
        )
    from sqlalchemy import create_engine
    # SQLAlchemy needs the psycopg3 driver named explicitly.
    return create_engine(dsn.replace("postgresql://", "postgresql+psycopg://", 1),
                         pool_pre_ping=True)


def query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Run arbitrary SQL and return a DataFrame. Database mode only."""
    engine = get_engine()          # raises a clear error when there is no database
    from sqlalchemy import text
    with engine.connect() as conn:
        return pd.read_sql_query(text(sql), conn, params=params or {})


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
def _read_file(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Either restore the CMaps/ directory or set "
            "DATABASE_URL to read from Neon instead."
        )
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != len(columns):
        raise ValueError(f"{path.name}: expected {len(columns)} columns, got {df.shape[1]}")
    df.columns = columns
    return df


# Match the database schema exactly. Two sensors hold whole numbers in the
# source files, so pandas would infer int64 for them while the REAL columns in
# Postgres come back as float64 — pin the dtypes so both paths agree.
READING_DTYPES = ({c: "int64" for c in ["unit_number", "time_cycles"]}
                  | {c: "float64" for c in OP_SETTINGS + SENSORS})


def _readings_from_files(table: str, dataset: str | None) -> pd.DataFrame:
    frames = []
    for name in ([dataset] if dataset else DATASETS):
        df = _read_file(DATA_DIR / f"{table}_{name}.txt", READING_COLUMNS)
        df = df.astype(READING_DTYPES)
        df.insert(0, "dataset", name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _rul_from_files(dataset: str | None) -> pd.DataFrame:
    frames = []
    for name in ([dataset] if dataset else DATASETS):
        # RUL_FD00X.txt is ordered by unit number, so line N -> unit_number N.
        df = _read_file(DATA_DIR / f"RUL_{name}.txt", ["rul"])
        df.insert(0, "unit_number", range(1, len(df) + 1))
        df.insert(0, "dataset", name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _apply_names(df: pd.DataFrame, names: str) -> pd.DataFrame:
    if names == "db":
        return df
    if names != "notebook":
        raise ValueError(f'names must be "db" or "notebook", got {names!r}')
    return df.rename(columns=NOTEBOOK_NAMES)


def _load_readings(table: str, dataset: str | None, units, names: str) -> pd.DataFrame:
    if using_database():
        sql = f"SELECT * FROM cmapss.{table}"
        where, params = [], {}
        if dataset:
            where.append("dataset = :dataset")
            params["dataset"] = dataset
        if units is not None:
            where.append("unit_number = ANY(:units)")
            params["units"] = list(units)
        if where:
            sql += " WHERE " + " AND ".join(where)
        df = query(sql, params)
    else:
        df = _readings_from_files(table, dataset)
        if units is not None:
            df = df[df.unit_number.isin(list(units))]

    return _apply_names(
        df.sort_values(["dataset", "unit_number", "time_cycles"]).reset_index(drop=True),
        names)


def load_train(dataset: str | None = "FD001", units=None,
               names: str = "db") -> pd.DataFrame:
    """Training readings. dataset=None loads all four sub-datasets."""
    return _load_readings("train", dataset, units, names)


def load_test(dataset: str | None = "FD001", units=None,
              names: str = "db") -> pd.DataFrame:
    """Test readings (truncated before failure)."""
    return _load_readings("test", dataset, units, names)


def load_rul(dataset: str | None = "FD001") -> pd.DataFrame:
    """Ground-truth RUL for each test unit."""
    if using_database():
        sql = "SELECT * FROM cmapss.rul"
        params = {}
        if dataset:
            sql += " WHERE dataset = :dataset"
            params["dataset"] = dataset
        df = query(sql, params)
    else:
        df = _rul_from_files(dataset)
    return df.sort_values(["dataset", "unit_number"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# RUL targets
#
# Computed in pandas rather than SQL so that the database and file paths run the
# identical calculation and cannot drift apart.
# ---------------------------------------------------------------------------
def _max_cycle(df: pd.DataFrame) -> pd.Series:
    return df.groupby(["dataset", "unit_number"])["time_cycles"].transform("max")


def load_train_with_rul(dataset: str | None = "FD001",
                        names: str = "db") -> pd.DataFrame:
    """Training readings plus a per-row `rul` target.

    Training trajectories run to failure, so a row's RUL is simply the number of
    cycles left until that unit's last recorded cycle.
    """
    df = load_train(dataset)
    df["rul"] = _max_cycle(df) - df["time_cycles"]
    return _apply_names(df, names)


def load_test_with_rul(dataset: str | None = "FD001",
                       names: str = "db") -> pd.DataFrame:
    """Test readings plus a per-row `rul` built from the ground-truth file.

    A test trajectory stops early, so RUL = (cycles remaining in the record)
    + (the true RUL after the last recorded cycle).
    """
    df = load_test(dataset)
    truth = load_rul(dataset).rename(columns={"rul": "_final_rul"})
    df = df.merge(truth, on=["dataset", "unit_number"], how="left")
    df["rul"] = (_max_cycle(df) - df["time_cycles"]) + df["_final_rul"]
    return _apply_names(df.drop(columns="_final_rul"), names)


def load_notebook_frames(dataset: str = "FD001"):
    """Drop-in replacement for the notebook's file-loading cells.

    Returns (df_train, df_test, df_RUL) shaped exactly as the notebook has them
    after its cells 4, 9 and 10 — notebook column names, no `dataset` column,
    and the same RUL_Max / RUL semantics:

        df_train.RUL_Max = that engine's last cycle (it runs to failure)
        df_train.RUL     = RUL_Max - time_in_cycles
        df_test.RUL_Max  = ground-truth RUL at the last observed cycle (the
                           `Y` value); this is what the notebook uses as y_test
        df_test.RUL      = RUL_Max - time_in_cycles
    """
    train = load_train(dataset, names="notebook").drop(columns=["dataset"])
    train["RUL_Max"] = train.groupby("unit_number")["time_in_cycles"].transform("max")
    train["RUL"] = train["RUL_Max"] - train["time_in_cycles"]

    df_rul = load_rul(dataset)[["unit_number", "rul"]].rename(columns={"rul": "Y"})

    test = load_test(dataset, names="notebook").drop(columns=["dataset"])
    test = test.merge(df_rul, on="unit_number", how="left")
    test = test.rename(columns={"Y": "RUL_Max"})
    test["RUL"] = test["RUL_Max"] - test["time_in_cycles"]

    return train, test, df_rul[["Y", "unit_number"]]


if __name__ == "__main__":
    print(f"Reading from {source_description()}\n")
    rows = []
    for name in DATASETS:
        tr, te = load_train(name), load_test(name)
        rows.append({"dataset": name,
                     "train_rows": len(tr), "train_units": tr.unit_number.nunique(),
                     "test_rows": len(te), "test_units": te.unit_number.nunique()})
    print(pd.DataFrame(rows).to_string(index=False))
