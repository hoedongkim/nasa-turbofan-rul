"""Upload the NASA C-MAPSS dataset in ./CMaps to a Neon Postgres database.

Usage:
    python upload_to_neon.py            # create schema + load everything
    python upload_to_neon.py --recreate # drop the cmapss schema first, then load

Requires DATABASE_URL in .env (or the environment) — the Neon connection string.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

DATA_DIR = Path(__file__).parent / "CMaps"
SCHEMA_FILE = Path(__file__).parent / "db_schema.sql"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]

READING_COLUMNS = (
    ["unit_number", "time_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# From CMaps/readme.txt. Trajectory counts are the ones actually present in the
# files — readme.txt has FD004's train/test counts the wrong way round (it says
# 248 train / 249 test; the files hold 249 train and 248 test units).
DATASET_META = [
    ("FD001", 1, 1, 100, 100, "Sea level; HPC degradation"),
    ("FD002", 6, 1, 260, 259, "Six conditions; HPC degradation"),
    ("FD003", 1, 2, 100, 100, "Sea level; HPC + fan degradation"),
    ("FD004", 6, 2, 249, 248, "Six conditions; HPC + fan degradation"),
]


def read_readings(path: Path) -> pd.DataFrame:
    """Read a train_/test_ file: 26 space-separated columns, trailing whitespace."""
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != len(READING_COLUMNS):
        raise ValueError(f"{path.name}: expected 26 columns, got {df.shape[1]}")
    df.columns = READING_COLUMNS
    return df


def read_rul(path: Path) -> pd.DataFrame:
    """Read a RUL_ file: one value per line, line N is test unit N."""
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python", names=["rul"])
    df.insert(0, "unit_number", range(1, len(df) + 1))
    return df


def copy_frame(conn: psycopg.Connection, table: str, df: pd.DataFrame) -> None:
    cols = ", ".join(df.columns)
    with conn.cursor() as cur:
        with cur.copy(f"COPY {table} ({cols}) FROM STDIN") as copy:
            for row in df.itertuples(index=False, name=None):
                copy.write_row(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true",
                        help="DROP SCHEMA cmapss CASCADE before loading")
    args = parser.parse_args()

    load_dotenv()
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL is not set. Put your Neon connection string in .env",
              file=sys.stderr)
        return 1

    if not DATA_DIR.is_dir():
        print(f"Data directory not found: {DATA_DIR}", file=sys.stderr)
        return 1

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            if args.recreate:
                print("Dropping schema cmapss ...")
                cur.execute("DROP SCHEMA IF EXISTS cmapss CASCADE")
            cur.execute(SCHEMA_FILE.read_text())

            cur.executemany(
                """INSERT INTO cmapss.datasets
                       (dataset, n_conditions, n_fault_modes,
                        train_trajectories, test_trajectories, description)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (dataset) DO UPDATE SET description = EXCLUDED.description""",
                DATASET_META,
            )
        conn.commit()

        for name in DATASETS:
            for split in ("train", "test"):
                df = read_readings(DATA_DIR / f"{split}_{name}.txt")
                df.insert(0, "dataset", name)
                with conn.cursor() as cur:
                    cur.execute(f"DELETE FROM cmapss.{split} WHERE dataset = %s", (name,))
                copy_frame(conn, f"cmapss.{split}", df)
                conn.commit()
                print(f"  {split}_{name}: {len(df):>6,} rows, "
                      f"{df.unit_number.nunique()} units")

            rul = read_rul(DATA_DIR / f"RUL_{name}.txt")
            rul.insert(0, "dataset", name)
            with conn.cursor() as cur:
                cur.execute("DELETE FROM cmapss.rul WHERE dataset = %s", (name,))
            copy_frame(conn, "cmapss.rul", rul)
            conn.commit()
            print(f"  RUL_{name}:   {len(rul):>6,} rows")

        with conn.cursor() as cur:
            cur.execute("ANALYZE cmapss.train")
            cur.execute("ANALYZE cmapss.test")
            for table in ("train", "test", "rul"):
                cur.execute(f"SELECT count(*) FROM cmapss.{table}")
                print(f"cmapss.{table}: {cur.fetchone()[0]:,} rows total")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
