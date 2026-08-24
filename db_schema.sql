-- NASA C-MAPSS Turbofan Engine Degradation dataset schema (Neon / PostgreSQL)
--
-- Source files: CMaps/{train,test}_FD00{1..4}.txt, CMaps/RUL_FD00{1..4}.txt
-- Each reading row has 26 space-separated columns:
--   1) unit number, 2) time in cycles, 3-5) operational settings, 6-26) sensors 1..21

CREATE SCHEMA IF NOT EXISTS cmapss;

-- Metadata about the four sub-datasets (conditions / fault modes differ)
CREATE TABLE IF NOT EXISTS cmapss.datasets (
    dataset            TEXT PRIMARY KEY,
    n_conditions       SMALLINT NOT NULL,
    n_fault_modes      SMALLINT NOT NULL,
    train_trajectories INT      NOT NULL,
    test_trajectories  INT      NOT NULL,
    description        TEXT
);

-- Training run-to-failure trajectories
CREATE TABLE IF NOT EXISTS cmapss.train (
    dataset       TEXT     NOT NULL REFERENCES cmapss.datasets(dataset),
    unit_number   INT      NOT NULL,
    time_cycles   INT      NOT NULL,
    op_setting_1  REAL,
    op_setting_2  REAL,
    op_setting_3  REAL,
    sensor_1      REAL, sensor_2  REAL, sensor_3  REAL, sensor_4  REAL,
    sensor_5      REAL, sensor_6  REAL, sensor_7  REAL, sensor_8  REAL,
    sensor_9      REAL, sensor_10 REAL, sensor_11 REAL, sensor_12 REAL,
    sensor_13     REAL, sensor_14 REAL, sensor_15 REAL, sensor_16 REAL,
    sensor_17     REAL, sensor_18 REAL, sensor_19 REAL, sensor_20 REAL,
    sensor_21     REAL,
    PRIMARY KEY (dataset, unit_number, time_cycles)
);

-- Test trajectories, truncated some time before failure
CREATE TABLE IF NOT EXISTS cmapss.test (
    dataset       TEXT     NOT NULL REFERENCES cmapss.datasets(dataset),
    unit_number   INT      NOT NULL,
    time_cycles   INT      NOT NULL,
    op_setting_1  REAL,
    op_setting_2  REAL,
    op_setting_3  REAL,
    sensor_1      REAL, sensor_2  REAL, sensor_3  REAL, sensor_4  REAL,
    sensor_5      REAL, sensor_6  REAL, sensor_7  REAL, sensor_8  REAL,
    sensor_9      REAL, sensor_10 REAL, sensor_11 REAL, sensor_12 REAL,
    sensor_13     REAL, sensor_14 REAL, sensor_15 REAL, sensor_16 REAL,
    sensor_17     REAL, sensor_18 REAL, sensor_19 REAL, sensor_20 REAL,
    sensor_21     REAL,
    PRIMARY KEY (dataset, unit_number, time_cycles)
);

-- Ground-truth Remaining Useful Life for each test unit.
-- RUL_FD00X.txt is ordered by unit number, so line N -> unit_number N.
CREATE TABLE IF NOT EXISTS cmapss.rul (
    dataset     TEXT NOT NULL REFERENCES cmapss.datasets(dataset),
    unit_number INT  NOT NULL,
    rul         INT  NOT NULL,
    PRIMARY KEY (dataset, unit_number)
);

CREATE INDEX IF NOT EXISTS train_unit_idx ON cmapss.train (dataset, unit_number);
CREATE INDEX IF NOT EXISTS test_unit_idx  ON cmapss.test  (dataset, unit_number);
