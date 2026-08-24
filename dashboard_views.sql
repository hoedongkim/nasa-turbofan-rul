-- Views the Tableau dashboard connects to.
--
-- Both are flat and pre-joined so the BI tool needs no relationship modelling:
-- point Tableau at one view and every field it needs is already there.
--
-- Populate the underlying tables first:  python export_predictions.py

-- ---------------------------------------------------------------------------
-- Fleet snapshot — one row per engine per model.
-- "Now" is each engine's last recorded cycle (see export_predictions.py).
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW cmapss.v_fleet_dashboard AS
SELECT
    f.dataset,
    f.model,
    f.unit_number,
    f.current_cycle,
    f.predicted_rul,
    f.true_rul,
    f.abs_error,
    f.risk_band,
    -- Plain-language instruction for the alert list
    CASE f.risk_band
        WHEN '1 Critical' THEN 'Ground now'
        WHEN '2 Warning'  THEN 'Schedule maintenance'
        ELSE 'In service'
    END AS action,
    -- Where the engine is expected to end up, for a "life used" gauge
    f.current_cycle + f.predicted_rul AS projected_total_life,
    -- Cast the whole ratio to numeric: round(double precision, int) does not exist
    ROUND((f.current_cycle::numeric
           / NULLIF((f.current_cycle + f.predicted_rul)::numeric, 0)) * 100, 1) AS pct_life_used,
    -- 1 = most urgent, so the alert table sorts without extra Tableau logic
    RANK() OVER (PARTITION BY f.dataset, f.model ORDER BY f.predicted_rul) AS urgency_rank
FROM cmapss.fleet_status f;

-- ---------------------------------------------------------------------------
-- Per-engine trajectory — the drill-down curve behind a selected engine.
-- `is_latest` marks the point that appears in the snapshot above.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW cmapss.v_rul_trajectory AS
SELECT
    h.dataset,
    h.model,
    h.unit_number,
    h.time_cycles,
    h.predicted_rul,
    h.true_rul,
    h.risk_band,
    f.current_cycle,
    (h.time_cycles = f.current_cycle) AS is_latest
FROM cmapss.rul_history h
JOIN cmapss.fleet_status f USING (dataset, model, unit_number);
