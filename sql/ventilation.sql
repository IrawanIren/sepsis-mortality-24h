-- Ventilation episodes, used to derive an invasive-ventilation flag at
-- each horizon from the episode start time.

SELECT
    v.stay_id,
    v.starttime,
    v.endtime,
    v.ventilation_status
FROM ventilation v
JOIN icustays ie ON v.stay_id = ie.stay_id
WHERE v.starttime < DATE_ADD(ie.intime, INTERVAL 24 HOUR);
