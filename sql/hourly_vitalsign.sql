-- Hourly vital signs for the cohort, with missing values preserved.
--
-- Records are assigned to calendar hours counted from the calendar hour
-- of ICU admission. A plain elapsed-time division would shift records by
-- the minutes-past-the-hour of admission and does not reproduce the
-- reference implementation.
--
-- NULLs are deliberately retained: the observation mask is derived from
-- them, and substituting zero would make an unmeasured hour
-- indistinguishable from a measured value of zero.

SELECT
    v.stay_id,
    v.charttime,
    TIMESTAMPDIFF(HOUR,
        DATE_FORMAT(ie.intime,    '%Y-%m-%d %H:00:00'),
        DATE_FORMAT(v.charttime,  '%Y-%m-%d %H:00:00')) AS hr,
    v.heart_rate, v.sbp, v.dbp, v.mbp,
    v.resp_rate, v.temperature, v.spo2, v.glucose
FROM vitalsign v
JOIN icustays ie ON v.stay_id = ie.stay_id
WHERE TIMESTAMPDIFF(HOUR,
        DATE_FORMAT(ie.intime,   '%Y-%m-%d %H:00:00'),
        DATE_FORMAT(v.charttime, '%Y-%m-%d %H:00:00')) BETWEEN 0 AND 23;
