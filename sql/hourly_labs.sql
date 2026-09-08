-- Laboratory results for the cohort, with missing values preserved.
--
-- The same calendar-hour convention as the vital-sign extract is used.
-- NULLs must not be coalesced to zero: zero is physiologically
-- impossible for every variable below, so a zero in the extract would be
-- indistinguishable from an unmeasured value and would corrupt both the
-- observation mask and the imputation.

SELECT
    l.stay_id,
    l.charttime,
    TIMESTAMPDIFF(HOUR,
        DATE_FORMAT(ie.intime,   '%Y-%m-%d %H:00:00'),
        DATE_FORMAT(l.charttime, '%Y-%m-%d %H:00:00')) AS hr,
    l.hematocrit, l.hemoglobin, l.platelets, l.wbc,
    l.albumin, l.aniongap, l.bicarbonate, l.bun,
    l.calcium, l.chloride, l.creatinine, l.glucose,
    l.sodium, l.potassium, l.fibrinogen, l.thrombin,
    l.inr, l.pt, l.ptt
FROM labevents_pivoted l
JOIN icustays ie ON l.stay_id = ie.stay_id
WHERE TIMESTAMPDIFF(HOUR,
        DATE_FORMAT(ie.intime,   '%Y-%m-%d %H:00:00'),
        DATE_FORMAT(l.charttime, '%Y-%m-%d %H:00:00')) BETWEEN 0 AND 23;
