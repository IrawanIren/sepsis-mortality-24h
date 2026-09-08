-- Recorded urine output.
--
-- Absence of a record for a given hour on a catheterised patient
-- indicates no output rather than no measurement, so unrecorded hours
-- are set to zero in preprocessing and this variable carries no
-- observation mask.

SELECT
    u.stay_id,
    u.charttime,
    u.urineoutput
FROM urine_output u
JOIN icustays ie ON u.stay_id = ie.stay_id
WHERE TIMESTAMPDIFF(HOUR,
        DATE_FORMAT(ie.intime,   '%Y-%m-%d %H:00:00'),
        DATE_FORMAT(u.charttime, '%Y-%m-%d %H:00:00')) BETWEEN 0 AND 23;
