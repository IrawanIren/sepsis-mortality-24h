-- Hourly SOFA with its six organ components, plus the inputs needed to
-- derive minimum GCS and vasopressor exposure at any horizon.
--
-- This single extract supports all three prediction horizons. Restricting
-- it to hours 0..H-1 in the analysis code is what makes the shorter
-- horizons temporally valid: severity and treatment features must not
-- draw on measurements recorded after the prediction time.
--
-- Note that gcs_min encodes "not measured" as zero, which is restored to
-- NULL in preprocessing since the scale has a floor of three.

SELECT
    s.stay_id,
    s.hr,
    s.starttime,
    s.endtime,
    s.gcs_min,
    s.rate_epinephrine, s.rate_norepinephrine,
    s.rate_dopamine,    s.rate_dobutamine,
    s.respiration_24hours, s.coagulation_24hours,
    s.liver_24hours,       s.cardiovascular_24hours,
    s.cns_24hours,         s.renal_24hours,
    s.sofa_24hours
FROM sofa s
JOIN icustays ie ON s.stay_id = ie.stay_id
WHERE s.hr BETWEEN 0 AND 23;
