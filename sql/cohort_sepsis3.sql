-- Sepsis-3 cohort, first ICU admission per patient.
--
-- Ported to MySQL from the official MIMIC-IV derived concepts at
-- https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts
-- The upstream queries target BigQuery and PostgreSQL; the changes here
-- are limited to dialect (DATE_SUB for DATETIME_SUB, no schema prefix).
--
-- Inclusion: Sepsis-3, age >= 18, first ICU stay, length of stay >= 24 h.

SELECT
    ie.subject_id,
    ie.hadm_id,
    ie.stay_id,
    ie.intime  AS icu_intime,
    ie.outtime AS icu_outtime,
    ie.los     AS los_icu,
    p.gender,
    p.dod,
    FLOOR(DATEDIFF(ie.intime, DATE_SUB(p.anchor_year_start, INTERVAL p.anchor_age YEAR)) / 365.25) AS admission_age,
    a.race,
    s3.sofa_time,
    s3.sofa_score AS sofa_onset,
    CASE WHEN p.dod IS NOT NULL
              AND p.dod <= DATE_ADD(ie.intime, INTERVAL 30 DAY)
         THEN 1 ELSE 0 END AS label
FROM icustays ie
JOIN patients  p  ON ie.subject_id = p.subject_id
JOIN admissions a ON ie.hadm_id    = a.hadm_id
JOIN sepsis3   s3 ON ie.stay_id    = s3.stay_id AND s3.sepsis3 = 1
WHERE ie.los >= 1.0
  AND ie.stay_id IN (
        SELECT stay_id FROM icustay_detail WHERE first_icu_stay = 1
  );
