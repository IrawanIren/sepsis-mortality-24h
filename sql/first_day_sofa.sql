-- ------------------------------------------------------------------
-- Title: Sequential Organ Failure Assessment (SOFA), first ICU day
--
-- MySQL 8.0 port of the MIMIC-IV derived `first_day_sofa` query.
-- Returns one row per ICU stay, scored on hours 0-23 of the stay.
--
-- Changes from the BigQuery original:
--   * project-qualified table names reduced to schema.table
--   * DATETIME_SUB / DATETIME_ADD  ->  DATE_SUB / DATE_ADD
--   * INTERVAL '6' HOUR            ->  INTERVAL 6 HOUR
--   * final SELECT restricted to the Sepsis-3 cohort to keep the
--     export small; remove the last JOIN to score every ICU stay
--
-- Reference:
--   Vincent JL, Moreno R, Takala J, et al. The SOFA (Sepsis-related
--   Organ Failure Assessment) score to describe organ dysfunction /
--   failure. Intensive Care Med. 1996;22(7):707-710.
--
-- Required derived tables:
--   norepinephrine, epinephrine, dopamine, dobutamine, bg, ventilation,
--   first_day_vitalsign, first_day_lab, first_day_urine_output,
--   first_day_gcs, sepsis3
-- ------------------------------------------------------------------

WITH vaso_stg AS (
    SELECT ie.stay_id, 'norepinephrine' AS treatment, mv.vaso_rate AS rate
    FROM mimiciv_icu.icustays ie
    INNER JOIN mimiciv_derived.norepinephrine mv
        ON ie.stay_id = mv.stay_id
        AND mv.starttime >= DATE_SUB(ie.intime, INTERVAL 6 HOUR)
        AND mv.starttime <= DATE_ADD(ie.intime, INTERVAL 1 DAY)
    UNION ALL
    SELECT ie.stay_id, 'epinephrine', mv.vaso_rate
    FROM mimiciv_icu.icustays ie
    INNER JOIN mimiciv_derived.epinephrine mv
        ON ie.stay_id = mv.stay_id
        AND mv.starttime >= DATE_SUB(ie.intime, INTERVAL 6 HOUR)
        AND mv.starttime <= DATE_ADD(ie.intime, INTERVAL 1 DAY)
    UNION ALL
    SELECT ie.stay_id, 'dobutamine', mv.vaso_rate
    FROM mimiciv_icu.icustays ie
    INNER JOIN mimiciv_derived.dobutamine mv
        ON ie.stay_id = mv.stay_id
        AND mv.starttime >= DATE_SUB(ie.intime, INTERVAL 6 HOUR)
        AND mv.starttime <= DATE_ADD(ie.intime, INTERVAL 1 DAY)
    UNION ALL
    SELECT ie.stay_id, 'dopamine', mv.vaso_rate
    FROM mimiciv_icu.icustays ie
    INNER JOIN mimiciv_derived.dopamine mv
        ON ie.stay_id = mv.stay_id
        AND mv.starttime >= DATE_SUB(ie.intime, INTERVAL 6 HOUR)
        AND mv.starttime <= DATE_ADD(ie.intime, INTERVAL 1 DAY)
),

vaso_mv AS (
    SELECT
        ie.stay_id,
        MAX(CASE WHEN v.treatment = 'norepinephrine' THEN v.rate END) AS rate_norepinephrine,
        MAX(CASE WHEN v.treatment = 'epinephrine'    THEN v.rate END) AS rate_epinephrine,
        MAX(CASE WHEN v.treatment = 'dopamine'       THEN v.rate END) AS rate_dopamine,
        MAX(CASE WHEN v.treatment = 'dobutamine'     THEN v.rate END) AS rate_dobutamine
    FROM mimiciv_icu.icustays ie
    LEFT JOIN vaso_stg v ON ie.stay_id = v.stay_id
    GROUP BY ie.stay_id
),

pafi1 AS (
    -- Join blood gas to ventilation durations to flag ventilated samples.
    SELECT
        ie.stay_id,
        bg.charttime,
        bg.pao2fio2ratio,
        CASE WHEN vd.stay_id IS NOT NULL THEN 1 ELSE 0 END AS isvent
    FROM mimiciv_icu.icustays ie
    LEFT JOIN mimiciv_derived.bg bg
        ON ie.subject_id = bg.subject_id
        AND bg.charttime >= DATE_SUB(ie.intime, INTERVAL 6 HOUR)
        AND bg.charttime <= DATE_ADD(ie.intime, INTERVAL 1 DAY)
        AND bg.specimen = 'ART.'
    LEFT JOIN mimiciv_derived.ventilation vd
        ON ie.stay_id = vd.stay_id
        AND bg.charttime >= vd.starttime
        AND bg.charttime <= vd.endtime
        AND vd.ventilation_status = 'InvasiveVent'
),

pafi2 AS (
    -- Ventilated and unventilated minima are kept apart: the lowest
    -- unventilated ratio may be 68 while the lowest ventilated ratio is
    -- 120, in which case the respiration score is 3 rather than 4.
    SELECT
        stay_id,
        MIN(CASE WHEN isvent = 0 THEN pao2fio2ratio END) AS pao2fio2_novent_min,
        MIN(CASE WHEN isvent = 1 THEN pao2fio2ratio END) AS pao2fio2_vent_min
    FROM pafi1
    GROUP BY stay_id
),

scorecomp AS (
    SELECT
        ie.stay_id,
        v.mbp_min,
        mv.rate_norepinephrine,
        mv.rate_epinephrine,
        mv.rate_dopamine,
        mv.rate_dobutamine,
        l.creatinine_max,
        l.bilirubin_total_max AS bilirubin_max,
        l.platelets_min       AS platelet_min,
        pf.pao2fio2_novent_min,
        pf.pao2fio2_vent_min,
        uo.urineoutput,
        gcs.gcs_min
    FROM mimiciv_icu.icustays ie
    LEFT JOIN vaso_mv mv                              ON ie.stay_id = mv.stay_id
    LEFT JOIN pafi2 pf                                ON ie.stay_id = pf.stay_id
    LEFT JOIN mimiciv_derived.first_day_vitalsign v   ON ie.stay_id = v.stay_id
    LEFT JOIN mimiciv_derived.first_day_lab l         ON ie.stay_id = l.stay_id
    LEFT JOIN mimiciv_derived.first_day_urine_output uo ON ie.stay_id = uo.stay_id
    LEFT JOIN mimiciv_derived.first_day_gcs gcs       ON ie.stay_id = gcs.stay_id
),

scorecalc AS (
    -- A NULL component means the underlying measurement is missing.
    -- Components are coalesced to 0 in the final SELECT, but are kept
    -- NULL here so that missingness stays visible.
    SELECT
        stay_id,

        CASE
            WHEN pao2fio2_vent_min   < 100 THEN 4
            WHEN pao2fio2_vent_min   < 200 THEN 3
            WHEN pao2fio2_novent_min < 300 THEN 2
            WHEN pao2fio2_novent_min < 400 THEN 1
            WHEN COALESCE(pao2fio2_vent_min, pao2fio2_novent_min) IS NULL THEN NULL
            ELSE 0
        END AS respiration,

        CASE
            WHEN platelet_min < 20  THEN 4
            WHEN platelet_min < 50  THEN 3
            WHEN platelet_min < 100 THEN 2
            WHEN platelet_min < 150 THEN 1
            WHEN platelet_min IS NULL THEN NULL
            ELSE 0
        END AS coagulation,

        CASE
            WHEN bilirubin_max >= 12.0 THEN 4
            WHEN bilirubin_max >=  6.0 THEN 3
            WHEN bilirubin_max >=  2.0 THEN 2
            WHEN bilirubin_max >=  1.2 THEN 1
            WHEN bilirubin_max IS NULL THEN NULL
            ELSE 0
        END AS liver,

        CASE
            WHEN rate_dopamine > 15
                OR rate_epinephrine > 0.1
                OR rate_norepinephrine > 0.1 THEN 4
            WHEN rate_dopamine > 5
                OR rate_epinephrine <= 0.1
                OR rate_norepinephrine <= 0.1 THEN 3
            WHEN rate_dopamine > 0 OR rate_dobutamine > 0 THEN 2
            WHEN mbp_min < 70 THEN 1
            WHEN COALESCE(mbp_min, rate_dopamine, rate_dobutamine,
                          rate_epinephrine, rate_norepinephrine) IS NULL THEN NULL
            ELSE 0
        END AS cardiovascular,

        CASE
            WHEN gcs_min BETWEEN 13 AND 14 THEN 1
            WHEN gcs_min BETWEEN 10 AND 12 THEN 2
            WHEN gcs_min BETWEEN  6 AND  9 THEN 3
            WHEN gcs_min < 6 THEN 4
            WHEN gcs_min IS NULL THEN NULL
            ELSE 0
        END AS cns,

        CASE
            WHEN creatinine_max >= 5.0 THEN 4
            WHEN urineoutput < 200 THEN 4
            WHEN creatinine_max >= 3.5 AND creatinine_max < 5.0 THEN 3
            WHEN urineoutput < 500 THEN 3
            WHEN creatinine_max >= 2.0 AND creatinine_max < 3.5 THEN 2
            WHEN creatinine_max >= 1.2 AND creatinine_max < 2.0 THEN 1
            WHEN COALESCE(urineoutput, creatinine_max) IS NULL THEN NULL
            ELSE 0
        END AS renal

    FROM scorecomp
)

SELECT
    ie.subject_id,
    ie.hadm_id,
    ie.stay_id,
      COALESCE(s.respiration, 0)
    + COALESCE(s.coagulation, 0)
    + COALESCE(s.liver, 0)
    + COALESCE(s.cardiovascular, 0)
    + COALESCE(s.cns, 0)
    + COALESCE(s.renal, 0)      AS sofa_first_day,
    s.respiration    AS sofa_respiration,
    s.coagulation    AS sofa_coagulation,
    s.liver          AS sofa_liver,
    s.cardiovascular AS sofa_cardiovascular,
    s.cns            AS sofa_cns,
    s.renal          AS sofa_renal
FROM mimiciv_icu.icustays ie
LEFT JOIN scorecalc s ON ie.stay_id = s.stay_id
-- Restrict to the Sepsis-3 cohort. Drop this JOIN to score every stay.
INNER JOIN mimiciv_derived.sepsis3 s3
    ON ie.stay_id = s3.stay_id AND s3.sepsis3 = 1
ORDER BY ie.stay_id;
