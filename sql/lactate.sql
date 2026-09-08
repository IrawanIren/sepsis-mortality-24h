-- Lactate measurements. Arterial, venous, and mixed specimens are
-- combined; arterial accounts for roughly three quarters of records.
-- Zero is restored to NULL in preprocessing, as lactate cannot be zero
-- in a living patient.

SELECT
    l.subject_id,
    l.hadm_id,
    l.charttime,
    l.specimen,
    l.lactate
FROM lactate l;
