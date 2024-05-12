Drop TABLE IF EXISTS Blood_pressure;
CREATE TABLE Blood_pressure AS
SELECT
	subject_id,
	hadm_id,
	stay_id,
	charttime,
	MAX(COALESCE(A_SBP, NI_SBP)) AS sbp,
	MAX(COALESCE(A_DBP, NI_DBP)) AS dbp,
	MAX(COALESCE(A_MAP, NI_MAP)) AS "map"
FROM
	(
		SELECT
			subject_id,
			hadm_id,
			stay_id,
			charttime,
			MAX(CASE WHEN itemid = 220050 THEN value END) AS a_sbp,
			MAX(CASE WHEN itemid = 220051 THEN value END) AS a_dbp,
			MAX(CASE WHEN itemid = 220052 THEN value END) AS a_map,
			MAX(CASE WHEN itemid = 220179 THEN value END) AS ni_sbp,
			MAX(CASE WHEN itemid = 220180 THEN value END) AS ni_dbp,
			MAX(CASE WHEN itemid = 220181 THEN value END) AS ni_map
		FROM
			mimiciv_icu.chartevents
		WHERE
			itemid IN (220050,220051,220052,220179,220180,220181)
		GROUP BY
			subject_id,
			hadm_id,
			stay_id,
			charttime
	) AS subquery
GROUP BY
	subject_id,
	hadm_id,
	stay_id,
	charttime
ORDER BY
	subject_id ASC,
	hadm_id ASC,
	stay_id ASC;
Drop TABLE IF EXISTS blood_pressure_with_label;
CREATE TABLE blood_pressure_with_label AS
SELECT
	"public".blood_pressure.subject_id,
	"public".blood_pressure.hadm_id,
	"public".blood_pressure.stay_id,
	mimiciv_hosp.admissions.hospital_expire_flag,
	mimiciv_hosp.admissions.deathtime,
	"public".blood_pressure.charttime,
	"public".blood_pressure.sbp,
	"public".blood_pressure.dbp,
	"public".blood_pressure."map"
FROM
	"public".blood_pressure
	INNER JOIN
	mimiciv_hosp.admissions
	ON
		"public".blood_pressure.subject_id = mimiciv_hosp.admissions.subject_id AND
		"public".blood_pressure.hadm_id = mimiciv_hosp.admissions.hadm_id
ORDER BY
	"public".blood_pressure.subject_id ASC,
	"public".blood_pressure.hadm_id ASC,
	"public".blood_pressure.stay_id ASC,
	"public".blood_pressure.charttime ASC;
Drop TABLE IF EXISTS Blood_pressure;

Drop TABLE IF EXISTS Heart_rate;
CREATE TABLE Heart_rate AS
SELECT
	subject_id,
	hadm_id,
	stay_id,
	charttime,
	CASE WHEN itemid = 220045 THEN value ELSE NULL END AS hr
FROM
	mimiciv_icu.chartevents
WHERE
	itemid IN (220045)
ORDER BY
	subject_id ASC,
	hadm_id ASC,
	stay_id ASC;
Drop TABLE IF EXISTS heart_rate_with_label;
CREATE TABLE heart_rate_with_label AS
SELECT
	"public".heart_rate.subject_id,
	"public".heart_rate.hadm_id,
	"public".heart_rate.stay_id,
	mimiciv_hosp.admissions.deathtime,
	mimiciv_hosp.admissions.hospital_expire_flag,
	"public".heart_rate.charttime,
	"public".heart_rate.hr
FROM
	"public".heart_rate
	INNER JOIN
	mimiciv_hosp.admissions
	ON
		"public".heart_rate.subject_id = mimiciv_hosp.admissions.subject_id AND
		"public".heart_rate.hadm_id = mimiciv_hosp.admissions.hadm_id
ORDER BY
	"public".heart_rate.subject_id ASC,
	"public".heart_rate.hadm_id ASC,
	"public".heart_rate.stay_id ASC,
	"public".heart_rate.charttime ASC;
Drop TABLE IF EXISTS Heart_rate;

Drop TABLE IF EXISTS Respiratory_Rate;
CREATE TABLE Respiratory_Rate AS
SELECT
    subject_id,
    hadm_id,
    stay_id,
    charttime,
    CASE WHEN itemid = 220210 THEN value ELSE NULL END AS RR
FROM
    mimiciv_icu.chartevents
WHERE
    itemid IN (220210)
		ORDER BY
	subject_id ASC,
	hadm_id ASC,
	stay_id ASC;
Drop TABLE IF EXISTS respiratory_rate_with_label;
CREATE TABLE respiratory_rate_with_label AS
SELECT
	"public".respiratory_rate.subject_id,
	"public".respiratory_rate.hadm_id,
	"public".respiratory_rate.stay_id,
	mimiciv_hosp.admissions.deathtime,
	mimiciv_hosp.admissions.hospital_expire_flag,
	"public".respiratory_rate.charttime,
	"public".respiratory_rate.rr
FROM
	"public".respiratory_rate
	INNER JOIN
	mimiciv_hosp.admissions
	ON
		"public".respiratory_rate.subject_id = mimiciv_hosp.admissions.subject_id AND
		"public".respiratory_rate.hadm_id = mimiciv_hosp.admissions.hadm_id
ORDER BY
	"public".respiratory_rate.subject_id ASC,
	"public".respiratory_rate.hadm_id ASC,
	"public".respiratory_rate.stay_id ASC,
	"public".respiratory_rate.charttime ASC;
Drop TABLE IF EXISTS Respiratory_Rate;

Drop TABLE IF EXISTS SpO2;
CREATE TABLE SpO2 AS
SELECT
    subject_id,
    hadm_id,
    stay_id,
    charttime,
    CASE WHEN itemid = 220277 THEN value ELSE NULL END AS SpO2
FROM
    mimiciv_icu.chartevents
WHERE
    itemid IN (220277)
		ORDER BY
	subject_id ASC,
	hadm_id ASC,
	stay_id ASC;
Drop TABLE IF EXISTS spo2_with_label;
CREATE TABLE spo2_with_label AS
SELECT
	"public".spo2.subject_id,
	"public".spo2.hadm_id,
	"public".spo2.stay_id,
	mimiciv_hosp.admissions.deathtime,
	mimiciv_hosp.admissions.hospital_expire_flag,
	"public".spo2.charttime,
	"public".spo2.spo2
FROM
	"public".spo2
	INNER JOIN
	mimiciv_hosp.admissions
	ON
		"public".spo2.subject_id = mimiciv_hosp.admissions.subject_id AND
		"public".spo2.hadm_id = mimiciv_hosp.admissions.hadm_id
ORDER BY
	"public".spo2.subject_id ASC,
	"public".spo2.hadm_id ASC,
	"public".spo2.stay_id ASC,
	"public".spo2.charttime ASC;
Drop TABLE IF EXISTS SpO2;

Drop TABLE IF EXISTS Temperature ;
CREATE TABLE Temperature  AS
SELECT
    subject_id,
    hadm_id,
    stay_id,
    charttime,
    CASE WHEN itemid = 223762 THEN value ELSE NULL END AS Temperature
FROM
    mimiciv_icu.chartevents
WHERE
    itemid IN (223762)
		ORDER BY
	subject_id ASC,
	hadm_id ASC,
	stay_id ASC;

Drop TABLE IF EXISTS temperature_with_label;
CREATE TABLE temperature_with_label AS
SELECT
	"public".temperature.subject_id,
	"public".temperature.hadm_id,
	"public".temperature.stay_id,
	mimiciv_hosp.admissions.deathtime,
	mimiciv_hosp.admissions.hospital_expire_flag,
	"public".temperature.charttime,
	"public".temperature.temperature
FROM
	"public".temperature
	INNER JOIN
	mimiciv_hosp.admissions
	ON
		"public".temperature.subject_id = mimiciv_hosp.admissions.subject_id AND
		"public".temperature.hadm_id = mimiciv_hosp.admissions.hadm_id
ORDER BY
	"public".temperature.subject_id ASC,
	"public".temperature.hadm_id ASC,
	"public".temperature.stay_id ASC,
	"public".temperature.charttime ASC;

Drop TABLE IF EXISTS Temperature;
