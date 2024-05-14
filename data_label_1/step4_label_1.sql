Drop TABLE IF EXISTS merged_data_4;
CREATE TABLE merged_data_4 AS
SELECT
	CAST("filled_data_1_20"."stay_id" AS numeric) AS stay_id_1,
	CAST("filled_data_1_20"."time" AS numeric) AS time_1,
	CAST("filled_data_1_20"."sbp" AS numeric) AS sbp,
	CAST("filled_data_1_20"."dbp" AS numeric) AS dbp,
	CAST("filled_data_1_20"."map" AS numeric) AS "map",
	CAST("filled_data_1_20"."label" AS numeric) AS label_1,

	CAST("filled_data_2_20"."stay_id" AS numeric) AS stay_id_2,
	CAST("filled_data_2_20"."time" AS numeric) AS time_2,
	CAST("filled_data_2_20"."hr" AS numeric) AS hr,
	CAST("filled_data_2_20"."label" AS numeric) AS label_2,

	CAST("filled_data_3_20"."stay_id" AS numeric) AS stay_id_3,
	CAST("filled_data_3_20"."time" AS numeric) AS time_3,
	CAST("filled_data_3_20"."rr" AS numeric) AS rr,
	CAST("filled_data_3_20"."label" AS numeric) AS label_3,

	CAST("filled_data_4_20"."stay_id" AS numeric) AS stay_id_4,
	CAST("filled_data_4_20"."time" AS numeric) AS time_4,
	CAST("filled_data_4_20"."spo2" AS numeric) AS spo2,
	CAST("filled_data_4_20"."label" AS numeric) AS label_4,

	CAST("filled_data_5_20"."stay_id" AS numeric) AS stay_id_5,
	CAST("filled_data_5_20"."time" AS numeric) AS time_5,
	CAST("filled_data_5_20"."temp" AS numeric) AS "temp",
	CAST("filled_data_5_20"."label" AS numeric) AS label_5
FROM
	"filled_data_1_20"
	FULL OUTER JOIN
	"filled_data_2_20"
	ON
		"filled_data_1_20".stay_id = "filled_data_2_20".stay_id AND
		"filled_data_1_20"."time" = "filled_data_2_20"."time"
	FULL OUTER JOIN
	"filled_data_3_20"
	ON
		"filled_data_2_20".stay_id = "filled_data_3_20".stay_id AND
		"filled_data_2_20"."time" = "filled_data_3_20"."time"
	FULL OUTER JOIN
	"filled_data_4_20"
	ON
		"filled_data_3_20".stay_id = "filled_data_4_20".stay_id AND
		"filled_data_3_20"."time" = "filled_data_4_20"."time"
	FULL OUTER JOIN
	"filled_data_5_20"
	ON
		"filled_data_4_20".stay_id = "filled_data_5_20".stay_id AND
		"filled_data_4_20"."time" = "filled_data_5_20"."time"
ORDER BY
	stay_id_1 ASC,
	time_1 ASC;

Drop TABLE IF EXISTS merged_data_6;
CREATE TABLE merged_data_6 AS
SELECT
	CAST("filled_data_1_30"."stay_id" AS numeric) AS stay_id_1,
	CAST("filled_data_1_30"."time" AS numeric) AS time_1,
	CAST("filled_data_1_30"."sbp" AS numeric) AS sbp,
	CAST("filled_data_1_30"."dbp" AS numeric) AS dbp,
	CAST("filled_data_1_30"."map" AS numeric) AS "map",
	CAST("filled_data_1_30"."label" AS numeric) AS label_1,

	CAST("filled_data_2_30"."stay_id" AS numeric) AS stay_id_2,
	CAST("filled_data_2_30"."time" AS numeric) AS time_2,
	CAST("filled_data_2_30"."hr" AS numeric) AS hr,
	CAST("filled_data_2_30"."label" AS numeric) AS label_2,

	CAST("filled_data_3_30"."stay_id" AS numeric) AS stay_id_3,
	CAST("filled_data_3_30"."time" AS numeric) AS time_3,
	CAST("filled_data_3_30"."rr" AS numeric) AS rr,
	CAST("filled_data_3_30"."label" AS numeric) AS label_3,

	CAST("filled_data_4_30"."stay_id" AS numeric) AS stay_id_4,
	CAST("filled_data_4_30"."time" AS numeric) AS time_4,
	CAST("filled_data_4_30"."spo2" AS numeric) AS spo2,
	CAST("filled_data_4_30"."label" AS numeric) AS label_4,

	CAST("filled_data_5_30"."stay_id" AS numeric) AS stay_id_5,
	CAST("filled_data_5_30"."time" AS numeric) AS time_5,
	CAST("filled_data_5_30"."temp" AS numeric) AS "temp",
	CAST("filled_data_5_30"."label" AS numeric) AS label_5
FROM
	"filled_data_1_30"
	FULL OUTER JOIN
	"filled_data_2_30"
	ON
		"filled_data_1_30".stay_id = "filled_data_2_30".stay_id AND
		"filled_data_1_30"."time" = "filled_data_2_30"."time"
	FULL OUTER JOIN
	"filled_data_3_30"
	ON
		"filled_data_2_30".stay_id = "filled_data_3_30".stay_id AND
		"filled_data_2_30"."time" = "filled_data_3_30"."time"
	FULL OUTER JOIN
	"filled_data_4_30"
	ON
		"filled_data_3_30".stay_id = "filled_data_4_30".stay_id AND
		"filled_data_3_30"."time" = "filled_data_4_30"."time"
	FULL OUTER JOIN
	"filled_data_5_30"
	ON
		"filled_data_4_30".stay_id = "filled_data_5_30".stay_id AND
		"filled_data_4_30"."time" = "filled_data_5_30"."time"
ORDER BY
	stay_id_1 ASC,
	time_1 ASC;

Drop TABLE IF EXISTS merged_data_8;
CREATE TABLE merged_data_8 AS
SELECT
	CAST("filled_data_1_36"."stay_id" AS numeric) AS stay_id_1,
	CAST("filled_data_1_36"."time" AS numeric) AS time_1,
	CAST("filled_data_1_36"."sbp" AS numeric) AS sbp,
	CAST("filled_data_1_36"."dbp" AS numeric) AS dbp,
	CAST("filled_data_1_36"."map" AS numeric) AS "map",
	CAST("filled_data_1_36"."label" AS numeric) AS label_1,

	CAST("filled_data_2_36"."stay_id" AS numeric) AS stay_id_2,
	CAST("filled_data_2_36"."time" AS numeric) AS time_2,
	CAST("filled_data_2_36"."hr" AS numeric) AS hr,
	CAST("filled_data_2_36"."label" AS numeric) AS label_2,

	CAST("filled_data_3_36"."stay_id" AS numeric) AS stay_id_3,
	CAST("filled_data_3_36"."time" AS numeric) AS time_3,
	CAST("filled_data_3_36"."rr" AS numeric) AS rr,
	CAST("filled_data_3_36"."label" AS numeric) AS label_3,

	CAST("filled_data_4_36"."stay_id" AS numeric) AS stay_id_4,
	CAST("filled_data_4_36"."time" AS numeric) AS time_4,
	CAST("filled_data_4_36"."spo2" AS numeric) AS spo2,
	CAST("filled_data_4_36"."label" AS numeric) AS label_4,

	CAST("filled_data_5_36"."stay_id" AS numeric) AS stay_id_5,
	CAST("filled_data_5_36"."time" AS numeric) AS time_5,
	CAST("filled_data_5_36"."temp" AS numeric) AS "temp",
	CAST("filled_data_5_36"."label" AS numeric) AS label_5
FROM
	"filled_data_1_36"
	FULL OUTER JOIN
	"filled_data_2_36"
	ON
		"filled_data_1_36".stay_id = "filled_data_2_36".stay_id AND
		"filled_data_1_36"."time" = "filled_data_2_36"."time"
	FULL OUTER JOIN
	"filled_data_3_36"
	ON
		"filled_data_2_36".stay_id = "filled_data_3_36".stay_id AND
		"filled_data_2_36"."time" = "filled_data_3_36"."time"
	FULL OUTER JOIN
	"filled_data_4_36"
	ON
		"filled_data_3_36".stay_id = "filled_data_4_36".stay_id AND
		"filled_data_3_36"."time" = "filled_data_4_36"."time"
	FULL OUTER JOIN
	"filled_data_5_36"
	ON
		"filled_data_4_36".stay_id = "filled_data_5_36".stay_id AND
		"filled_data_4_36"."time" = "filled_data_5_36"."time"
ORDER BY
	stay_id_1 ASC,
	time_1 ASC;

Drop TABLE IF EXISTS merged_data_12;
CREATE TABLE merged_data_12 AS
SELECT
	CAST("filled_data_1_48"."stay_id" AS numeric) AS stay_id_1,
	CAST("filled_data_1_48"."time" AS numeric) AS time_1,
	CAST("filled_data_1_48"."sbp" AS numeric) AS sbp,
	CAST("filled_data_1_48"."dbp" AS numeric) AS dbp,
	CAST("filled_data_1_48"."map" AS numeric) AS "map",
	CAST("filled_data_1_48"."label" AS numeric) AS label_1,

	CAST("filled_data_2_48"."stay_id" AS numeric) AS stay_id_2,
	CAST("filled_data_2_48"."time" AS numeric) AS time_2,
	CAST("filled_data_2_48"."hr" AS numeric) AS hr,
	CAST("filled_data_2_48"."label" AS numeric) AS label_2,

	CAST("filled_data_3_48"."stay_id" AS numeric) AS stay_id_3,
	CAST("filled_data_3_48"."time" AS numeric) AS time_3,
	CAST("filled_data_3_48"."rr" AS numeric) AS rr,
	CAST("filled_data_3_48"."label" AS numeric) AS label_3,

	CAST("filled_data_4_48"."stay_id" AS numeric) AS stay_id_4,
	CAST("filled_data_4_48"."time" AS numeric) AS time_4,
	CAST("filled_data_4_48"."spo2" AS numeric) AS spo2,
	CAST("filled_data_4_48"."label" AS numeric) AS label_4,

	CAST("filled_data_5_48"."stay_id" AS numeric) AS stay_id_5,
	CAST("filled_data_5_48"."time" AS numeric) AS time_5,
	CAST("filled_data_5_48"."temp" AS numeric) AS "temp",
	CAST("filled_data_5_48"."label" AS numeric) AS label_5
FROM
	"filled_data_1_48"
	FULL OUTER JOIN
	"filled_data_2_48"
	ON
		"filled_data_1_48".stay_id = "filled_data_2_48".stay_id AND
		"filled_data_1_48"."time" = "filled_data_2_48"."time"
	FULL OUTER JOIN
	"filled_data_3_48"
	ON
		"filled_data_2_48".stay_id = "filled_data_3_48".stay_id AND
		"filled_data_2_48"."time" = "filled_data_3_48"."time"
	FULL OUTER JOIN
	"filled_data_4_48"
	ON
		"filled_data_3_48".stay_id = "filled_data_4_48".stay_id AND
		"filled_data_3_48"."time" = "filled_data_4_48"."time"
	FULL OUTER JOIN
	"filled_data_5_48"
	ON
		"filled_data_4_48".stay_id = "filled_data_5_48".stay_id AND
		"filled_data_4_48"."time" = "filled_data_5_48"."time"
ORDER BY
	stay_id_1 ASC,
	time_1 ASC;

Drop TABLE IF EXISTS merged_data_24;
CREATE TABLE merged_data_24 AS
SELECT
	CAST("filled_data_1_24"."stay_id" AS numeric) AS stay_id_1,
	CAST("filled_data_1_24"."time" AS numeric) AS time_1,
	CAST("filled_data_1_24"."sbp" AS numeric) AS sbp,
	CAST("filled_data_1_24"."dbp" AS numeric) AS dbp,
	CAST("filled_data_1_24"."map" AS numeric) AS "map",
	CAST("filled_data_1_24"."label" AS numeric) AS label_1,

	CAST("filled_data_2_24"."stay_id" AS numeric) AS stay_id_2,
	CAST("filled_data_2_24"."time" AS numeric) AS time_2,
	CAST("filled_data_2_24"."hr" AS numeric) AS hr,
	CAST("filled_data_2_24"."label" AS numeric) AS label_2,

	CAST("filled_data_3_24"."stay_id" AS numeric) AS stay_id_3,
	CAST("filled_data_3_24"."time" AS numeric) AS time_3,
	CAST("filled_data_3_24"."rr" AS numeric) AS rr,
	CAST("filled_data_3_24"."label" AS numeric) AS label_3,

	CAST("filled_data_4_24"."stay_id" AS numeric) AS stay_id_4,
	CAST("filled_data_4_24"."time" AS numeric) AS time_4,
	CAST("filled_data_4_24"."spo2" AS numeric) AS spo2,
	CAST("filled_data_4_24"."label" AS numeric) AS label_4,

	CAST("filled_data_5_24"."stay_id" AS numeric) AS stay_id_5,
	CAST("filled_data_5_24"."time" AS numeric) AS time_5,
	CAST("filled_data_5_24"."temp" AS numeric) AS "temp",
	CAST("filled_data_5_24"."label" AS numeric) AS label_5
FROM
	"filled_data_1_24"
	FULL OUTER JOIN
	"filled_data_2_24"
	ON
		"filled_data_1_24".stay_id = "filled_data_2_24".stay_id AND
		"filled_data_1_24"."time" = "filled_data_2_24"."time"
	FULL OUTER JOIN
	"filled_data_3_24"
	ON
		"filled_data_2_24".stay_id = "filled_data_3_24".stay_id AND
		"filled_data_2_24"."time" = "filled_data_3_24"."time"
	FULL OUTER JOIN
	"filled_data_4_24"
	ON
		"filled_data_3_24".stay_id = "filled_data_4_24".stay_id AND
		"filled_data_3_24"."time" = "filled_data_4_24"."time"
	FULL OUTER JOIN
	"filled_data_5_24"
	ON
		"filled_data_4_24".stay_id = "filled_data_5_24".stay_id AND
		"filled_data_4_24"."time" = "filled_data_5_24"."time"
ORDER BY
	stay_id_1 ASC,
	time_1 ASC;