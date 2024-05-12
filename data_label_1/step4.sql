Drop TABLE IF EXISTS merged_data_4;
CREATE TABLE merged_data_4 AS
SELECT
	CAST("filled_data_1-4"."stay_id" AS numeric) AS stay_id_1,
	CAST("filled_data_1-4"."time" AS numeric) AS time_1,
	CAST("filled_data_1-4"."sbp" AS numeric) AS sbp,
	CAST("filled_data_1-4"."dbp" AS numeric) AS dbp,
	CAST("filled_data_1-4"."map" AS numeric) AS "map",
	CAST("filled_data_1-4"."label" AS numeric) AS label_1,

	CAST("filled_data_2-4"."stay_id" AS numeric) AS stay_id_2,
	CAST("filled_data_2-4"."time" AS numeric) AS time_2,
	CAST("filled_data_2-4"."hr" AS numeric) AS hr,
	CAST("filled_data_2-4"."label" AS numeric) AS label_2,

	CAST("filled_data_3-4"."stay_id" AS numeric) AS stay_id_3,
	CAST("filled_data_3-4"."time" AS numeric) AS time_3,
	CAST("filled_data_3-4"."rr" AS numeric) AS rr,
	CAST("filled_data_3-4"."label" AS numeric) AS label_3,

	CAST("filled_data_4-4"."stay_id" AS numeric) AS stay_id_4,
	CAST("filled_data_4-4"."time" AS numeric) AS time_4,
	CAST("filled_data_4-4"."spo2" AS numeric) AS spo2,
	CAST("filled_data_4-4"."label" AS numeric) AS label_4,

	CAST("filled_data_5-4"."stay_id" AS numeric) AS stay_id_5,
	CAST("filled_data_5-4"."time" AS numeric) AS time_5,
	CAST("filled_data_5-4"."temp" AS numeric) AS "temp",
	CAST("filled_data_5-4"."label" AS numeric) AS label_5
FROM
	"filled_data_1-4"
	FULL OUTER JOIN
	"filled_data_2-4"
	ON
		"filled_data_1-4".stay_id = "filled_data_2-4".stay_id AND
		"filled_data_1-4"."time" = "filled_data_2-4"."time"
	FULL OUTER JOIN
	"filled_data_3-4"
	ON
		"filled_data_2-4".stay_id = "filled_data_3-4".stay_id AND
		"filled_data_2-4"."time" = "filled_data_3-4"."time"
	FULL OUTER JOIN
	"filled_data_4-4"
	ON
		"filled_data_3-4".stay_id = "filled_data_4-4".stay_id AND
		"filled_data_3-4"."time" = "filled_data_4-4"."time"
	FULL OUTER JOIN
	"filled_data_5-4"
	ON
		"filled_data_4-4".stay_id = "filled_data_5-4".stay_id AND
		"filled_data_4-4"."time" = "filled_data_5-4"."time"
ORDER BY
	stay_id_1 ASC,
	time_1 ASC;

Drop TABLE IF EXISTS merged_data_6;
CREATE TABLE merged_data_6 AS
SELECT
	CAST("filled_data_1-6"."stay_id" AS numeric) AS stay_id_1,
	CAST("filled_data_1-6"."time" AS numeric) AS time_1,
	CAST("filled_data_1-6"."sbp" AS numeric) AS sbp,
	CAST("filled_data_1-6"."dbp" AS numeric) AS dbp,
	CAST("filled_data_1-6"."map" AS numeric) AS "map",
	CAST("filled_data_1-6"."label" AS numeric) AS label_1,

	CAST("filled_data_2-6"."stay_id" AS numeric) AS stay_id_2,
	CAST("filled_data_2-6"."time" AS numeric) AS time_2,
	CAST("filled_data_2-6"."hr" AS numeric) AS hr,
	CAST("filled_data_2-6"."label" AS numeric) AS label_2,

	CAST("filled_data_3-6"."stay_id" AS numeric) AS stay_id_3,
	CAST("filled_data_3-6"."time" AS numeric) AS time_3,
	CAST("filled_data_3-6"."rr" AS numeric) AS rr,
	CAST("filled_data_3-6"."label" AS numeric) AS label_3,

	CAST("filled_data_4-6"."stay_id" AS numeric) AS stay_id_4,
	CAST("filled_data_4-6"."time" AS numeric) AS time_4,
	CAST("filled_data_4-6"."spo2" AS numeric) AS spo2,
	CAST("filled_data_4-6"."label" AS numeric) AS label_4,

	CAST("filled_data_5-6"."stay_id" AS numeric) AS stay_id_5,
	CAST("filled_data_5-6"."time" AS numeric) AS time_5,
	CAST("filled_data_5-6"."temp" AS numeric) AS "temp",
	CAST("filled_data_5-6"."label" AS numeric) AS label_5
FROM
	"filled_data_1-6"
	FULL OUTER JOIN
	"filled_data_2-6"
	ON
		"filled_data_1-6".stay_id = "filled_data_2-6".stay_id AND
		"filled_data_1-6"."time" = "filled_data_2-6"."time"
	FULL OUTER JOIN
	"filled_data_3-6"
	ON
		"filled_data_2-6".stay_id = "filled_data_3-6".stay_id AND
		"filled_data_2-6"."time" = "filled_data_3-6"."time"
	FULL OUTER JOIN
	"filled_data_4-6"
	ON
		"filled_data_3-6".stay_id = "filled_data_4-6".stay_id AND
		"filled_data_3-6"."time" = "filled_data_4-6"."time"
	FULL OUTER JOIN
	"filled_data_5-6"
	ON
		"filled_data_4-6".stay_id = "filled_data_5-6".stay_id AND
		"filled_data_4-6"."time" = "filled_data_5-6"."time"
ORDER BY
	stay_id_1 ASC,
	time_1 ASC;

Drop TABLE IF EXISTS merged_data_8;
CREATE TABLE merged_data_8 AS
SELECT
	CAST("filled_data_1-8"."stay_id" AS numeric) AS stay_id_1,
	CAST("filled_data_1-8"."time" AS numeric) AS time_1,
	CAST("filled_data_1-8"."sbp" AS numeric) AS sbp,
	CAST("filled_data_1-8"."dbp" AS numeric) AS dbp,
	CAST("filled_data_1-8"."map" AS numeric) AS "map",
	CAST("filled_data_1-8"."label" AS numeric) AS label_1,

	CAST("filled_data_2-8"."stay_id" AS numeric) AS stay_id_2,
	CAST("filled_data_2-8"."time" AS numeric) AS time_2,
	CAST("filled_data_2-8"."hr" AS numeric) AS hr,
	CAST("filled_data_2-8"."label" AS numeric) AS label_2,

	CAST("filled_data_3-8"."stay_id" AS numeric) AS stay_id_3,
	CAST("filled_data_3-8"."time" AS numeric) AS time_3,
	CAST("filled_data_3-8"."rr" AS numeric) AS rr,
	CAST("filled_data_3-8"."label" AS numeric) AS label_3,

	CAST("filled_data_4-8"."stay_id" AS numeric) AS stay_id_4,
	CAST("filled_data_4-8"."time" AS numeric) AS time_4,
	CAST("filled_data_4-8"."spo2" AS numeric) AS spo2,
	CAST("filled_data_4-8"."label" AS numeric) AS label_4,

	CAST("filled_data_5-8"."stay_id" AS numeric) AS stay_id_5,
	CAST("filled_data_5-8"."time" AS numeric) AS time_5,
	CAST("filled_data_5-8"."temp" AS numeric) AS "temp",
	CAST("filled_data_5-8"."label" AS numeric) AS label_5
FROM
	"filled_data_1-8"
	FULL OUTER JOIN
	"filled_data_2-8"
	ON
		"filled_data_1-8".stay_id = "filled_data_2-8".stay_id AND
		"filled_data_1-8"."time" = "filled_data_2-8"."time"
	FULL OUTER JOIN
	"filled_data_3-8"
	ON
		"filled_data_2-8".stay_id = "filled_data_3-8".stay_id AND
		"filled_data_2-8"."time" = "filled_data_3-8"."time"
	FULL OUTER JOIN
	"filled_data_4-8"
	ON
		"filled_data_3-8".stay_id = "filled_data_4-8".stay_id AND
		"filled_data_3-8"."time" = "filled_data_4-8"."time"
	FULL OUTER JOIN
	"filled_data_5-8"
	ON
		"filled_data_4-8".stay_id = "filled_data_5-8".stay_id AND
		"filled_data_4-8"."time" = "filled_data_5-8"."time"
ORDER BY
	stay_id_1 ASC,
	time_1 ASC;

Drop TABLE IF EXISTS merged_data_12;
CREATE TABLE merged_data_12 AS
SELECT
	CAST("filled_data_1-12"."stay_id" AS numeric) AS stay_id_1,
	CAST("filled_data_1-12"."time" AS numeric) AS time_1,
	CAST("filled_data_1-12"."sbp" AS numeric) AS sbp,
	CAST("filled_data_1-12"."dbp" AS numeric) AS dbp,
	CAST("filled_data_1-12"."map" AS numeric) AS "map",
	CAST("filled_data_1-12"."label" AS numeric) AS label_1,

	CAST("filled_data_2-12"."stay_id" AS numeric) AS stay_id_2,
	CAST("filled_data_2-12"."time" AS numeric) AS time_2,
	CAST("filled_data_2-12"."hr" AS numeric) AS hr,
	CAST("filled_data_2-12"."label" AS numeric) AS label_2,

	CAST("filled_data_3-12"."stay_id" AS numeric) AS stay_id_3,
	CAST("filled_data_3-12"."time" AS numeric) AS time_3,
	CAST("filled_data_3-12"."rr" AS numeric) AS rr,
	CAST("filled_data_3-12"."label" AS numeric) AS label_3,

	CAST("filled_data_4-12"."stay_id" AS numeric) AS stay_id_4,
	CAST("filled_data_4-12"."time" AS numeric) AS time_4,
	CAST("filled_data_4-12"."spo2" AS numeric) AS spo2,
	CAST("filled_data_4-12"."label" AS numeric) AS label_4,

	CAST("filled_data_5-12"."stay_id" AS numeric) AS stay_id_5,
	CAST("filled_data_5-12"."time" AS numeric) AS time_5,
	CAST("filled_data_5-12"."temp" AS numeric) AS "temp",
	CAST("filled_data_5-12"."label" AS numeric) AS label_5
FROM
	"filled_data_1-12"
	FULL OUTER JOIN
	"filled_data_2-12"
	ON
		"filled_data_1-12".stay_id = "filled_data_2-12".stay_id AND
		"filled_data_1-12"."time" = "filled_data_2-12"."time"
	FULL OUTER JOIN
	"filled_data_3-12"
	ON
		"filled_data_2-12".stay_id = "filled_data_3-12".stay_id AND
		"filled_data_2-12"."time" = "filled_data_3-12"."time"
	FULL OUTER JOIN
	"filled_data_4-12"
	ON
		"filled_data_3-12".stay_id = "filled_data_4-12".stay_id AND
		"filled_data_3-12"."time" = "filled_data_4-12"."time"
	FULL OUTER JOIN
	"filled_data_5-12"
	ON
		"filled_data_4-12".stay_id = "filled_data_5-12".stay_id AND
		"filled_data_4-12"."time" = "filled_data_5-12"."time"
ORDER BY
	stay_id_1 ASC,
	time_1 ASC;

Drop TABLE IF EXISTS merged_data_24;
CREATE TABLE merged_data_24 AS
SELECT
	CAST("filled_data_1-24"."stay_id" AS numeric) AS stay_id_1,
	CAST("filled_data_1-24"."time" AS numeric) AS time_1,
	CAST("filled_data_1-24"."sbp" AS numeric) AS sbp,
	CAST("filled_data_1-24"."dbp" AS numeric) AS dbp,
	CAST("filled_data_1-24"."map" AS numeric) AS "map",
	CAST("filled_data_1-24"."label" AS numeric) AS label_1,

	CAST("filled_data_2-24"."stay_id" AS numeric) AS stay_id_2,
	CAST("filled_data_2-24"."time" AS numeric) AS time_2,
	CAST("filled_data_2-24"."hr" AS numeric) AS hr,
	CAST("filled_data_2-24"."label" AS numeric) AS label_2,

	CAST("filled_data_3-24"."stay_id" AS numeric) AS stay_id_3,
	CAST("filled_data_3-24"."time" AS numeric) AS time_3,
	CAST("filled_data_3-24"."rr" AS numeric) AS rr,
	CAST("filled_data_3-24"."label" AS numeric) AS label_3,

	CAST("filled_data_4-24"."stay_id" AS numeric) AS stay_id_4,
	CAST("filled_data_4-24"."time" AS numeric) AS time_4,
	CAST("filled_data_4-24"."spo2" AS numeric) AS spo2,
	CAST("filled_data_4-24"."label" AS numeric) AS label_4,

	CAST("filled_data_5-24"."stay_id" AS numeric) AS stay_id_5,
	CAST("filled_data_5-24"."time" AS numeric) AS time_5,
	CAST("filled_data_5-24"."temp" AS numeric) AS "temp",
	CAST("filled_data_5-24"."label" AS numeric) AS label_5
FROM
	"filled_data_1-24"
	FULL OUTER JOIN
	"filled_data_2-24"
	ON
		"filled_data_1-24".stay_id = "filled_data_2-24".stay_id AND
		"filled_data_1-24"."time" = "filled_data_2-24"."time"
	FULL OUTER JOIN
	"filled_data_3-24"
	ON
		"filled_data_2-24".stay_id = "filled_data_3-24".stay_id AND
		"filled_data_2-24"."time" = "filled_data_3-24"."time"
	FULL OUTER JOIN
	"filled_data_4-24"
	ON
		"filled_data_3-24".stay_id = "filled_data_4-24".stay_id AND
		"filled_data_3-24"."time" = "filled_data_4-24"."time"
	FULL OUTER JOIN
	"filled_data_5-24"
	ON
		"filled_data_4-24".stay_id = "filled_data_5-24".stay_id AND
		"filled_data_4-24"."time" = "filled_data_5-24"."time"
ORDER BY
	stay_id_1 ASC,
	time_1 ASC;