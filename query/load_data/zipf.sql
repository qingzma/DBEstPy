# HIVE
CREATE EXTERNAL TABLE zipf (
x DOUBLE,
y DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
LOCATION '/user/hive/warehouse/zipf';


#blinkdb
CREATE TABLE zipf (
x DOUBLE,
y DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
LOCATION '/user/hive/warehouse/zipf';

create table zifp_10k_cached AS SELECT * FROM zipf SAMPLEWITH 0.0001;
create table zifp_100k_cached AS SELECT * FROM zipf SAMPLEWITH 0.001;