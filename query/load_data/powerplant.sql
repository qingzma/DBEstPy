# HIVE
CREATE EXTERNAL TABLE  IF NOT EXISTS powerplant( 
T            DOUBLE,
EV           DOUBLE,
AP           DOUBLE,
RH           DOUBLE,
EP           DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
LOCATION '/user/hive/warehouse/powerplant/';

