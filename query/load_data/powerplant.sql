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

#mysql 
CREATE  TABLE  pp10k( 
T            DOUBLE,
EV           DOUBLE,
AP           DOUBLE,
RH           DOUBLE,
EP           DOUBLE
);

CREATE  TABLE  pp100k( 
T            DOUBLE,
EV           DOUBLE,
AP           DOUBLE,
RH           DOUBLE,
EP           DOUBLE
);


# load the file;
LOAD DATA INFILE "/var/lib/mysql-files/powerplant10k.csv"
INTO TABLE pp10k
COLUMNS TERMINATED BY ','
LINES TERMINATED BY '\n';
LOAD DATA INFILE "/var/lib/mysql-files/powerplant100k.csv"
INTO TABLE pp100k
COLUMNS TERMINATED BY ','
LINES TERMINATED BY '\n';

#query to mysql
mysql -u root -p dbest < query.sql  > 10k.txt



#blinkdb



# mysql verdictdb
CREATE  TABLE  powerplant( 
T            DOUBLE,
EV           DOUBLE,
AP           DOUBLE,
RH           DOUBLE,
EP           DOUBLE
);

LOAD DATA INFILE "/var/lib/mysql-files/powerplant.csv"
INTO TABLE powerplant
COLUMNS TERMINATED BY ','
LINES TERMINATED BY '\n';