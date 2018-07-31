# for mysql...............................
CREATE  TABLE data8 ( 
timestamp               DOUBLE,
Global_active_power     DOUBLE,
Global_reactive_power   DOUBLE,
Voltage                 DOUBLE,
Global_intensity        DOUBLE,
Sub_metering_1          DOUBLE,
Sub_metering_2          DOUBLE,
Sub_metering_3          DOUBLE,
energy                  DOUBLE
);

# copy the table to the directory to be submitted to mysql
sudo cp /disk/hadoopDir/warehouse/sample.csv /var/lib/mysql-files/sample.csv
# load the file;
LOAD DATA INFILE "/var/lib/mysql-files/8data.txt"
INTO TABLE data8
COLUMNS TERMINATED BY ','
LINES TERMINATED BY '\n';





# export the table to csv file.
INSERT OVERWRITE LOCAL DIRECTORY '/data/whole/pc_whole.csv'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
select * from pc_1m_cached;





# for blinkdb
CREATE  TABLE data8 ( ts DOUBLE,
Global_active_power     DOUBLE,
Global_reactive_power   DOUBLE,
Voltage                 DOUBLE,
Global_intensity        DOUBLE,
Sub_metering_1          DOUBLE,
Sub_metering_2          DOUBLE,
Sub_metering_3          DOUBLE,
energy                  DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
LOCATION '/user/hive/warehouse/data8';

LOAD DATA INPATH '/data/8data.txt' INTO TABLE data8;