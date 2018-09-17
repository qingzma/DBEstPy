# HIVE
CREATE EXTERNAL TABLE  IF NOT EXISTS ETrade( T_ID           INT,
T_DTS           CHAR(23),
T_ST_ID         CHAR(4),
T_TT_ID         CHAR(3),
T_IS_CASH       INT,
T_S_SYMB        CHAR(15),
T_QTY           INT,
T_BID_PRICE     DECIMAL(7,2),
T_CA_ID         INT,
T_EXEC_NAME     CHAR(49),
T_TRADE_PRICE   DECIMAL(7,2),
T_CHRG          DECIMAL(7,2),
T_COMM          DECIMAL(7,2),
T_TAX           DECIMAL(7,2),
T_LIFO          INT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '|'
LOCATION '/user/hive/warehouse/tpce';



CREATE TABLE etrade_price_comm AS SELECT T_TRADE_PRICE,T_COMM FROM ETrade
    WHERE T_TRADE_PRICE IS NOT NULL
    AND T_COMM IS NOT NULL;


CREATE TABLE etrade_price_charge AS SELECT T_TRADE_PRICE,T_CHRG FROM ETrade
    WHERE T_TRADE_PRICE IS NOT NULL
    AND T_CHRG IS NOT NULL;


# export the table to csv file.
INSERT OVERWRITE LOCAL DIRECTORY 'etrade_price_comm'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
select * from etrade_price_comm;

# export the table to csv file.
INSERT OVERWRITE LOCAL DIRECTORY 'etrade_price_charge'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
select * from etrade_price_charge;


# merge splits into one csv file
cat 000* > etrade_price_comm.csv
cat 000* > etrade_price_charge.csv

# create sample here


# sort
sort -k 1,1 -g  -o  etrade_price_comm_sorted.csv etrade_price_comm.csv
sort -k 1,1 -g  -o  etrade_price_charge_sorted.csv etrade_price_charge.csv



## in mysql, create the corresponding full tables

create table etrade_price_comm (
T_TRADE_PRICE DOUBLE,
T_COMM DOUBLE
);

create table etrade_price_charge (
T_TRADE_PRICE DOUBLE,
T_CHRG DOUBLE
);

# copy the table to the directory to be submitted to mysql
sudo mv /home/hduser/etrade_price_comm/etrade_price_comm_sorted.csv /var/lib/mysql-files/etrade_price_comm_sorted.csv
sudo mv /home/hduser/etrade_price_charge/etrade_price_charge_sorted.csv /var/lib/mysql-files/etrade_price_charge_sorted.csv
# load the file;
LOAD DATA INFILE "/var/lib/mysql-files/etrade_price_comm_sorted.csv"
INTO TABLE etrade_price_comm
COLUMNS TERMINATED BY ','
LINES TERMINATED BY '\n';

LOAD DATA INFILE "/var/lib/mysql-files/etrade_price_charge_sorted.csv"
INTO TABLE etrade_price_charge
COLUMNS TERMINATED BY ','
LINES TERMINATED BY '\n';

# Create the index
CREATE INDEX idx_T_TRADE_PRICE
ON etrade_price_comm(T_TRADE_PRICE);

CREATE INDEX idx_T_TRADE_PRICE
ON etrade_price_charge(T_TRADE_PRICE);


# insert headers to the csv file
sed -i '1s/^/T_TRADE_PRICE,T_COMM\n/' etrade_price_comm_1m_sorted.csv
sed -i '1s/^/T_TRADE_PRICE,T_CHRG\n/' etrade_price_charge_1m.csv



2160000000