CREATE EXTERNAL TABLE store_sales_1t ( ss_sold_date_sk           INT,
ss_sold_time_sk           INT,
ss_item_sk                INT,
ss_customer_sk            INT,
ss_cdemo_sk               INT,
ss_hdemo_sk               INT,
ss_addr_sk                INT,
ss_store_sk               INT,
ss_promo_sk               INT,
ss_ticket_number          INT,
ss_quantity               INT,
ss_wholesale_cost         DECIMAL(7,2),
ss_list_price             DECIMAL(7,2),
ss_sales_price            DECIMAL(7,2),
ss_ext_discount_amt       DECIMAL(7,2),
ss_ext_sales_price        DECIMAL(7,2),
ss_ext_wholesale_cost     DECIMAL(7,2),
ss_ext_list_price         DECIMAL(7,2),
ss_ext_tax                DECIMAL(7,2),
ss_coupon_amt             DECIMAL(7,2),
ss_net_paid               DECIMAL(7,2),
ss_net_paid_inc_tax       DECIMAL(7,2),
ss_net_profit             DECIMAL(7,2)
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '|'
LOCATION '/user/hive/warehouse/store_sales_1t.dat';

LOAD DATA INPATH 'hdfs:/data/store_sales_1g.dat' INTO TABLE store_sales_1g;
LOAD DATA INPATH 'hdfs:/data/1T.dat' INTO TABLE store_sales_1t;


# select the x y columns
CREATE TABLE xy AS SELECT ss_wholesale_cost,ss_list_price  FROM store_sales;
# remove the null values
CREATE TABLE price_cost AS SELECT ss_list_price,ss_wholesale_cost FROM xy
    WHERE ss_list_price IS NOT NULL
    AND ss_wholesale_cost IS NOT NULL;
# the two sql commands above could be merged to one:
CREATE TABLE price_cost_1t AS SELECT ss_list_price,ss_wholesale_cost FROM store_sales_1t
    WHERE ss_list_price IS NOT NULL
    AND ss_wholesale_cost IS NOT NULL;



#hive -e 'set hive.cli.print.header=true; select * from xy' | sed 's/[\t]/,/g'  > /home/u1796377/Desktop/xy_with_header.csv

#hive -e 'select * from xy' | sed 's/[\t]/,/g'  > /home/u1796377/Desktop/xy_without_header.csv

# export the table to csv file.
INSERT OVERWRITE LOCAL DIRECTORY '1t'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
select * from price_cost_1t;


# load table to mysql
# fisrt start the mysql server client
create table price_cost_1t_sample_5m (
ss_list_price DOUBLE,
ss_wholesale_cost DOUBLE
);
# copy the table to the directory to be submitted to mysql
sudo cp /disk/hadoopDir/warehouse/sample.csv /var/lib/mysql-files/sample.csv
# load the file;
LOAD DATA INFILE "/var/lib/mysql-files/1t.csv"
INTO TABLE price_cost_1t
COLUMNS TERMINATED BY ','
LINES TERMINATED BY '\n';






CREATE EXTERNAL TABLE store_sales_5m (
ss_list_price             DECIMAL(7,2),
ss_wholesale_cost         DECIMAL(7,2)
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '|'
LOCATION '/user/hive/warehouse/price_cost_1t_sample_5m.dat';

LOAD DATA INPATH 'hdfs:/data/store_sales_1g.dat' INTO TABLE store_sales_1g;
LOAD DATA INPATH 'hdfs:/data/5m.csv' INTO TABLE price_cost_1t_sample_5m;


#
# create the sorted dataset
CREATE TABLE price_cost_1t_sorted
AS SELECT * FROM price_cost_1t
ORDER BY ss_list_price;

sort --parallel=8 -g  -o  1t_sorted.csv 1t.csv
sort -k 1,1 -g  -o  1t_sorted.csv 1t.csv

# export the table to csv file.
INSERT OVERWRITE LOCAL DIRECTORY '1t_sorted'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
select * from price_cost_1t_sorted;


# copy the file to csv, and load the data
# fisrt start the mysql server client
create table price_cost_1t_sorted (
ss_list_price DOUBLE,
ss_wholesale_cost DOUBLE
);
# copy the table to the directory to be submitted to mysql
sudo cp /disk/hadoopDir/warehouse/price_cost_1t_sorted.csv /var/lib/mysql-files/price_cost_1t_sorted.csv
# load the file;
LOAD DATA INFILE "/var/lib/mysql-files/price_cost_1t_sorted.csv"
INTO TABLE price_cost_1t_sorted
COLUMNS TERMINATED BY ','
LINES TERMINATED BY '\n';



# Create the index
CREATE INDEX idx_ss_list_price
ON price_cost_1t_sorted(ss_list_price);