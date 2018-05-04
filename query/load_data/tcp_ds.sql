CREATE EXTERNAL TABLE store_sales ( ss_sold_date_sk           INT,
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
LOCATION '/hive/data/store_sales.dat';

LOAD DATA INPATH 'hdfs:/data/store_sales.dat' INTO TABLE store_sales;

# select the x y columns
CREATE TABLE xy AS SELECT ss_wholesale_cost,ss_list_price  FROM store_sales;
# remove the null values
CREATE TABLE price_cost AS SELECT ss_list_price,ss_wholesale_cost FROM xy
    WHERE ss_list_price IS NOT NULL
    AND ss_wholesale_cost IS NOT NULL;
# the two sql commands above could be merged to one:
CREATE TABLE price_cost AS SELECT ss_list_price,ss_wholesale_cost FROM store_sales
    WHERE ss_list_price IS NOT NULL
    AND ss_wholesale_cost IS NOT NULL;



#hive -e 'set hive.cli.print.header=true; select * from xy' | sed 's/[\t]/,/g'  > /home/u1796377/Desktop/xy_with_header.csv

#hive -e 'select * from xy' | sed 's/[\t]/,/g'  > /home/u1796377/Desktop/xy_without_header.csv

# export the table to csv file.
INSERT OVERWRITE LOCAL DIRECTORY '/disk/hadoopDir/warehouse'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
select * from price_cost;


# load table to mysql
# fisrt start the mysql server client
create table price_cost_sample_1000000 (
ss_list_price DOUBLE,
ss_wholesale_cost DOUBLE
);
# copy the table to the directory to be submitted to mysql
sudo cp /disk/hadoopDir/warehouse/sample.csv /var/lib/mysql-files/sample.csv
# load the file;
LOAD DATA INFILE "/var/lib/mysql-files/sample.csv"
INTO TABLE price_cost_sample_1000000
COLUMNS TERMINATED BY ','
LINES TERMINATED BY '\n';
