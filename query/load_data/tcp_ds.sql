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
create table price_cost_100k (
ss_list_price DOUBLE,
ss_wholesale_cost DOUBLE
);
# copy the table to the directory to be submitted to mysql
sudo cp /disk/hadoopDir/warehouse/price_cost_1t_sorted.csv /var/lib/mysql-files/price_cost_1t_sorted.csv
# load the file;
LOAD DATA INFILE "/var/lib/mysql-files/100k.csv"
INTO TABLE price_cost_100k
COLUMNS TERMINATED BY ','
LINES TERMINATED BY '\n';



# Create the index
CREATE INDEX idx_ss_list_price
ON price_cost_1t_sorted(ss_list_price);

sed -i '1s/^/ss_list_price,ss_wholesale_cost\n/' 100k.csv




CREATE  TABLE store_sales ( ss_sold_date_sk           INT,
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
ss_wholesale_cost         DOUBLE,
ss_list_price             DOUBLE,
ss_sales_price            DOUBLE,
ss_ext_discount_amt       DOUBLE,
ss_ext_sales_price        DOUBLE,
ss_ext_wholesale_cost     DOUBLE,
ss_ext_list_price         DOUBLE,
ss_ext_tax                DOUBLE,
ss_coupon_amt             DOUBLE,
ss_net_paid               DOUBLE,
ss_net_paid_inc_tax       DOUBLE,
ss_net_profit             DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '|'
LOCATION '/user/hive/warehouse/store_sales';


LOAD DATA INPATH '/data/tcp/tcp-ds/1t/store_sales.dat' INTO TABLE store_sales;

LOAD DATA INPATH 'hdfs://137.205.118.65:50075/user/hive/warehouse/store_sales_1t.dat/1T.dat' INTO TABLE store_sales_1t;
LOAD DATA INPATH 'hdfs:/user/hive/warehouse/store_sales_1t.dat/1T.dat' INTO TABLE store_sales_1t;
-- LOAD DATA INPATH '/user/hive/warehouse/store_sales_1t.dat/1T.dat' INTO TABLE store_sales_1t;
-- LOAD DATA  INPATH '/disk/dataset/hadoop/store_sales.dat' INTO TABLE store_sales_1t;

ALTER TABLE store_sales SET LOCATION "/data/tcp/tcp-ds/1t/store_sales.dat";
-- ALTER TABLE store_sales_1t SET LOCATION 'hdfs://137.205.118.65:50075/user/hive/warehouse/store_sales/1T.dat';
ALTER TABLE store_sales_1t SET LOCATION 'hdfs://guest-wl-65.dcs.warwick.ac.uk:9000/user/hive/warehouse/store_sales/1T.dat';


CREATE TABLE store_sales_sample_1_percent1 AS SELECT * FROM store_sales SAMPLEWITH 0.01;
CREATE TABLE store_sales_sample_1_percent_cached AS SELECT * FROM store_sales_sample_1_percent;

set blinkdb.sample.size=28794695;
set blinkdb.dataset.size=2685596178;

cat 1m_1.log | grep "^INFO - SELECT" >1m_1.hiveql
./bin/blinkdb -i ~/results/1m_1.hiveql > ~/results/1m_1.log
# append ; to the end  of each line
sed -e 's/$/;/' -i 1m_1.hiveql

# fetch the results from the file
 cat 1m_1.log | grep '(99% Confidence)' > results.log




CREATE TABLE store_sales_1_percent  AS SELECT * FROM store_sales SAMPLEWITH 0.01;
CREATE TABLE store_sales_1m AS SELECT * FROM store_sales_1_percent SAMPLEWITH 0.034728619;
#################################################################################
#create a new sample in blinkdb, the size is 1 million.
CREATE TABLE store_sales_sample_1m_cached AS SELECT * FROM store_sales_sample_1_percent SAMPLEWITH 0.034728619;
set blinkdb.sample.size=999173;
set blinkdb.dataset.size=2685596178;





awk 'FNR % 7 == 4' max >> max_qreg


# export the table to csv file.
INSERT OVERWRITE LOCAL DIRECTORY '/home/u1796377/1_percent.csv'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
select * from price_cost_1t;

INSERT OVERWRITE LOCAL DIRECTORY '/home/u1796377/1_percent.csv' ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' SELECT * FROM pc_1_percent;


2685596178
999173
99917
9991
968;


CREATE EXTERNAL TABLE time_dim (
    t_time_sk                 INT,
    t_time_id                 char(16),
    t_time                    INT,
    t_hour                    INT,
    t_minute                  INT,
    t_second                  INT,
    t_am_pm                   char(2),
    t_shift                   char(20),
    t_sub_shift               char(20),
    t_meal_time               char(20)
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '|'
LOCATION '/user/hive/warehouse/tpcDs/time_dim';




create external table web_page
(
    wp_web_page_sk            INT               ,
    wp_web_page_id            char(16)              ,
    wp_rec_start_date         date                          ,
    wp_rec_end_date           date                          ,
    wp_creation_date_sk       INT                       ,
    wp_access_date_sk         INT                       ,
    wp_autogen_flag           char(1)                       ,
    wp_customer_sk            INT                       ,
    wp_url                    varchar(100)                  ,
    wp_type                   char(50)                      ,
    wp_char_count             INT                       ,
    wp_link_count             INT                       ,
    wp_image_count            INT                       ,
    wp_max_ad_count           INT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '|'
LOCATION '/user/hive/warehouse/tpcDs/web_page';

create external table web_sales
(
    ws_sold_date_sk           INT                       ,
    ws_sold_time_sk           INT                       ,
    ws_ship_date_sk           INT                       ,
    ws_item_sk                INT               ,
    ws_bill_customer_sk       INT                       ,
    ws_bill_cdemo_sk          INT                       ,
    ws_bill_hdemo_sk          INT                       ,
    ws_bill_addr_sk           INT                       ,
    ws_ship_customer_sk       INT                       ,
    ws_ship_cdemo_sk          INT                       ,
    ws_ship_hdemo_sk          INT                       ,
    ws_ship_addr_sk           INT                       ,
    ws_web_page_sk            INT                       ,
    ws_web_site_sk            INT                       ,
    ws_ship_mode_sk           INT                       ,
    ws_warehouse_sk           INT                       ,
    ws_promo_sk               INT                       ,
    ws_order_number           INT               ,
    ws_quantity               INT                       ,
    ws_wholesale_cost         decimal(7,2)                  ,
    ws_list_price             decimal(7,2)                  ,
    ws_sales_price            decimal(7,2)                  ,
    ws_ext_discount_amt       decimal(7,2)                  ,
    ws_ext_sales_price        decimal(7,2)                  ,
    ws_ext_wholesale_cost     decimal(7,2)                  ,
    ws_ext_list_price         decimal(7,2)                  ,
    ws_ext_tax                decimal(7,2)                  ,
    ws_coupon_amt             decimal(7,2)                  ,
    ws_ext_ship_cost          decimal(7,2)                  ,
    ws_net_paid               decimal(7,2)                  ,
    ws_net_paid_inc_tax       decimal(7,2)                  ,
    ws_net_paid_inc_ship      decimal(7,2)                  ,
    ws_net_paid_inc_ship_tax  decimal(7,2)                  ,
    ws_net_profit             decimal(7,2)                  
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '|'
LOCATION '/user/hive/warehouse/tpcDs/web_sales';


#remove comma , at end of each line
sed 's/,$//' web_page.csv > web_page.csv.nocomma



create table store_returns
(
    sr_returned_date_sk       int                       ,
    sr_return_time_sk         int                       ,
    sr_item_sk                int               ,
    sr_customer_sk            int                       ,
    sr_cdemo_sk               int                       ,
    sr_hdemo_sk               int                       ,
    sr_addr_sk                int                       ,
    sr_store_sk               int                       ,
    sr_reason_sk              int                       ,
    sr_ticket_number          int               ,
    sr_return_quantity        int                       ,
    sr_return_amt             decimal(7,2)                  ,
    sr_return_tax             decimal(7,2)                  ,
    sr_return_amt_inc_tax     decimal(7,2)                  ,
    sr_fee                    decimal(7,2)                  ,
    sr_return_ship_cost       decimal(7,2)                  ,
    sr_refunded_cash          decimal(7,2)                  ,
    sr_reversed_charge        decimal(7,2)                  ,
    sr_store_credit           decimal(7,2)                  ,
    sr_net_loss               decimal(7,2)
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '|'
LOCATION '/user/hive/warehouse/tpcDs/store_returns';



hive -e 'select ss_store_sk, count(*) from store_sales group by ss_store_sk' > ~/hahah