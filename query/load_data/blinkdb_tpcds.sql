
CREATE TABLE time_dim (
    t_time_sk                 INT,
    t_time_id                 STRING,
    t_time                    INT,
    t_hour                    INT,
    t_minute                  INT,
    t_second                  INT,
    t_am_pm                   STRING,
    t_shift                   STRING,
    t_sub_shift               STRING,
    t_meal_time               STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '|'
LOCATION '/user/hive/warehouse/tpcds/time_dim';




create  table web_page
(
    wp_web_page_sk            INT               ,
    wp_web_page_id            STRING              ,
    wp_rec_start_date         date                          ,
    wp_rec_end_date           date                          ,
    wp_creation_date_sk       INT                       ,
    wp_access_date_sk         INT                       ,
    wp_autogen_flag           STRING                       ,
    wp_customer_sk            INT                       ,
    wp_url                    STRING                  ,
    wp_type                   STRING                      ,
    wp_char_count             INT                       ,
    wp_link_count             INT                       ,
    wp_image_count            INT                       ,
    wp_max_ad_count           INT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '|'
LOCATION '/user/hive/warehouse/tpcds/web_page';




create  table web_sales
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
    ws_wholesale_cost         DOUBLE                  ,
    ws_list_price             DOUBLE                  ,
    ws_sales_price            DOUBLE                  ,
    ws_ext_discount_amt       DOUBLE                  ,
    ws_ext_sales_price        DOUBLE                  ,
    ws_ext_wholesale_cost     DOUBLE                  ,
    ws_ext_list_price         DOUBLE                  ,
    ws_ext_tax                DOUBLE                  ,
    ws_coupon_amt             DOUBLE                  ,
    ws_ext_ship_cost          DOUBLE                  ,
    ws_net_paid               DOUBLE                  ,
    ws_net_paid_inc_tax       DOUBLE                  ,
    ws_net_paid_inc_ship      DOUBLE                  ,
    ws_net_paid_inc_ship_tax  DOUBLE                  ,
    ws_net_profit             DOUBLE                  
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '|'
LOCATION '/user/hive/warehouse/tpcds/web_sales';


set store_sales_full_size=2685596178;

# create 10k sample
CREATE TABLE store_sales_10k_cached AS SELECT * FROM store_sales SAMPLEWITH 0.0000038;
set blinkdb.sample.size=;
set blinkdb.dataset.size=2685596178;

CREATE TABLE web_sales_10k_cached AS SELECT * FROM web_sales SAMPLEWITH 0.0000139 ;
set blinkdb.sample.size=;
set blinkdb.dataset.size=720000376;

CREATE TABLE time_dim_10k_cached AS SELECT * FROM time_dim SAMPLEWITH 0.1157407407;
set blinkdb.sample.size=;
set blinkdb.dataset.size=86400;

CREATE TABLE web_page_10k_cached AS SELECT * FROM web_page SAMPLEWITH 0.3;
set blinkdb.sample.size=;
set blinkdb.dataset.size=3000;



# create 100k sample
CREATE TABLE store_sales_100k_cached AS SELECT * FROM store_sales SAMPLEWITH 0.000038;
set blinkdb.sample.size=;
set blinkdb.dataset.size=2685596178;

CREATE TABLE web_sales_100k_cached AS SELECT * FROM web_sales SAMPLEWITH 0.000139 ;
set blinkdb.sample.size=;
set blinkdb.dataset.size=720000376;

CREATE TABLE time_dim_100k_cached AS SELECT * FROM time_dim SAMPLEWITH 0.60;
set blinkdb.sample.size=;
set blinkdb.dataset.size=86400;

CREATE TABLE web_page_100k_cached AS SELECT * FROM web_page SAMPLEWITH 0.60;
set blinkdb.sample.size=;
set blinkdb.dataset.size=3000;












set store_sales_full_size=2685596178;
set web_sales_full_size=720000376;
set time_dim_full_size=86400;
set web_page_full_size=3000;
set store_sales_10k_size=;
set web_sales_10k_size=;
set time_dim_10k_size=;
set web_page_10k_size=;
set store_sales_100k_size=;
set web_sales_100k_size=;
set time_dim_100k_size=;
set web_page_100k_size=;


# create 10k sample
CREATE TABLE store_sales_10k_cached AS SELECT * FROM store_sales SAMPLEWITH 0.0000038;
set blinkdb.sample.size=${hiveconf:store_sales_10k_size};
set blinkdb.dataset.size=${hiveconf:store_sales_full_size};

CREATE TABLE web_sales_10k_cached AS SELECT * FROM web_sales SAMPLEWITH 0.0000139 ;
set blinkdb.sample.size=${hiveconf:web_sales_10k_size};
set blinkdb.dataset.size=${hiveconf:web_sales_full_size};

CREATE TABLE time_dim_10k_cached AS SELECT * FROM time_dim SAMPLEWITH 0.1157407407;
set blinkdb.sample.size=${hiveconf:time_dim_10k_size};
set blinkdb.dataset.size=${hiveconf:time_dim_full_size};

CREATE TABLE web_page_10k_cached AS SELECT * FROM web_page SAMPLEWITH 0.3;
set blinkdb.sample.size=${hiveconf:web_page_10k_size};
set blinkdb.dataset.size=${hiveconf:web_page_full_size};



# create 100k sample
CREATE TABLE store_sales_100k_cached AS SELECT * FROM store_sales SAMPLEWITH 0.000038;
set blinkdb.sample.size=${hiveconf:store_sales_100k_size};
set blinkdb.dataset.size=${hiveconf:store_sales_full_size};

CREATE TABLE web_sales_100k_cached AS SELECT * FROM web_sales SAMPLEWITH 0.000139 ;
set blinkdb.sample.size=${hiveconf:web_sales_100k_size};
set blinkdb.dataset.size=${hiveconf:web_sales_full_size};

CREATE TABLE time_dim_100k_cached AS SELECT * FROM time_dim SAMPLEWITH 0.60;
set blinkdb.sample.size=${hiveconf:time_dim_100k_size};
set blinkdb.dataset.size=${hiveconf:time_dim_full_size};

CREATE TABLE web_page_100k_cached AS SELECT * FROM web_page SAMPLEWITH 0.60;
set blinkdb.sample.size=${hiveconf:web_page_100k_size};
set blinkdb.dataset.size=${hiveconf:web_page_full_size};



CREATE TABLE store_sales_10k_cached AS SELECT * FROM store_sales SAMPLEWITH 0.0000038;
CREATE TABLE web_sales_10k_cached AS SELECT * FROM web_sales SAMPLEWITH 0.0000139 ;
CREATE TABLE time_dim_10k_cached AS SELECT * FROM time_dim SAMPLEWITH 0.1157407407;
CREATE TABLE web_page_10k_cached AS SELECT * FROM web_page SAMPLEWITH 0.3;

select count(*) from store_sales_10k_cached;
select count(*) from web_sales_10k_cached;
select count(*) from time_dim_10k_cached;
select count(*) from web_page_10k_cached;

CREATE TABLE store_sales_100k_cached AS SELECT * FROM store_sales SAMPLEWITH 0.000038;
CREATE TABLE web_sales_100k_cached AS SELECT * FROM web_sales SAMPLEWITH 0.000139;
CREATE TABLE time_dim_100k_cached AS SELECT * FROM time_dim SAMPLEWITH 0.60;
CREATE TABLE web_page_100k_cached AS SELECT * FROM web_page SAMPLEWITH 0.60;

select count(*) from store_sales_100k_cached;
select count(*) from web_sales_100k_cached;
select count(*) from time_dim_100k_cached;
select count(*) from web_page_100k_cached;