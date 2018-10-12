CREATE  TABLE store_sales_group ( ss_sold_date_sk           INT,
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
FIELDS TERMINATED BY ','
LOCATION '/user/hive/warehouse/store_sales_group';


CREATE  TABLE store_sales_group_d ( ss_sold_date_sk           DOUBLE,
ss_sold_time_sk           DOUBLE,
ss_item_sk                DOUBLE,
ss_customer_sk            DOUBLE,
ss_cdemo_sk               DOUBLE,
ss_hdemo_sk               DOUBLE,
ss_addr_sk                DOUBLE,
ss_store_sk               DOUBLE,
ss_promo_sk               DOUBLE,
ss_ticket_number          DOUBLE,
ss_quantity               DOUBLE,
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
FIELDS TERMINATED BY ','
LOCATION '/user/hive/warehouse/store_sales_group';


CREATE TABLE store_sales_group_1m_cached AS SELECT * FROM store_sales_group_d SAMPLEWITH 0.000372357;


CREATE external TABLE store_sales_group ( ss_sold_date_sk           INT,
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
FIELDS TERMINATED BY ','
LOCATION '/user/hive/warehouse/store_sales_group';




hive -e 'select ss_store_sk, count(*) from store_sales_group group by ss_store_sk' > ~/group_count.csv