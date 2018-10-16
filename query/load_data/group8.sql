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






CREATE  TABLE store_sales_group_d (
ss_sold_date_sk           DOUBLE,
ss_store_sk               DOUBLE,
ss_sales_price            DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
LOCATION '/user/hive/warehouse/store_sales_group';


CREATE TABLE store_sales_group_1m_cached AS SELECT * FROM store_sales_group_d SAMPLEWITH 0.000347224;

CREATE TABLE store_sales_group_100k_cached AS SELECT * FROM store_sales_group_d SAMPLEWITH 0.0000347224;


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




hive -e 'select ss_store_sk, count(*) from store_sales_group_d group by ss_store_sk' > ~/group_count8.csv


db.query_simple_groupby('select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451119 and 2451483 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/sum1.txt")
db.query_simple_groupby('select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451300 and 2451665 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/sum2.txt")
db.query_simple_groupby('select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451392 and 2451757 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/sum3.txt")
db.query_simple_groupby('select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451484 and 2451849 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/sum4.txt")
db.query_simple_groupby('select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451545 and 2451910 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/sum5.txt")
db.query_simple_groupby('select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451636 and 2452000 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/sum6.txt")
db.query_simple_groupby('select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451727 and 2452091 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/sum7.txt")
db.query_simple_groupby('select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451850 and 2452214 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/sum8.txt")
db.query_simple_groupby('select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451911 and 2452275 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/sum9.txt")
db.query_simple_groupby('select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2452031 and 2452395 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/sum10.txt")
db.query_simple_groupby('select count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451119 and 2451483 group by ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/count1.txt")
db.query_simple_groupby('select count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451300 and 2451665 group by ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/count2.txt")
db.query_simple_groupby('select count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451392 and 2451757 group by ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/count3.txt")
db.query_simple_groupby('select count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451484 and 2451849 group by ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/count4.txt")
db.query_simple_groupby('select count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451545 and 2451910 group by ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/count5.txt")
db.query_simple_groupby('select count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451636 and 2452000 group by ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/count6.txt")
db.query_simple_groupby('select count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451727 and 2452091 group by ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/count7.txt")
db.query_simple_groupby('select count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451850 and 2452214 group by ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/count8.txt")
db.query_simple_groupby('select count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451911 and 2452275 group by ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/count9.txt")
db.query_simple_groupby('select count(ss_sales_price) from store_sales where ss_sold_date_sk between 2452031 and 2452395 group by ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/count10.txt")
db.query_simple_groupby('select avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451119 and 2451483 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/avg1.txt")
db.query_simple_groupby('select avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451300 and 2451665 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/avg2.txt")
db.query_simple_groupby('select avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451392 and 2451757 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/avg3.txt")
db.query_simple_groupby('select avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451484 and 2451849 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/avg4.txt")
db.query_simple_groupby('select avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451545 and 2451910 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/avg5.txt")
db.query_simple_groupby('select avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451636 and 2452000 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/avg6.txt")
db.query_simple_groupby('select avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451727 and 2452091 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/avg7.txt")
db.query_simple_groupby('select avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451850 and 2452214 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/avg8.txt")
db.query_simple_groupby('select avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451911 and 2452275 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/avg9.txt")
db.query_simple_groupby('select avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2452031 and 2452395 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral/avg10.txt")






db1.query_simple_groupby('select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451119 and 2451483 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/sum1.txt")
db1.query_simple_groupby('select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451300 and 2451665 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/sum2.txt")
db1.query_simple_groupby('select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451392 and 2451757 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/sum3.txt")
db1.query_simple_groupby('select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451484 and 2451849 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/sum4.txt")
db1.query_simple_groupby('select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451545 and 2451910 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/sum5.txt")
db1.query_simple_groupby('select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451636 and 2452000 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/sum6.txt")
db1.query_simple_groupby('select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451727 and 2452091 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/sum7.txt")
db1.query_simple_groupby('select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451850 and 2452214 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/sum8.txt")
db1.query_simple_groupby('select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451911 and 2452275 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/sum9.txt")
db1.query_simple_groupby('select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2452031 and 2452395 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/sum10.txt")
db1.query_simple_groupby('select count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451119 and 2451483 group by ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/count1.txt")
db1.query_simple_groupby('select count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451300 and 2451665 group by ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/count2.txt")
db1.query_simple_groupby('select count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451392 and 2451757 group by ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/count3.txt")
db1.query_simple_groupby('select count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451484 and 2451849 group by ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/count4.txt")
db1.query_simple_groupby('select count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451545 and 2451910 group by ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/count5.txt")
db1.query_simple_groupby('select count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451636 and 2452000 group by ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/count6.txt")
db1.query_simple_groupby('select count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451727 and 2452091 group by ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/count7.txt")
db1.query_simple_groupby('select count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451850 and 2452214 group by ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/count8.txt")
db1.query_simple_groupby('select count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451911 and 2452275 group by ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/count9.txt")
db1.query_simple_groupby('select count(ss_sales_price) from store_sales where ss_sold_date_sk between 2452031 and 2452395 group by ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/count10.txt")
db1.query_simple_groupby('select avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451119 and 2451483 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/avg1.txt")
db1.query_simple_groupby('select avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451300 and 2451665 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/avg2.txt")
db1.query_simple_groupby('select avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451392 and 2451757 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/avg3.txt")
db1.query_simple_groupby('select avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451484 and 2451849 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/avg4.txt")
db1.query_simple_groupby('select avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451545 and 2451910 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/avg5.txt")
db1.query_simple_groupby('select avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451636 and 2452000 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/avg6.txt")
db1.query_simple_groupby('select avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451727 and 2452091 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/avg7.txt")
db1.query_simple_groupby('select avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451850 and 2452214 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/avg8.txt")
db1.query_simple_groupby('select avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451911 and 2452275 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/avg9.txt")
db1.query_simple_groupby('select avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2452031 and 2452395 group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds_groupby_few_groups/DBEst_integral_1m/avg10.txt")