#select sum(ss_sales_price) from store_sales where ss_list_price between 20 and 30  group by ss_store_sk
select sum(ss_list_price) from store_sales where ss_wholesale_cost between 20 and 30  group by ss_store_sk

insert overwrite  directory '/user/hive/counts' row format delimited fields terminated by '\t' stored as textfile 
select ss_store_sk, count(ss_list_price) from store_sales group by ss_store_sk


#to train, get the number of points 
hive -e "select ss_store_sk, count(ss_sales_price) from store_sales group by ss_store_sk" > haha


hive -e "select ss_store_sk, sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451119 and 2451483 group by ss_store_sk" > ~/results/groundtruth/1.result
hive -e "select ss_store_sk, sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451300 and 2451665 group by ss_store_sk" > ~/results/groundtruth/2.result
hive -e "select ss_store_sk, sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451392 and 2451757 group by ss_store_sk" > ~/results/groundtruth/3.result
hive -e "select ss_store_sk, sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451484 and 2451849 group by ss_store_sk" > ~/results/groundtruth/4.result
hive -e "select ss_store_sk, sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451545 and 2451910 group by ss_store_sk" > ~/results/groundtruth/5.result
hive -e "select ss_store_sk, sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451636 and 2452000 group by ss_store_sk" > ~/results/groundtruth/6.result
hive -e "select ss_store_sk, sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451727 and 2452091 group by ss_store_sk" > ~/results/groundtruth/7.result
hive -e "select ss_store_sk, sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451850 and 2452214 group by ss_store_sk" > ~/results/groundtruth/8.result
hive -e "select ss_store_sk, sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451911 and 2452275 group by ss_store_sk" > ~/results/groundtruth/9.result
hive -e "select ss_store_sk, sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2452031 and 2452395 group by ss_store_sk" > ~/results/groundtruth/10.result

#1186
select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451119 and 2451483 group by ss_store_sk  
#1998-11-01  1999-10-31

#1192
select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451300 and 2451665 group by ss_store_sk 
#1999-05-01  2000-04-30


#1195 1207
select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451392 and 2451757 group by ss_store_sk 
#1999-08-01 2000-07-31

#1198 1210
select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451484 and 2451849 group by ss_store_sk 
#1999-11-01 2000-10-31

#1200 1212
select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451545 and 2451910 group by ss_store_sk 
#2000-01-01 2000-12-31

#1203 1215
select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451636 and 2452000 group by ss_store_sk 
#2000-04-01 2001-03-31

#1206 1218
select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451727 and 2452091 group by ss_store_sk 
#2000-07-01 2001-06-30


#1210 1222
select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451850 and 2452214 group by ss_store_sk 
#2000-11-01 2001-10-31

#1212 1224
select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451911 and 2452275 group by ss_store_sk 
#2001-01-01 2001-12-31

#1216 1228
select sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2452031 and 2452395 group by ss_store_sk 
#2001-05-01 2002-04-30






# blinkdb
CREATE TABLE store_sales_5m_cached AS SELECT * FROM store_sales SAMPLEWITH 0.001936255;

select count(*) from store_sales_5m_cached;



set blinkdb.sample.size=5577863;
set blinkdb.dataset.size=2685596178;



select ss_store_sk, approx_sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451119 and 2451483 group by ss_store_sk;
select ss_store_sk, approx_sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451300 and 2451665 group by ss_store_sk;
select ss_store_sk, approx_sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451392 and 2451757 group by ss_store_sk;
select ss_store_sk, approx_sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451484 and 2451849 group by ss_store_sk;
select ss_store_sk, approx_sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451545 and 2451910 group by ss_store_sk;
select ss_store_sk, approx_sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451636 and 2452000 group by ss_store_sk;
select ss_store_sk, approx_sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451727 and 2452091 group by ss_store_sk;
select ss_store_sk, approx_sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451850 and 2452214 group by ss_store_sk;
select ss_store_sk, approx_sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2451911 and 2452275 group by ss_store_sk;
select ss_store_sk, approx_sum(ss_sales_price) from store_sales where ss_sold_date_sk between 2452031 and 2452395 group by ss_store_sk;

select ss_store_sk, approx_count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451119 and 2451483 group by ss_store_sk;
select ss_store_sk, approx_count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451300 and 2451665 group by ss_store_sk;
select ss_store_sk, approx_count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451392 and 2451757 group by ss_store_sk;
select ss_store_sk, approx_count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451484 and 2451849 group by ss_store_sk;
select ss_store_sk, approx_count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451545 and 2451910 group by ss_store_sk;
select ss_store_sk, approx_count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451636 and 2452000 group by ss_store_sk;
select ss_store_sk, approx_count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451727 and 2452091 group by ss_store_sk;
select ss_store_sk, approx_count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451850 and 2452214 group by ss_store_sk;
select ss_store_sk, approx_count(ss_sales_price) from store_sales where ss_sold_date_sk between 2451911 and 2452275 group by ss_store_sk;
select ss_store_sk, approx_count(ss_sales_price) from store_sales where ss_sold_date_sk between 2452031 and 2452395 group by ss_store_sk;

select ss_store_sk, approx_avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451119 and 2451483 group by ss_store_sk;
select ss_store_sk, approx_avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451300 and 2451665 group by ss_store_sk;
select ss_store_sk, approx_avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451392 and 2451757 group by ss_store_sk;
select ss_store_sk, approx_avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451484 and 2451849 group by ss_store_sk;
select ss_store_sk, approx_avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451545 and 2451910 group by ss_store_sk;
select ss_store_sk, approx_avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451636 and 2452000 group by ss_store_sk;
select ss_store_sk, approx_avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451727 and 2452091 group by ss_store_sk;
select ss_store_sk, approx_avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451850 and 2452214 group by ss_store_sk;
select ss_store_sk, approx_avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2451911 and 2452275 group by ss_store_sk;
select ss_store_sk, approx_avg(ss_sales_price) from store_sales where ss_sold_date_sk between 2452031 and 2452395 group by ss_store_sk;











# 9 group
set blinkdb.sample.size=1072974;
set blinkdb.dataset.size=2685596178;



#HIVE to file