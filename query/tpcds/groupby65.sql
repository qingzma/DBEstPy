#select sum(ss_sales_price) from store_sales where ss_list_price between 20 and 30  group by ss_store_sk
select sum(ss_list_price) from store_sales where ss_wholesale_cost between 20 and 30  group by ss_store_sk

insert overwrite  directory '/user/hive/counts' row format delimited fields terminated by '\t' stored as textfile 
select ss_store_sk, count(ss_list_price) from store_sales group by ss_store_sk

hive -e "select ss_store_sk, count(ss_list_price) from store_sales group by ss_store_sk" > haha