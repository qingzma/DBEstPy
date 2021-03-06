select avg(ss_ext_discount_amt) from store_sales where ss_quantity       between 1      and 1.5
select avg(ss_ext_discount_amt) from store_sales where ss_quantity       between 21     and 21.5
select avg(ss_ext_discount_amt) from store_sales where ss_quantity       between 41     and 41.5
select avg(ss_ext_discount_amt) from store_sales where ss_quantity       between 61     and 61.5
select avg(ss_ext_discount_amt) from store_sales where ss_quantity       between 81     and 81.5
select avg(ss_ext_sales_price)  from store_sales where ss_quantity       between 1      and 1.5
select avg(ss_ext_sales_price)  from store_sales where ss_quantity       between 21     and 21.5
select avg(ss_ext_sales_price)  from store_sales where ss_quantity       between 41     and 41.5
select avg(ss_ext_sales_price)  from store_sales where ss_quantity       between 61     and 61.5
select avg(ss_ext_sales_price)  from store_sales where ss_quantity       between 81     and 81.5
select avg(ss_ext_list_price)   from store_sales where ss_quantity       between 1      and 1.5
select avg(ss_ext_list_price)   from store_sales where ss_quantity       between 21     and 21.5
select avg(ss_ext_list_price)   from store_sales where ss_quantity       between 41     and 41.5
select avg(ss_ext_list_price)   from store_sales where ss_quantity       between 61     and 61.5
select avg(ss_ext_list_price)   from store_sales where ss_quantity       between 81     and 81.5
select avg(ss_ext_tax)          from store_sales where ss_quantity       between 1      and 1.5
select avg(ss_ext_tax)          from store_sales where ss_quantity       between 21     and 21.5
select avg(ss_ext_tax)          from store_sales where ss_quantity       between 41     and 41.5
select avg(ss_ext_tax)          from store_sales where ss_quantity       between 61     and 61.5
select avg(ss_ext_tax)          from store_sales where ss_quantity       between 81     and 81.5
select avg(ss_net_paid)         from store_sales where ss_quantity       between 1      and 1.5
select avg(ss_net_paid)         from store_sales where ss_quantity       between 21     and 21.5
select avg(ss_net_paid)         from store_sales where ss_quantity       between 41     and 41.5
select avg(ss_net_paid)         from store_sales where ss_quantity       between 61     and 61.5
select avg(ss_net_paid)         from store_sales where ss_quantity       between 81     and 81.5
select avg(ss_net_paid_inc_tax) from store_sales where ss_quantity       between 1      and 1.5
select avg(ss_net_paid_inc_tax) from store_sales where ss_quantity       between 21     and 21.5
select avg(ss_net_paid_inc_tax) from store_sales where ss_quantity       between 41     and 41.5
select avg(ss_net_paid_inc_tax) from store_sales where ss_quantity       between 61     and 61.5
select avg(ss_net_paid_inc_tax) from store_sales where ss_quantity       between 81     and 81.5
select avg(ss_net_profit)       from store_sales where ss_quantity       between 1      and 1.5
select avg(ss_net_profit)       from store_sales where ss_quantity       between 21     and 21.5
select avg(ss_net_profit)       from store_sales where ss_quantity       between 41     and 41.5
select avg(ss_net_profit)       from store_sales where ss_quantity       between 61     and 61.5
select avg(ss_net_profit)       from store_sales where ss_quantity       between 81     and 81.5
select avg(ss_list_price)       from store_sales where ss_quantity       between 0      and 0.5
select avg(ss_list_price)       from store_sales where ss_quantity       between 6      and 6.5
select avg(ss_list_price)       from store_sales where ss_quantity       between 11     and 11.5
select avg(ss_list_price)       from store_sales where ss_quantity       between 16     and 16.5
select avg(ss_list_price)       from store_sales where ss_quantity       between 21     and 21.5
select avg(ss_list_price)       from store_sales where ss_quantity       between 26     and 26.5
select avg(ss_list_price)       from store_sales where ss_list_price     between 90     and 91
select avg(ss_list_price)       from store_sales where ss_list_price     between 70     and 71 
select avg(ss_list_price)       from store_sales where ss_list_price     between 80     and 81
select avg(ss_list_price)       from store_sales where ss_list_price     between 100    and 101
select avg(ss_list_price)       from store_sales where ss_list_price     between 110    and 111
select avg(ss_list_price)       from store_sales where ss_list_price     between 120    and 121
select avg(ss_list_price)       from store_sales where ss_coupon_amt     between 7000   and 7090
select avg(ss_list_price)       from store_sales where ss_coupon_amt     between 8000   and 8090
select avg(ss_list_price)       from store_sales where ss_coupon_amt     between 9000   and 9090
select avg(ss_list_price)       from store_sales where ss_coupon_amt     between 10000  and 10090
select avg(ss_list_price)       from store_sales where ss_coupon_amt     between 11000  and 11090
select avg(ss_list_price)       from store_sales where ss_coupon_amt     between 12000  and 12090
select avg(ss_list_price)       from store_sales where ss_wholesale_cost between 10     and 10.5 
select avg(ss_list_price)       from store_sales where ss_wholesale_cost between 20     and 20.5 
select avg(ss_list_price)       from store_sales where ss_wholesale_cost between 30     and 30.5
select avg(ss_list_price)       from store_sales where ss_wholesale_cost between 40     and 40.5
select avg(ss_list_price)       from store_sales where ss_wholesale_cost between 50     and 50.5
select avg(ss_list_price)       from store_sales where ss_wholesale_cost between 60     and 60.5
select avg(ws_quantity)         from web_sales   where ws_sales_price    between 100.00 and 101.5
select avg(ws_quantity)         from web_sales   where ws_sales_price    between  50.00 and 51.5
select avg(ws_quantity)         from web_sales   where ws_sales_price    between 150.00 and 151.5
select count(*) 			    from store_sales where ss_quantity 	     between 1    and 1.5
select count(*) 			    from store_sales where ss_quantity 	     between 21   and 21.5
select count(*) 			    from store_sales where ss_quantity 	     between 41   and 41.5
select count(*) 			    from store_sales where ss_quantity 	     between 61   and 61.5
select count(*) 			    from store_sales where ss_quantity 	     between 81   and 81.5
select count(ss_list_price)     from store_sales where ss_quantity 	     between 0    and 0.5
select count(ss_list_price)     from store_sales where ss_quantity 	     between 6    and 6.5
select count(ss_list_price)     from store_sales where ss_quantity 	     between 11   and 11.5
select count(ss_list_price)     from store_sales where ss_quantity 	     between 16   and 16.5
select count(ss_list_price)     from store_sales where ss_quantity 	     between 21   and 21.5
select count(ss_list_price)     from store_sales where ss_quantity 	     between 26   and 26.5
select count(ss_list_price)     from store_sales where ss_list_price     between 90   and 91
select count(ss_list_price)     from store_sales where ss_list_price     between 70   and 71 
select count(ss_list_price)     from store_sales where ss_list_price     between 80   and 81
select count(ss_list_price)     from store_sales where ss_list_price     between 100  and 101
select count(ss_list_price)     from store_sales where ss_list_price     between 110  and 111
select count(ss_list_price)     from store_sales where ss_list_price     between 120  and 121
select count(ss_list_price)     from store_sales where ss_coupon_amt     between 7000   and 7090
select count(ss_list_price)     from store_sales where ss_coupon_amt     between 8000   and 8090
select count(ss_list_price)     from store_sales where ss_coupon_amt     between 9000   and 9090
select count(ss_list_price)     from store_sales where ss_coupon_amt     between 10000  and 10090
select count(ss_list_price)     from store_sales where ss_coupon_amt     between 11000  and 11090
select count(ss_list_price)     from store_sales where ss_coupon_amt     between 12000  and 12090
select count(ss_list_price)     from store_sales where ss_wholesale_cost between 10     and 10.5 
select count(ss_list_price)     from store_sales where ss_wholesale_cost between 20     and 20.5 
select count(ss_list_price)     from store_sales where ss_wholesale_cost between 30     and 30.5
select count(ss_list_price)     from store_sales where ss_wholesale_cost between 40     and 40.5
select count(ss_list_price)     from store_sales where ss_wholesale_cost between 50     and 50.5
select count(ss_list_price)     from store_sales where ss_wholesale_cost between 60     and 60.5
select sum (ss_quantity)        from store_sales where ss_sales_price    between 50.00  and 51
select sum (ss_quantity)        from store_sales where ss_sales_price    between 100.00 and 101
select sum (ss_quantity)        from store_sales where ss_sales_price    between 150.00 and 151
select sum (ss_quantity)        from store_sales where ss_net_profit     between 0      and 100
select sum (ss_quantity)        from store_sales where ss_net_profit     between 150    and 250
select sum (ss_quantity)        from store_sales where ss_net_profit     between 50     and 150