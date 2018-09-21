#9
select count(*) from store_sales where ss_quantity between 1 and 20;
select count(*) from store_sales where ss_quantity between 21 and 40;
select count(*) from store_sales where ss_quantity between 41 and 60;
select count(*) from store_sales where ss_quantity between 61 and 80;
select count(*) from store_sales where ss_quantity between 81 and 100;

select avg(ss_ext_discount_amt) from store_sales where ss_quantity between 1 and 20;
select avg(ss_ext_discount_amt) from store_sales where ss_quantity between 21 and 40;
select avg(ss_ext_discount_amt) from store_sales where ss_quantity between 41 and 60;
select avg(ss_ext_discount_amt) from store_sales where ss_quantity between 61 and 80;
select avg(ss_ext_discount_amt) from store_sales where ss_quantity between 81 and 100;

select avg(ss_ext_sales_price) from store_sales where ss_quantity between 1 and 20;
select avg(ss_ext_sales_price) from store_sales where ss_quantity between 21 and 40;
select avg(ss_ext_sales_price) from store_sales where ss_quantity between 41 and 60;
select avg(ss_ext_sales_price) from store_sales where ss_quantity between 61 and 80;
select avg(ss_ext_sales_price) from store_sales where ss_quantity between 81 and 100;

select avg(ss_ext_list_price) from store_sales where ss_quantity between 1 and 20;
select avg(ss_ext_list_price) from store_sales where ss_quantity between 21 and 40;
select avg(ss_ext_list_price) from store_sales where ss_quantity between 41 and 60;
select avg(ss_ext_list_price) from store_sales where ss_quantity between 61 and 80;
select avg(ss_ext_list_price) from store_sales where ss_quantity between 81 and 100;

select avg(ss_ext_tax) from store_sales where ss_quantity between 1 and 20;
select avg(ss_ext_tax) from store_sales where ss_quantity between 21 and 40;
select avg(ss_ext_tax) from store_sales where ss_quantity between 41 and 60;
select avg(ss_ext_tax) from store_sales where ss_quantity between 61 and 80;
select avg(ss_ext_tax) from store_sales where ss_quantity between 81 and 100;

select avg(ss_net_paid) from store_sales where ss_quantity between 1 and 20;
select avg(ss_net_paid) from store_sales where ss_quantity between 21 and 40;
select avg(ss_net_paid) from store_sales where ss_quantity between 41 and 60;
select avg(ss_net_paid) from store_sales where ss_quantity between 61 and 80;
select avg(ss_net_paid) from store_sales where ss_quantity between 81 and 100;

select avg(ss_net_paid_inc_tax) from store_sales where ss_quantity between 1 and 20;
select avg(ss_net_paid_inc_tax) from store_sales where ss_quantity between 21 and 40;
select avg(ss_net_paid_inc_tax) from store_sales where ss_quantity between 41 and 60;
select avg(ss_net_paid_inc_tax) from store_sales where ss_quantity between 61 and 80;
select avg(ss_net_paid_inc_tax) from store_sales where ss_quantity between 81 and 100;

select avg(ss_net_profit) from store_sales where ss_quantity between 1 and 20;
select avg(ss_net_profit) from store_sales where ss_quantity between 21 and 40;
select avg(ss_net_profit) from store_sales where ss_quantity between 41 and 60;
select avg(ss_net_profit) from store_sales where ss_quantity between 61 and 80;
select avg(ss_net_profit) from store_sales where ss_quantity between 81 and 100;

#28
select avg(ss_list_price) from store_sales where ss_quantity between 0 and 5;
select avg(ss_list_price) from store_sales where ss_quantity between 6 and 10;
select avg(ss_list_price) from store_sales where ss_quantity between 11 and 15;
select avg(ss_list_price) from store_sales where ss_quantity between 16 and 20;
select avg(ss_list_price) from store_sales where ss_quantity between 21 and 25;
select avg(ss_list_price) from store_sales where ss_quantity between 26 and 30;
select count(ss_list_price) from store_sales where ss_quantity between 0 and 5;
select count(ss_list_price) from store_sales where ss_quantity between 6 and 10;
select count(ss_list_price) from store_sales where ss_quantity between 11 and 15;
select count(ss_list_price) from store_sales where ss_quantity between 16 and 20;
select count(ss_list_price) from store_sales where ss_quantity between 21 and 25;
select count(ss_list_price) from store_sales where ss_quantity between 26 and 30;

#define LISTPRICE=ulist(random(0, 190, uniform),6);
select avg(ss_list_price) from store_sales where ss_list_price between 90 and 100; 
select avg(ss_list_price) from store_sales where ss_list_price between 70 and 80; 
select avg(ss_list_price) from store_sales where ss_list_price between 80 and 90;
select avg(ss_list_price) from store_sales where ss_list_price between 100 and 110;
select avg(ss_list_price) from store_sales where ss_list_price between 110 and 120;
select avg(ss_list_price) from store_sales where ss_list_price between 120 and 130;
select count(ss_list_price) from store_sales where ss_list_price between 90 and 100; 
select count(ss_list_price) from store_sales where ss_list_price between 70 and 80; 
select count(ss_list_price) from store_sales where ss_list_price between 80 and 90;
select count(ss_list_price) from store_sales where ss_list_price between 100 and 110;
select count(ss_list_price) from store_sales where ss_list_price between 110 and 120;
select count(ss_list_price) from store_sales where ss_list_price between 120 and 130;

#define COUPONAMT=ulist(random(0, 18000, uniform),6); 9000
select avg(ss_list_price)   from store_sales where ss_coupon_amt between 7000    and 8000 ; 
select avg(ss_list_price)   from store_sales where ss_coupon_amt between 8000    and 9000 ; 
select avg(ss_list_price)   from store_sales where ss_coupon_amt between 9000    and 10000;
select avg(ss_list_price)   from store_sales where ss_coupon_amt between 10000   and 11000;
select avg(ss_list_price)   from store_sales where ss_coupon_amt between 11000   and 12000;
select avg(ss_list_price)   from store_sales where ss_coupon_amt between 12000   and 13000;
select count(ss_list_price) from store_sales where ss_coupon_amt between 7000    and 8000 ;
select count(ss_list_price) from store_sales where ss_coupon_amt between 8000    and 9000 ;
select count(ss_list_price) from store_sales where ss_coupon_amt between 9000    and 10000;
select count(ss_list_price) from store_sales where ss_coupon_amt between 10000   and 11000;
select count(ss_list_price) from store_sales where ss_coupon_amt between 11000   and 12000;
select count(ss_list_price) from store_sales where ss_coupon_amt between 12000   and 13000;

#define WHOLESALECOST=ulist(random(0, 80, uniform),6);40
select avg(ss_list_price)   from store_sales where ss_wholesale_cost between 10     and 30; 
select avg(ss_list_price)   from store_sales where ss_wholesale_cost between 20     and 40; 
select avg(ss_list_price)   from store_sales where ss_wholesale_cost between 30     and 50;
select avg(ss_list_price)   from store_sales where ss_wholesale_cost between 40     and 60;
select avg(ss_list_price)   from store_sales where ss_wholesale_cost between 50     and 70;
select avg(ss_list_price)   from store_sales where ss_wholesale_cost between 60     and 80;
select count(ss_list_price) from store_sales where ss_wholesale_cost between 10     and 30;
select count(ss_list_price) from store_sales where ss_wholesale_cost between 20     and 40;
select count(ss_list_price) from store_sales where ss_wholesale_cost between 30     and 50;
select count(ss_list_price) from store_sales where ss_wholesale_cost between 40     and 60;
select count(ss_list_price) from store_sales where ss_wholesale_cost between 50     and 70;
select count(ss_list_price) from store_sales where ss_wholesale_cost between 60     and 80;


#48 contains other conditions as well
select sum (ss_quantity) from store_sales where ss_sales_price between 50.00  and 100.00;
select sum (ss_quantity) from store_sales where ss_sales_price between 100.00 and 150.00;
select sum (ss_quantity) from store_sales where ss_sales_price between 150.00 and 200.00;

select sum (ss_quantity) from store_sales where ss_net_profit between 0 and 2000;
select sum (ss_quantity) from store_sales where ss_net_profit between 150 and 3000;
select sum (ss_quantity) from store_sales where ss_net_profit between 50 and 25000;

#79 need join
-- select sum(ss_coupon_amt) from store_sales, store where store.s_number_employees between 200 and 295;
-- select sum(ss_net_profit) from store_sales, store where store.s_number_employees between 200 and 295;

#85 need join
select avg(ws_quantity) from web_sales where ws_sales_price between 100.00 and 150.00;
select avg(ws_quantity) from web_sales where ws_sales_price between  50.00 and 100.00;
select avg(ws_quantity) from web_sales where ws_sales_price between 150.00 and 200.00;

-- select avg(wr_refunded_cash) from web_sales, web_returns where ws_sales_price between 100.00 and 150.00; 
-- select avg(wr_refunded_cash) from web_sales, web_returns where ws_sales_price between  50.00 and 100.00;
-- select avg(wr_refunded_cash) from web_sales, web_returns where ws_sales_price between 150.00 and 200.00;

-- select avg(wr_fee) from web_sales, web_returns where ws_sales_price between 100.00 and 150.00; 
-- select avg(wr_fee) from web_sales, web_returns where ws_sales_price between  50.00 and 100.00;
-- select avg(wr_fee) from web_sales, web_returns where ws_sales_price between 150.00 and 200.00;