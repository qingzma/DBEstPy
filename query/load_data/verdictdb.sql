##CCPP





##TPC-H
-- nation
spark.sql("CREATE TABLE IF NOT EXISTS TPCH.nation ( n_nationkey  INT,  n_name       CHAR(25),  n_regionkey  INT,  n_comment  VARCHAR(152),  n_dummy      VARCHAR(10))  ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'  STORED AS TEXTFILE  LOCATION 'hdfs://137.205.118.65:9000/user/hive/warehouse/verdictdb/tpch1g/nation/'")
spark.sql("CREATE TABLE IF NOT EXISTS TPCH.region (  r_regionkey  INT,  r_name       CHAR(25),  r_comment    VARCHAR(152),  r_dummy      VARCHAR(10))  ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'  STORED AS TEXTFILE  LOCATION 'hdfs://137.205.118.65:9000/user/hive/warehouse/verdictdb/tpch1g/region/'")


-- region
CREATE TABLE IF NOT EXISTS region (
  r_regionkey  INT,
  r_name       CHAR(25),
  r_comment    VARCHAR(152),
  r_dummy      VARCHAR(10))
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
  STORED AS TEXTFILE
  LOCATION '/user/hive/warehouse/verdictdb/tpch1g/region/region';


-- supplier
CREATE TABLE IF NOT EXISTS supplier (
  s_suppkey     INT,
  s_name        CHAR(25),
  s_address     VARCHAR(40),
  s_nationkey   INT,
  s_phone       CHAR(15),
  s_acctbal     DECIMAL(15,2),
  s_comment     VARCHAR(101),
  s_dummy varchar(10))
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
  STORED AS TEXTFILE
  LOCATION '/user/hive/warehouse/verdictdb/tpch1g/supplier/supplier';


-- customer
CREATE TABLE IF NOT EXISTS customer (
  c_custkey     INT,
  c_name        VARCHAR(25),
  c_address     VARCHAR(40),
  c_nationkey   INT,
  c_phone       CHAR(15),
  c_acctbal     DECIMAL(15,2),
  c_mktsegment  CHAR(10),
  c_comment     VARCHAR(117),
  c_dummy       VARCHAR(10))
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
  STORED AS TEXTFILE
  LOCATION '/user/hive/warehouse/verdictdb/tpch1g/customer/customer';


-- part
CREATE TABLE IF NOT EXISTS part (
  p_partkey     INT,
  p_name        VARCHAR(55),
  p_mfgr        CHAR(25),
  p_brand       CHAR(10),
  p_type        VARCHAR(25),
  p_size        INT,
  p_container   CHAR(10),
  p_retailprice DECIMAL(15,2) ,
  p_comment     VARCHAR(23) ,
  p_dummy       VARCHAR(10))
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
  STORED AS TEXTFILE
  LOCATION '/user/hive/warehouse/verdictdb/tpch1g/part/part';

-- partsupp
CREATE TABLE IF NOT EXISTS partsupp (
  ps_partkey     INT,
  ps_suppkey     INT,
  ps_availqty    INT,
  ps_supplycost  DECIMAL(15,2),
  ps_comment     VARCHAR(199),
  ps_dummy       VARCHAR(10))
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
  STORED AS TEXTFILE
  LOCATION '/user/hive/warehouse/verdictdb/tpch1g/partsupp/partsupp';

-- orders
CREATE TABLE IF NOT EXISTS orders (
  o_orderkey       INT,
  o_custkey        INT,
  o_orderstatus    CHAR(1),
  o_totalprice     DECIMAL(15,2),
  o_orderdate      DATE,
  o_orderpriority  CHAR(15),
  o_clerk          CHAR(15),
  o_shippriority   INT,
  o_comment        VARCHAR(79),
  o_dummy          VARCHAR(10))
   ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
   STORED AS TEXTFILE
   LOCATION '/user/hive/warehouse/verdictdb/tpch1g/orders/orders';

-- lineitem
CREATE TABLE IF NOT EXISTS lineitem (
  l_orderkey          INT,
  l_partkey           INT,
  l_suppkey           INT,
  l_linenumber        INT,
  l_quantity          DECIMAL(15,2),
  l_extendedprice     DECIMAL(15,2),
  l_discount          DECIMAL(15,2),
  l_tax               DECIMAL(15,2),
  l_returnflag        CHAR(1),
  l_linestatus        CHAR(1),
  l_shipdate          DATE,
  l_commitdate        DATE,
  l_receiptdate       DATE,
  l_shipinstruct      CHAR(25),
  l_shipmode          CHAR(10),
  l_comment           VARCHAR(44),
  l_dummy             VARCHAR(10))
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
  STORED AS TEXTFILE
  LOCATION '/user/hive/warehouse/verdictdb/tpch1g/lineitem/lineitem';





  spark.sql("select count(*) from nation").collect().foreach(println)



  #tpc-ds

CREATE external TABLE tpcds.store ( s_store_sk int,    s_store_id string, s_rec_start_date string, s_rec_end_date string,   s_closed_date_sk string, s_store_name string,     s_number_employees int,  s_floor_space int, s_hours string,    s_manager string,  s_market_id int,   s_geography_class string,s_market_desc string,    s_market_manager string, s_division_id int,     s_division_name string,s_company_id int,      s_company_name string, s_street_number string,s_street_name string,  s_street_type string,  s_suite_number string, s_city string, s_county string, s_state string, s_zip string,   s_country string, s_gmt_offset double, s_tax_percentage double ) ROW FORMAT DELIMITED FIELDS TERMINATED BY '|' LOCATION 'hdfs://137.205.118.65:9000/user/hive/warehouse/tpcds40g/store/'"



s_store_sk int,    s_store_id string, s_rec_start_date string, s_rec_end_date string,   s_closed_date_sk string, s_store_name string,     s_number_employees int,  s_floor_space int, s_hours string,    s_manager string,  s_market_id int,   s_geography_class string,s_market_desc string,    s_market_manager string, s_division_id int,     s_division_name string,s_company_id int,      s_company_name string, s_street_number string,s_street_name string,  s_street_type string,  s_suite_number string, s_city string, s_county string, s_state string, s_zip string,   s_country string, s_gmt_offset double, s_tax_percentage double 
