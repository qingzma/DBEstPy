CREATE TABLE table8u (x DOUBLE, y DOUBLE) ROW FORMAT DELIMITED FIELDS TERMINATED BY ",";
LOAD DATA LOCAL INPATH '/Users/hduser/workspace/CRegressionRDBM/data/file8u.csv' INTO TABLE table8u;