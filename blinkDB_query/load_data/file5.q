CREATE TABLE table5 (x DOUBLE, y DOUBLE) ROW FORMAT DELIMITED FIELDS TERMINATED BY ",";
LOAD DATA LOCAL INPATH 'data/file5.csv' INTO TABLE table5;