CREATE  TABLE  zipf_10k( 
x            DOUBLE,
y           DOUBLE
);

CREATE  TABLE  zipf_100k( 
x            DOUBLE,
y           DOUBLE
);


CREATE  TABLE  zipf_1m( 
x            DOUBLE,
y           DOUBLE
);





# load the file;
LOAD DATA INFILE "/var/lib/mysql-files/zipf_10k.csv"
INTO TABLE zipf_10k
COLUMNS TERMINATED BY ','
LINES TERMINATED BY '\n';

LOAD DATA INFILE "/var/lib/mysql-files/zipf_100k.csv"
INTO TABLE zipf_100k
COLUMNS TERMINATED BY ','
LINES TERMINATED BY '\n';

LOAD DATA INFILE "/var/lib/mysql-files/zipf_1m.csv"
INTO TABLE zipf_1m
COLUMNS TERMINATED BY ','
LINES TERMINATED BY '\n';

#query to mysql
mysql -u root -p dbest < query.sql  > 10k.txt