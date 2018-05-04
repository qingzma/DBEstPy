#!/usr/bin/env python
# coding=utf-8
from core import CRegression
import data_loader as dl
from query_engine import QueryEngine
import logs
import random
import commands
# from pyhive import hive
# import subprocess
import os
import pyhs2
import MySQLdb

from datetime import datetime

default_mass_query_number = 5
logger_file = "../results/deletable.log"

class Query_Engine_2d:
    def __init__(self,dataID,b_allow_repeated_value=True,logger_file=logger_file):
        self.logger = logs.QueryLogs(log=logger_file)
        self.logger.set_no_output()
        self.data = dl.load2d(dataID)
        if not b_allow_repeated_value:
            self.data.remove_repeated_x_1d()
        self.cRegression = CRegression(logger_object=self.logger)
        self.cRegression.fit(self.data)
        self.logger.set_logging(file_name=logger_file)
        self.qe = QueryEngine(self.cRegression, logger_object=self.logger)
        self.qe.density_estimation()
        self.q_min = min(self.data.features)
        self.q_max = max(self.data.features)
        self.dataID = dataID

    def query_2d_avg(self,l=0,h=100):
        """query to 2d data sets.

        Args:
            l (int, optional): query lower boundary
            h (int, optional): query higher boundary
        """
        avgs,time = self.qe.approximate_avg_from_to(l ,h, 0)  #0.05E8,0.1E8,
        return avgs, time

    def query_2d_sum(self,l=0,h=100):
        """query to 2d data sets.

        Args:
            l (int, optional): query lower boundary
            h (int, optional): query higher boundary
        """
        sums,time = self.qe.approximate_sum_from_to(l ,h, 0)
        return sums, time
    def query_2d_count(self,l=0,h=100):
        count,time = self.qe.approximate_count_from_to(l ,h, 0)
        return count, time
    def mass_query_sum(self,percent=5,number=default_mass_query_number):
        q_range_half_length = (self.q_max-self.q_min)*percent/100.0/2.0
        random.seed(1.0)
        exact_results = []
        exact_times = []
        approx_results = []
        approx_times = []
        for i in range(number):
            q_centre = random.uniform(self.q_min,self.q_max)
            q_left = q_centre - q_range_half_length
            q_right = q_centre + q_range_half_length
            approx_result, approx_time = self.query_2d_sum(l=q_left,h=q_right)
            self.logger.logger.info(approx_result)

            sqlStr = "SELECT SUM(y) FROM table" + str(self.dataID) +" WHERE  x BETWEEN " + str(q_left[0]) +" AND " +str(q_right[0])
            self.logger.logger.info(sqlStr)
            exact_result, exact_time = self.query2hive(sql=sqlStr)

            self.logger.logger.info(exact_result)

            if (exact_result is not None) and (exact_result is not 0):
                exact_results.append(exact_result)
                exact_times.append(exact_time)
                approx_results.append(approx_result)
                approx_times.append(approx_time)
            else:
                self.logger.logger.warning("HIVE returns None, so this record is ignored.")
        self.logger.logger.warning("HIVE query results: " + str(exact_results))
        self.logger.logger.warning("HIVE query time cost: " + str(exact_times))
        self.logger.logger.warning("Approximate query results: " + str(approx_results))
        self.logger.logger.warning("Approximate query time cost: " + str(approx_times))
        return exact_results, approx_results, exact_times, approx_times

    def mass_query_avg(self,percent=5,number=default_mass_query_number):
        q_range_half_length = (self.q_max-self.q_min)*percent/100.0/2.0
        random.seed(1.0)
        exact_results = []
        exact_times = []
        approx_results = []
        approx_times = []
        for i in range(number):
            self.logger.logger.info("start query No."+str(i+1) +" out of "+str(number))
            q_centre = random.uniform(self.q_min,self.q_max)
            q_left = q_centre - q_range_half_length
            q_right = q_centre + q_range_half_length
            approx_result, approx_time = self.query_2d_avg(l=q_left,h=q_right)
            self.logger.logger.info(approx_result)

            sqlStr = "SELECT AVG(y) FROM table" + str(self.dataID) +" WHERE  x BETWEEN " + str(q_left[0]) +" AND " +str(q_right[0])
            self.logger.logger.info(sqlStr)
            exact_result, exact_time = self.query2hive(sql=sqlStr)

            self.logger.logger.info(exact_result)
            if (exact_result is not None) and (exact_result is not 0):
                exact_results.append(exact_result)
                exact_times.append(exact_time)
                approx_results.append(approx_result)
                approx_times.append(approx_time)
            else:
                self.logger.logger.warning("HIVE returns None, so this record is ignored.")
        self.logger.logger.warning("HIVE query results: " + str(exact_results))
        self.logger.logger.warning("HIVE query time cost: " + str(exact_times))
        self.logger.logger.warning("Approximate query results: " + str(approx_results))
        self.logger.logger.warning("Approximate query time cost: " + str(approx_times))
        return exact_results, approx_results, exact_times, approx_times

    def mass_query_count(self,percent=5,number=default_mass_query_number):
        q_range_half_length = (self.q_max-self.q_min)*percent/100.0/2.0
        random.seed(1.0)
        exact_results = []
        exact_times = []
        approx_results = []
        approx_times = []
        for i in range(number):
            self.logger.logger.info("start query No."+str(i+1) +" out of "+str(number))
            q_centre = random.uniform(self.q_min,self.q_max)
            q_left = q_centre - q_range_half_length
            q_right = q_centre + q_range_half_length
            approx_result, approx_time = self.query_2d_count(l=q_left,h=q_right)
            self.logger.logger.info(approx_result)

            sqlStr = "SELECT COUNT(y) FROM table" + str(self.dataID) +" WHERE  x BETWEEN " + str(q_left[0]) +" AND " +str(q_right[0])
            self.logger.logger.info(sqlStr)
            exact_result, exact_time = self.query2hive(sql=sqlStr)

            self.logger.logger.info(exact_result)
            if (exact_result is not None) and (exact_result is not 0):
                exact_results.append(exact_result)
                exact_times.append(exact_time)
                approx_results.append(approx_result)
                approx_times.append(approx_time)
            else:
                self.logger.logger.warning("HIVE returns None, so this record is ignored.")
        self.logger.logger.warning("HIVE query results: " + str(exact_results))
        self.logger.logger.warning("HIVE query time cost: " + str(exact_times))
        self.logger.logger.warning("Approximate query results: " + str(approx_results))
        self.logger.logger.warning("Approximate query time cost: " + str(approx_times))
        return exact_results, approx_results, exact_times, approx_times

    def relative_error(self,exact_results, approx_results):
        # abs_errors = [abs(i - j) for i, j in zip(exact_results,approx_results) ]
        rel_errors = [abs(i - j)*1.0/i for i, j in zip(exact_results,approx_results) ]
        # abs_time_reduction = [(j - i) for i, j in zip(exact_times, approx_times) ]
        result = sum(rel_errors)/len(rel_errors)
        self.logger.logger.info("Relative error is : " + str(result))
        return result
    def time_ratio(self,exact_times, approx_times):
        result = sum(approx_times)/sum(exact_times)
        self.logger.logger.info("Time ratio is : " + str(result))
        return result






    def query2hive(self,sql="SHOW TABLES"):


        with pyhs2.connect(host='localhost',
                       port=10000,
                       authMechanism="PLAIN",
                       user='hiveuser',
                       password='hivepassword',
                       database='default') as conn:
            with conn.cursor() as cur:
                #Show databases
                # print cur.getDatabases()

                #Execute query
                # cur.execute("select * from src")
                start = datetime.now()
                cur.execute(sql)
                end = datetime.now()
                time_cost =  (end - start).total_seconds()
                #Return column info from query
                # print cur.getSchema()

                #Fetch table results
                for i in cur.fetch():
                    self.logger.logger.info("Time spent for HIVE query: %.4fs." % time_cost)
                    return i[0], time_cost

    def query2mysql(self,sql="SHOW TABLES"):
        # Open database connection
        db = MySQLdb.connect("127.0.0.1","hiveuser","hivepassword","hivedb" )

        # prepare a cursor object using cursor() method
        cursor = db.cursor()

        # Drop table if it already exist using execute() method.
        # cursor.execute("DROP TABLE IF EXISTS EMPLOYEE")

        # Create table as per requirement
        sql = sql

        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            print row

        # disconnect from server
        db.close()




if __name__ == '__main__':
    qe2d = Query_Engine_2d(5)
    # exact_results, approx_results, exact_times, approx_times = qe2d.mass_query_sum()
    #
    # exact_results, approx_results, exact_times, approx_times = qe2d.mass_query_count(number=1,percent=1)
    # qe2d.relative_error(exact_results, approx_results)
    # qe2d.time_ratio(exact_times, approx_times)
    qe2d.query2mysql("show columns from price_cost_sample_1000000");

