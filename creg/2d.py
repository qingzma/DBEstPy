#!/usr/bin/env python
# coding=utf-8
from core import CRegression
import data_loader as dl
from query_engine import QueryEngine
import logs
import random
import commands
from pyhive import hive
import subprocess
import os


class Query_Engine_2d:
    def __init__(self,dataID,b_allow_repeated_value=False):
        self.logger = logs.QueryLogs()
        self.logger.set_no_output()
        self.data = dl.load2d(dataID)
        if not b_allow_repeated_value:
            self.data.remove_repeated_x_1d()
        self.cRegression = CRegression(logger_object=self.logger)
        self.cRegression.fit(self.data)
        self.logger.set_logging()
        self.qe = QueryEngine(self.cRegression, logger_object=self.logger)
        self.qe.density_estimation()
        self.q_min = min(self.data.features)
        self.q_max = max(self.data.features)
    def query_2d_avg(self,l=0,h=100):
        """query to 2d data sets.

        Args:
            l (int, optional): query lower boundary
            h (int, optional): query higher boundary
        """
        avgs = self.qe.approximate_avg_from_to(l ,h, 0)  #0.05E8,0.1E8,
        return avgs

    def query_2d_sum(self,l=0,h=100):
        """query to 2d data sets.

        Args:
            l (int, optional): query lower boundary
            h (int, optional): query higher boundary
        """
        sums = self.qe.approximate_sum_from_to(l ,h, 0)
        return sums
    def mass_query_sum(self,percent=5,number=1000):
        q_range_half_length = (self.q_max-self.q_min)*percent/100.0/2.0
        random.seed(1.0)
        for i in range(number):
            q_centre = random.uniform(self.q_min,self.q_max)
            q_left = q_centre - q_range_half_length
            q_right = q_centre + q_range_half_length
            print(self.query_2d_sum(l=q_left,h=q_right))
    def blinkdb(self):
        # cmd1 = "cd /home/u1796377/Program/blinkdb/"
        # cmd2 = "./bin/blinkdb -S -e 'SELECT * FROM db_name.table_name LIMIT 1;' "

        # status, output = commands.getstatusoutput(cmd1)
        # print(output)
        # status, output = commands.getstatusoutput(cmd2)

        # if status == 0:
        #     print output
        # else:
        #     print output

        cmd1 = "cd /home/u1796377/Program/blinkdb"
        cmd2 = "./bin/blinkdb -e 'show tables;' "


        p1=subprocess.Popen(['/bin/bash','pwd'], shell=True,cwd="/home/u1796377/Program/blinkdb",stdout=subprocess.PIPE)
        print(p1.communicate())
        # print(output)
        # status, output = commands.getstatusoutput(cmd2)

        # if status == 0:
        #     print output
        # else:
        #     print output
        # os.popen("start cmd")

    def query(self):
        # from pyhive import hive
        # conn = hive.Connection(host="localhost", port=10000, username="APP")
        # cursor = conn.cursor()
        # cursor.execute("SHOW TABLES")
        # for result in cursor.fetchall():
        #     use_result(result)
        #     print(result)
        #
        #
        # from pyhive import hive
        # try:
        #     cursor = hive.connect('localhost').cursor()
        #     cursor.execute('SHOW TABLES',async=True)
        #     print cursor.fetchall()
        # except Exception as e:
        #     print(e)
        #     # raise e
        # finally:
        #     print("end")


        import pyhs2
        with pyhs2.connect(host='localhost',
                       port=10000,
                       authMechanism="PLAIN",
                       user='bayern',
                       password='as86442576',
                       database='default') as conn:
            with conn.cursor() as cur:
                #Show databases
                print(cur.getDatabases())

                #Execute query
                cur.execute("show tables")

                #Return column info from query
                print(cur.getSchema())

                #Fetch table results
                for i in cur.fetch():
                    print(i)


if __name__ == '__main__':
    qe2d = Query_Engine_2d(5)
    # qe2d.mass_query_sum()
    qe2d.query()
