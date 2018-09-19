#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function, division
import numpy as np
import logs
import pandas as pd
from tools import DataSource
from core import CRegression
from query_engine import QueryEngine
from datetime import datetime
import json
import re
logger_file = "../results/deletable.log"


class DBEst:

    def __init__(self, sampleSize="100k",num_of_points=None, dataset="tpcds", logger_file=logger_file):
        start_time = datetime.now()
        self.logger = logs.QueryLogs(log=logger_file)
        self.logger.set_level("INFO")
        self.logger.logger.info("Initialising DBEst...")
        if dataset is "tpcds":
            # initialise the column set, tables in TPC-DS dataset.
            if num_of_points is None:
                num_of_points = {}
                num_of_points["store_sales"]=2685596178
                num_of_points["web_page"]=3000
                num_of_points["time_dim"]=86400
                num_of_points["web_sales"]=720000376
            tableColumnSets = [
                # ["store_sales", "ss_quantity", "*"],
                ["store_sales", "ss_quantity", "ss_ext_discount_amt"],
                ["store_sales", "ss_quantity", "ss_ext_sales_price"],
                ["store_sales", "ss_quantity", "ss_ext_list_price"],
                ["store_sales", "ss_quantity", "ss_ext_tax"],
                ["store_sales", "ss_quantity", "ss_net_paid"],
                ["store_sales", "ss_quantity", "ss_net_paid_inc_tax"],
                ["store_sales", "ss_quantity", "ss_net_profit"],
                ["store_sales", "ss_quantity", "ss_list_price"],
                ["store_sales", "ss_list_price", "ss_list_price"],
                ["store_sales", "ss_coupon_amt", "ss_list_price"],
                ["store_sales", "ss_wholesale_cost", "ss_list_price"],
                ["store_sales", "ss_sales_price", "ss_quantity"],
                ["store_sales", "ss_net_profit", "ss_quantity"],
                ["web_page", "wp_char_count", "wp_link_count"],             # *
                ["time_dim", "t_minute", "t_hour"],                         # *
                ["web_sales", "ws_sales_price", "ws_quantity"]
                ]
            tables = [sublist[0] for sublist in tableColumnSets]
            self.uniqueTables = list(set(tables))
            self.logger.logger.info(
                "Dataset contains " + str(len(self.uniqueTables)) +
                " tables, which are:")
            self.logger.logger.info(self.uniqueTables)
            self.uniqueTCS = []
            for element in tableColumnSets:
                if element not in self.uniqueTCS:
                    self.uniqueTCS.append(element)
            self.numOfCsOfTables = [tables.count(uniqueTableName) for
                                    uniqueTableName in self.uniqueTables]
            self.logger.logger.info(
                "Talbes in the dataset need " + str(self.numOfCsOfTables) +
                " Column Sets.")

            # get column set in each table
            self.CSinTable = {}
            for uniqueTable in self.uniqueTables:
                columnSet = [[item[1], item[2]]
                             for item in self.uniqueTCS if item[0] is uniqueTable]
                self.logger.logger.debug(columnSet)
                self.CSinTable[uniqueTable]=columnSet
            self.logger.logger.debug(self.CSinTable)

            self.DBEstClients={}     #store all QeuryEngines, for each table
            # load data
            for uniqueTable in self.uniqueTables:
                if sampleSize is "100k":
                    csv_file="../data/tpcDs100k/"+uniqueTable+".csv"
                if sampleSize is "10k":
                    csv_file="../data/tpcDs10k/"+uniqueTable+".csv"
                #self.logger.logger.info(csv_file)
                df=pd.read_csv(csv_file)
                #self.logger.logger.info(df.to_string())
                df=df.dropna()
                DBEstiClient={}     #store all QeuryEngines within each table
                for columnItem in self.CSinTable[uniqueTable]:
                    self.logger.logger.info("--------------------------------------------------")
                    self.logger.logger.info("Start training Qeury Engine for Table "+uniqueTable+
                        ", Column Set: "+str(columnItem))
                    headerX = columnItem[0]
                    headerY = columnItem[1]
                    x=df[[headerX]].values
                    y=df[[headerY]].values.reshape(-1)
                    
                    self.data=DataSource()
                    self.data.features=x
                    self.data.labels=y
                    self.data.headers=columnItem
                    self.data.file=csv_file
                    #self.data.removeNAN()

                    cRegression = CRegression(logger_object=self.logger)
                    cRegression.fit(self.data)
                    
                    qe = QueryEngine(
                        cRegression, logger_object=self.logger,
                        num_training_points=num_of_points[uniqueTable])
                    qe.density_estimation()
                    DBEstiClient[str(columnItem)]=qe
                    self.logger.logger.info("Finish training Qeury Engine for Table "+uniqueTable+
                        ", Column Set: "+str(columnItem))
                    self.logger.logger.info("--------------------------------------------------")
                self.logger.logger.debug(DBEstiClient)
                self.DBEstClients[uniqueTable]=DBEstiClient
            self.logger.logger.info(self.DBEstClients)
            # self.logger.logger.info(json.dumps(DBEstClients, indent=4))
        end_time = datetime.now()
        time_cost = (end_time-start_time).total_seconds()
        self.logger.logger.info("DBEsti has been initialised, ready to serve... (%.1fs)"%time_cost)

    def mass_query_simple(self, file):
        AQP_results = []
        time_costs = []
        index = 1
        with open(file) as fin:
            for line in fin:
                self.logger.logger.info("Starting Query " + str(index) + ":")
                self.logger.logger.info(line)
                index = index + 1
                query_list=line.replace("(","").replace(")","")
                query_list =  re.split('\s+',query_list)
                # remove empty strings caused by sequential blank spaces.
                query_list=list(filter(None,query_list))
                func = query_list[1]
                tbl =  query_list[4]
                x = query_list[6]
                y = query_list[2]
                lb = query_list[8]
                hb = query_list[10]


                if y =="*":
                    columnSetsInTable = self.CSinTable[tbl]
                    for cs in columnSetsInTable:
                        if x ==cs[0]:
                            self.logger.logger.debug("Find a column set in table "+ tbl+ " to replace *: " +
                                str(cs))
                            y = cs[1]
                        break
                columnItem=str([x,y])
                if func == "avg":
                    DBEstClient=self.DBEstClients[tbl]
                    result, time=DBEstClient[columnItem].approximate_avg_from_to(float(lb),float(hb),0)
                    AQP_results.append(result)
                    time_costs.append(time)
                elif func == "sum":
                    DBEstClient=self.DBEstClients[tbl]
                    result, time=DBEstClient[columnItem].approximate_sum_from_to(float(lb),float(hb),0)
                    AQP_results.append(result)
                    time_costs.append(time)
                elif func == "count":
                    DBEstClient=self.DBEstClients[tbl]
                    result, time=DBEstClient[columnItem].approximate_count_from_to(float(lb),float(hb),0)
                    AQP_results.append(result)
                    time_costs.append(time)
                else:
                    self.logger.logger.warning(func +" is currently not supported!")
            self.logger.logger.info(AQP_results)
            self.logger.logger.info(time_costs)



                
                # result, time = self.query_2d_percentile(float(line))
                

if __name__ == "__main__":
    db = DBEst(dataset="tpcds",sampleSize="10k")
    db.mass_query_simple(file="../query/tpcds/qreg/sample.qreg")
