#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function, division
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dbest import logs
from dbest.tools import DataSource
from dbest import tools
from dbest.qreg import CRegression
from dbest.query_engine import QueryEngine

import pandas as pd
from datetime import datetime
import re
import gc  # to delete variables
import warnings
import pickle

# import threading
# from multiprocessing.pool import ThreadPool
from multiprocessing import Process
import multiprocessing as mp


logger_file = "../results/deletable.log"
line_break="-----------------------------------------------------------------------------------------"

# epsabs = 10         #1E-3
# epsrel = 1E-01      #1E-1
# mesh_grid_num = 20  #30
# limit =30


epsabs = 1E-3
epsrel = 1E-1
mesh_grid_num = 30
limit = 30


class DBEst:

    """The implementation of DBEst, which uses regression models to 
    give approximate answers for aggregate functions. 
    Currently support queries includes AVG, COUNT, SUM, MIN, MAX, VARIANCE, COVARIANCE, PERCENTIRL, and GROUP BY.

    Attributes:
        CSinTable (dict): Description
        dataset (TYPE): Description
        DBEstClients (dict): Description
        df (dict): Description
        group_names (dict): Description
        logger (TYPE): Description
        num_of_points (dict): Description
        num_of_points_per_group_tbls (dict): Description
        numOfCsOfTables (TYPE): Description
        tableColumnSets (list): Description
        uniqueTables (TYPE): Description
        uniqueTCS (list): Description
    """

    def __init__(self, dataset, logger_file=logger_file, base_models=[tools.app_xgboost], n_jobs=4): 
        self.dataset = dataset
        self.logger = logs.QueryLogs(log=logger_file)
        self.logger.set_level("INFO")
        self.logger.logger.info("Initialising DBEst...")
        self.group_names = {}
        self.num_of_points_per_group_tbls = {}
        self.num_of_points = {}
        self.df = {}
        self.DBEstClients = {}
        self.model_catalog = {}
        self.tableColumnSets = []  # store all QeuryEngines, for each table
        self.base_models = base_models
        self.logger.logger.info("Ready to serve queries!")
        self.logger.logger.info(line_break)
        self.n_jobs=n_jobs
        self.warehouse=[]
        self.warehouse_serielize=[]
        # warnings.filterwarnings(action='ignore', category=DeprecationWarning)

    def generate_model_catalog_string(self, table_name, columnPair, groupbyID=None, groupby_value=None):
        """Generate the string holding the model catalog information, each model has a unique model catalog

        Args:
            client (TYPE):  DBEst client, consists of a regrssion model and a density estimator
            table_name (String): the table name
            columnPair (List): the coolumn pair for a query
            groupbyID (None, String): groupby attribute
            groupby_value (None, String): groupby value, for example, group by month, you should input 'January', etc.
        Returns:
            String: the model catalog
        """
        client_identifier = str(columnPair)
        if groupbyID != None:
            # Check if the groupby_value is provided.
            if groupby_value == None:
                self.logger.logger.warning("Please provide the groupby_value!")
                return
            client_identifier += "-groupby-"
            client_identifier += str([groupbyID, groupby_value])
        return client_identifier

    def check_and_add_client(self, client, table_name, columnPair, groupbyID=None, groupby_value=None):
        """Summary

        Args:
            client (TYPE): DBEst client, consists of a regrssion model and a density estimator
            table_name (String): the table name
            columnPair (List): the coolumn pair for a query
            groupbyID (None, String): groupby attribute
            groupby_value (None, String): groupby value, for example, group by month, you should input 'January', etc.
        """
        if table_name not in self.DBEstClients:
            self.logger.logger.debug("DBEst does not hold any model catalog for Table "
                                     + table_name + ", so create the model catalog for it.")
            self.DBEstClients[table_name] = {}

        # Check if the groupby_value and groupbyID are provided at the same
        # time.
        if groupbyID != None:
            if groupby_value == None:
                self.logger.logger.warning("Please provide the groupby_value!")
                return
        client_identifier = self.generate_model_catalog_string(
            table_name, columnPair, groupbyID, groupby_value)
        if client_identifier not in self.DBEstClients:
            self.DBEstClients[table_name][client_identifier]=client
        else:
            self.logger.logger.info(client_identifier+" exists in DBEst, so ignore adding it.")

    def del_client(self,client, table_name, columnPair, groupbyID=None, groupby_value=None):
        """ delete a DBEst client
        
        Args:
            client (TYPE): DBEst client, consists of a regrssion model and a density estimator
            table_name (String): the table name
            columnPair (List): the coolumn pair for a query
            groupbyID (None, String): groupby attribute
            groupby_value (None, String): groupby value, for example, group by month, you should input 'January', etc.
        """
        client_identifier = self.generate_model_catalog_string(
            table_name, columnPair, groupbyID, groupby_value)
        if client_identifier in self.DBEstClients:
            del self.DBEstClients[client_identifier]
        else:
            self.logger.logger.info("No such model in DBEst, so could not operate the deletion operation.")

    def add_pair_client(self, file, table_name, columnPair, num_of_points, groupbyID=None, groupby_value=None):
        start_time = datetime.now()
        # check if the table exists in model catalog, if not, create one.
        if table_name not in self.model_catalog:
            self.model_catalog[table_name]={}
            self.df[table_name]={}

        model_index = self.generate_model_catalog_string(table_name, columnPair, groupbyID, groupby_value)
        if model_index in self.model_catalog[table_name]:
            self.logger.logger.warning(model_index +" already in DBEst, skip training!")
            self.logger.logger.info(line_break)
            return
        else:
            self.logger.logger.info("Start training "+model_index)

            self.df[table_name][model_index] = pd.read_csv(file)
            self.df[table_name][model_index] = self.df[table_name][model_index].dropna()
            headerX = columnPair[0]
            headerY = columnPair[1]
            x = self.df[table_name][model_index][[headerX]].values
            y = self.df[table_name][model_index][[headerY]].values.reshape(-1)

            data = DataSource()
            data.features = x
            data.labels = y
            data.headers = columnPair
            data.file = file
            # self.data.removeNAN()

            cRegression = CRegression(
                logger_object=self.logger, base_models=self.base_models)
            cRegression.fit(data)

            qe = QueryEngine(
                cRegression, logger_object=self.logger,
                num_training_points=num_of_points)
            qe.density_estimation()
            cRegression.clear_training_data()
            # qe.get_size()
            self.model_catalog[table_name][model_index] = qe

            self.logger.logger.info("Finish training model "+model_index + " for Table "+table_name)
            end_time = datetime.now()
            time_cost = (end_time - start_time).total_seconds()
            self.logger.logger.info("Ready to serve... (%.1fs)"% time_cost)
            self.logger.logger.info(line_break)
            return qe
            
    def del_pair_client(self, table_name, columnPair,  groupbyID=None, groupby_value=None):
        model_index = self.generate_model_catalog_string(table_name, columnPair, groupbyID, groupby_value)
        del self.model_catalog[table_name][model_index]
        self.logger.logger.info(model_index+" has been deleted from model catalog.")
        self.logger.logger.info(line_break)











    def init_whole_range(self, file, table, columnItems, num_of_points):
        """Build DBEst table for a table, with different combinations of column pairs.

        Args:
            table (str): table names, for example: 'store_sales'
            file (str): path to the file
            columnItem (list): columnItem of each table, for example, 
                        [["ss_quantity", "ss_ext_discount_amt"],...]
            num_of_points (dict, optional): Description

        Deleted Parameters:
            tableColumnSets (None, optional): Description
        """
        start_time = datetime.now()

        tableColumnSets = [[table] + columnItems[i]
                           for i in range(len(columnItems))]

        # self.logger.logger.info(tableColumnSets)

        # if not hasattr(self, 'df'):
        #     self.df = {}

        if self.dataset == "tpcds":
            # initialise the column set, tables in TPC-DS dataset.
            if num_of_points is None:
                num_of_points = {}
                num_of_points["store_sales"] = 2685596178
                num_of_points["web_page"] = 3000
                num_of_points["time_dim"] = 86400
                num_of_points["web_sales"] = 720000376
            # merge dict num_of_points to self.num_of_points
            # self.logger.logger.info(num_of_points)
            self.num_of_points = {**self.num_of_points, **num_of_points}
            # self.logger.logger.info(self.num_of_points)

            if tableColumnSets is None:
                tableColumnSets = [
                    ["store_sales", "ss_quantity", "ss_ext_discount_amt"]  # ,
                    # ["store_sales", "ss_quantity", "ss_ext_sales_price"],
                    # ["store_sales", "ss_quantity", "ss_ext_list_price"],
                    # ["store_sales", "ss_quantity", "ss_ext_tax"],
                    # ["store_sales", "ss_quantity", "ss_net_paid"],
                    # ["store_sales", "ss_quantity", "ss_net_paid_inc_tax"],
                    # ["store_sales", "ss_quantity", "ss_net_profit"],
                    # ["store_sales", "ss_quantity", "ss_list_price"],
                    # ["store_sales", "ss_list_price", "ss_list_price"],
                    # ["store_sales", "ss_coupon_amt", "ss_list_price"],
                    # ["store_sales", "ss_wholesale_cost", "ss_list_price"],
                    # ["store_sales", "ss_sales_price", "ss_quantity"],
                    # ["store_sales", "ss_net_profit", "ss_quantity"]  # ,
                    # ["web_page", "wp_char_count", "wp_link_count"],   # *
                    # ["time_dim", "t_minute", "t_hour"],               # *
                    # ["web_sales", "ws_sales_price", "ws_quantity"]
                ]
            # self.logger.logger.info(tableColumnSets)
            self.tableColumnSets = self.tableColumnSets + tableColumnSets
            # self.logger.logger.info(self.tableColumnSets)
        if self.dataset == "pp":
            if num_of_points is None:
                num_of_points = {}
                num_of_points["powerplant"] = 26000000000
            # merge dict num_of_points to self.num_of_points
            self.num_of_points = {**self.num_of_points, **num_of_points}

            if tableColumnSets is None:
                tableColumnSets = [
                    # ["store_sales", "ss_quantity", "*"],
                    ["powerplant", "T", "EP"],
                    ["powerplant", "AP", "EP"],
                    ["powerplant", "RH", "EP"]
                ]
            self.tableColumnSets = self.tableColumnSets + tableColumnSets

        if self.dataset == "zipf":
            if num_of_points is None:
                num_of_points = {}
                num_of_points["zipf"] = 100000000
            # merge dict num_of_points to self.num_of_points
            self.num_of_points = {**self.num_of_points, **num_of_points}

            if tableColumnSets is None:
                tableColumnSets = [
                    # ["store_sales", "ss_quantity", "*"],
                    ["zipf", "x", "y"]
                ]
            self.tableColumnSets = self.tableColumnSets + tableColumnSets


        tables = [sublist[0] for sublist in self.tableColumnSets]
        
        

        
        self.uniqueTables = list(set(tables))
        self.logger.logger.info(
            "Dataset contains " + str(len(self.uniqueTables)) +
            " tables, which are:")
        self.logger.logger.info(self.uniqueTables)
        self.uniqueTCS = []
        for element in self.tableColumnSets:
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
            self.CSinTable[uniqueTable] = columnSet
        self.logger.logger.debug(self.CSinTable)

        # load data
        for uniqueTable in self.uniqueTables:
            self.df[uniqueTable] = pd.read_csv(file)
            # self.logger.logger.info(df.to_string())
            self.df[uniqueTable] = self.df[uniqueTable].dropna()

            if uniqueTable in self.DBEstClients:
                DBEstiClient = self.DBEstClients[uniqueTable]
            else:
                DBEstiClient = {}  # store all QeuryEngines within each table

            for columnItem in self.CSinTable[uniqueTable]:
                if ((uniqueTable in self.DBEstClients) and
                        (str(columnItem) not in self.DBEstClients[uniqueTable])
                        or uniqueTable not in self.DBEstClients):
                    # the if sentence above judges whether previous model has
                    #  been trained, if so, skip re-train it
                    self.logger.logger.info(
                        "--------------------------------------------------")
                    self.logger.logger.info("Start training Qeury Engine for" +
                                            " Table " + uniqueTable +
                                            ", Column Set: " + str(columnItem))
                    headerX = columnItem[0]
                    headerY = columnItem[1]
                    x = self.df[uniqueTable][[headerX]].values
                    y = self.df[uniqueTable][[headerY]].values.reshape(-1)

                    data = DataSource()
                    data.features = x
                    data.labels = y
                    data.headers = columnItem
                    data.file = file
                    # self.data.removeNAN()

                    cRegression = CRegression(
                        logger_object=self.logger, base_models=self.base_models)
                    cRegression.fit(data)

                    qe = QueryEngine(
                        cRegression, logger_object=self.logger,
                        num_training_points=self.num_of_points[uniqueTable])
                    qe.density_estimation()
                    cRegression.clear_training_data()
                    # qe.get_size()
                    DBEstiClient[str(columnItem)] = qe

                    self.logger.logger.info("Finish training Qeury Engine " +
                                            "for Table " + uniqueTable +
                                            ", Column Set: " + str(columnItem))
                    self.logger.logger.info(
                        "--------------------------------------------------")
                    self.logger.logger.debug(DBEstiClient)
                    self.DBEstClients[uniqueTable] = DBEstiClient
                else:
                    self.logger.logger.info(
                        "This model exsits, not need to train, so just skip training! ")
        self.logger.logger.info(self.DBEstClients)
        # self.logger.logger.info(json.dumps(DBEstClients, indent=4))
        end_time = datetime.now()
        time_cost = (end_time - start_time).total_seconds()
        self.logger.logger.info(
            "DBEsti has been initialised, ready to serve... (%.1fs)"
            % time_cost)

    def init_groupby(self, file="../data/tpcDs10k/store_sales.csv",
                     table="store_sales", group="ss_store_sk",
                     columnItem=["ss_wholesale_cost", "ss_list_price"],
                     num_of_points_per_group=None):
        """ support simple group by,

        Args:
            table (str, optional): table name
            group (str, optional): column name of the group
            columnItem (list, optional): [x, y], in list format
            num_of_points_per_group (Dictionary, optional): store the total number of points of each group 
        """
        start_time = datetime.now()
        self.logger.logger.info("")
        self.logger.logger.info("Start building GROUP BY for Table " + table)
        self.df[table] = pd.read_csv(file)
        self.df[table] = self.df[table].dropna()
        # self.logger.logger.info(self.df[table]["ss_sold_date_sk"])
        # self.df[group] =pd.Series([], dtype=int)
        grouped = self.df[table].groupby(group)
        group_name = str([table, group])
        self.group_names[group_name] = []
        if table not in self.DBEstClients:
            self.DBEstClients[table] = {}

        # initiate the number of points per group
        if num_of_points_per_group is None:
            self.logger.logger.error("Please provide the information\
                (num_of_points_per_group) for init_groupby() ")
            sys.exit()
        else:
            self.num_of_points_per_group_tbls[str(
                [table, group])] = num_of_points_per_group

        for grp_name, group in grouped:

            self.group_names[group_name].append(grp_name)
            columnItemGroup = str(columnItem) + "-" + str(grp_name)
            self.logger.logger.debug(
                "--------------------------------------------------")
            self.logger.logger.debug(
                "Start building groupy by for " + columnItemGroup)
            headerX = columnItem[0]
            headerY = columnItem[1]
            x = group[[headerX]].values
            y = group[[headerY]].values.reshape(-1)

            data = DataSource()
            data.features = x
            data.labels = y
            data.headers = columnItem
            data.file = columnItemGroup
            # self.data.removeNAN()

            cRegression = CRegression(
                logger_object=self.logger, b_cross_validation=True, base_models=self.base_models)
            cRegression.fit(data)
            #
            #
            #
            #
            #
            #
            #
            #
            #
            # number of points
            # self.logger.logger.info(num_of_points_per_group)
            self.logger.logger.info(grp_name)
            # self.logger.logger.info(group)
            try:
                qe = QueryEngine(
                    cRegression, logger_object=self.logger,
                    num_training_points=int(
                        num_of_points_per_group[str((grp_name))]))
            except:
                qe = QueryEngine(
                    cRegression, logger_object=self.logger,
                    num_training_points=int(
                        # num_of_points_per_group[str((grp_name))]))
                        num_of_points_per_group[str((int(grp_name)))]))
                # num_of_points_per_group[str(int(grp_name))]))
            qe.density_estimation()
            cRegression.clear_training_data()
            # qe.get_size()
            self.DBEstClients[table][columnItemGroup] = qe
            self.logger.logger.info(
                "Start building groupy by for " + columnItemGroup)
            self.logger.logger.info(
                "--------------------------------------------------")

            # result, time = self.query_2d_percentile(float(line))
        end_time = datetime.now()
        time_cost = (end_time - start_time).total_seconds()
        self.logger.logger.info(
            "GROUP BY has been initialised, ready to serve... (%.1fs)"
            % time_cost)

    def mass_query_simple(self, file, epsabs=epsabs, epsrel=epsrel, limit=limit, ci=True, confidence=0.95):
        AQP_results = []
        time_costs = []
        index = 1
        with open(file) as fin:
            for line in fin:
                self.logger.logger.info("Starting Query " + str(index) + ":")
                self.logger.logger.info(line)
                index = index + 1
                query_list = line.replace(
                    "(", " ").replace(")", " ").replace(";", "").replace(",", "")
                # self.logger.logger.info(query_list)
                query_list = re.split('\s+', query_list)
                # remove empty strings caused by sequential blank spaces.
                query_list = list(filter(None, query_list))
                func = query_list[1]

                if func.lower() != "percentile":
                    x = query_list[6]
                    y = query_list[2]
                    lb = query_list[8]
                    hb = query_list[10]
                    tbl = query_list[4]
                    p=None

                else:
                    x = query_list[2]
                    y = "*"
                    lb=None
                    hb=None
                    p = query_list[3]
                    tbl = query_list[5]

                if y == "*":
                    columnSetsInTable = self.CSinTable[tbl]
                    for cs in columnSetsInTable:
                        # self.logger.logger.info("x is "+ str(x) +",  cs[0] is "+ str(cs[0]))
                        if x == cs[0]:
                            self.logger.logger.debug("Find a column \
                                set in table " + tbl + " to replace *: " +
                                                     str(cs))
                            y = cs[1]
                            break
                if y == "*":
                    self.logger.logger.error(
                        "There is no model to predict percentile!")
                    AQP_results.append("Null")
                    time_costs.append("Null")
                    break

                columnItem = str([x, y])
                DBEstClient = self.DBEstClients[tbl]

                result,time=querySQL(DBEstClient,columnItem,func,lb,hb,p)
                AQP_results.append(result)
                time_costs.append(time)
                # if func.lower() == "avg":

                #     result, time = DBEstClient[columnItem].\
                #         approximate_avg_from_to(float(lb), float(
                #             hb), 0, epsabs=epsabs, epsrel=epsrel, limit=limit)
                #     AQP_results.append(result)
                #     time_costs.append(time)
                # elif func.lower() == "sum":
                #     # DBEstClient = self.DBEstClients[tbl]
                #     result, time = DBEstClient[columnItem].\
                #         approximate_sum_from_to(float(lb), float(
                #             hb), 0, epsabs=epsabs, epsrel=epsrel, limit=limit)
                #     AQP_results.append(result)
                #     time_costs.append(time)
                # elif func.lower() == "count":
                #     # DBEstClient = self.DBEstClients[tbl]
                #     # self.logger.logger.info("table "+ str(tbl))
                #     # self.logger.logger.info("lb "+ str(lb))
                #     # self.logger.logger.info("hb "+ str(hb))
                #     result, time = DBEstClient[columnItem].\
                #         approximate_count_from_to(float(lb), float(
                #             hb), 0, epsabs=epsabs, epsrel=epsrel, limit=limit)
                #     AQP_results.append(result)
                #     time_costs.append(time)
                # elif func.lower() == 'variance_x':
                #     result, time = DBEstClient[columnItem].approximate_variance_x_from_to(
                #         float(lb), float(hb), 0, epsabs=epsabs, epsrel=epsrel, limit=limit)
                #     AQP_results.append(result)
                #     time_costs.append(time)

                # elif func.lower() == 'min':
                #     result, time = DBEstClient[columnItem].approximate_min_from_to(
                #         float(lb), float(hb), 0, ci=ci, confidence=confidence)
                #     AQP_results.append(result)
                #     time_costs.append(time)

                # elif func.lower() == 'max':
                #     result, time = DBEstClient[columnItem].approximate_max_from_to(
                #         float(lb), float(hb), 0, ci=ci, confidence=confidence)
                #     AQP_results.append(result)
                #     time_costs.append(time)

                # elif func.lower() == 'covar':
                #     result, time = DBEstClient[columnItem].approximate_covar_from_to(
                #         float(lb), float(hb), 0, epsabs=epsabs, epsrel=epsrel, limit=limit)
                #     AQP_results.append(result)
                #     time_costs.append(time)

                # elif func.lower() == 'corr':
                #     result, time = DBEstClient[columnItem].approximate_corr_from_to(
                #         float(lb), float(hb), 0)
                #     AQP_results.append(result)
                #     time_costs.append(time)
                # elif func.lower() == 'percentile':
                #     result, time = DBEstClient[columnItem].approximate_percentile_from_to(
                #         float(p))
                #     AQP_results.append(result)
                #     time_costs.append(time)

                # else:
                #     self.logger.logger.warning(
                #         func + " is currently not supported!")
            self.logger.logger.info(AQP_results)
            self.logger.logger.info(time_costs)
    def mass_query_simple_parallel(self, file, epsabs=epsabs, epsrel=epsrel, limit=limit, ci=True, confidence=0.95):
        AQP_results = []
        time_costs = []
        index = 1
        queries=[]
        with open(file) as fin:
            for line in fin:
                queries.append(line)

        width = int(len(queries)/self.n_jobs)
        subgroups=[queries[inde:inde+width] for inde in range(0,len(queries),width)]
        if len(queries)%self.n_jobs !=0:
            subgroups[self.n_jobs-1]=subgroups[self.n_jobs-1]+subgroups[self.n_jobs]
            del subgroups[-1]
        # self.logger.logger.info(subgroups)

        # 


        # clients=pickle.dumps(self.DBEstClients)

        # for fin in subgroups:
            
        start_time=datetime.now()
        # print(querySQLs(queries,self.DBEstClients,self.CSinTable))
        processes=[]
        times_in_group=[]
        # for idx,subgroup in enumerate(subgroups):

        for subgroup in subgroups:
            # print("group is "+ str(idx))
            
            predictions=[]
            time_costs=[]
            time=0.0
            # clientss=pickle.loads(clients)
            # t = threading.Thread(target=querySQLgroups, args=(subgroup,DBEstClient,columnItem,func,x,y,lb,hb,p,results_parallel,index_in_groups[idx]))
            t = Process(target=querySQLsWithReturnValues, args=(subgroup,self.DBEstClients,self.CSinTable,predictions,time_costs,time))
            processes.append(t)
            t.start()
            times_in_group.append(time)
        start_time=datetime.now()
        # for t in processes:
        #     t.start()
        for t in processes:
            t.join()

        # print("Finished!")
        end_time=datetime.now()
        time_cost = (end_time - start_time).total_seconds()
        self.logger.logger.info(
            "Query response time is (%.4fs)"
            % time_cost)
        self.logger.logger.info(times_in_group)
        self.logger.logger.info("_______________________________________________________________________________")
        return time_cost
            




            
                
        #     self.logger.logger.info(AQP_results)
        #     self.logger.logger.info(time_costs)

    def query_simple_groupby(self, query, output=None, epsabs=epsabs, epsrel=epsrel, limit=limit, ci=True, confidence=0.95):
        
        AQP_results = []
        time_costs = []
        groups_results = []
        index = 1
        line = query
        self.logger.logger.info("Starting Query " + str(index) + ":")
        self.logger.logger.info(line)
        index = index + 1
        query_list = line.replace("(", " ").replace(")", " ").replace(";", "")
        # self.logger.logger.info(query_list)
        query_list = re.split('\s+', query_list)
        # remove empty strings caused by sequential blank spaces.
        query_list = list(filter(None, query_list))
        func = query_list[1]
        if func.lower() != "percentile":
            x = query_list[6]
            y = query_list[2]
            lb = query_list[8]
            hb = query_list[10]
            tbl = query_list[4]
            grp = query_list[13]
            p=None
        else:
            x = query_list[2]
            y = "*"
            p = query_list[3]
            tbl = query_list[5]
            lb=None
            hb=None

        if y == "*":
            columnSetsInTable = self.CSinTable[tbl]
            for cs in columnSetsInTable:
                if x == cs[0]:
                    self.logger.logger.debug("Find a column \
                        set in table " + tbl + " to replace *: " +
                                             str(cs))
                    y = cs[1]
                    break
        if y == "*":
            self.logger.logger.error(
                "There is no model to predict percentile!")
            AQP_results.append("Null")
            time_costs.append("Null")
        else:
            columnItem = str([x, y])


            groups=self.group_names[str([tbl, grp])]
            # self.logger.logger.info(groups)
            width = int(len(groups)/self.n_jobs)
            # self.logger.logger.info(width)
            subgroups=[groups[inde:inde+width] for inde in range(0,len(groups),width)]
            # self.logger.logger.info(subgroups)
            if len(groups)%self.n_jobs !=0:
                subgroups[self.n_jobs-1]=subgroups[self.n_jobs-1]+subgroups[self.n_jobs]
                del subgroups[-1]
            
            index_in_groups=[[groups.index(sgname) for sgname in sgnames] for sgnames in subgroups]
            # self.logger.logger.info(subgroups)
            # self.logger.logger.info(index_in_groups)
            


            DBEstClient = self.DBEstClients[tbl]

            threads=[]
            results_parallel=[[0.0 for _ in range(len(groups))] for _ in [1,2,3]]
            start_time=datetime.now()
            for idx,subgroup in enumerate(subgroups):
                # t = threading.Thread(target=querySQLgroups, args=(subgroup,DBEstClient,columnItem,func,x,y,lb,hb,p,results_parallel,index_in_groups[idx]))
                t = Process(target=querySQLgroups, args=(subgroup,DBEstClient,columnItem,func,x,y,lb,hb,p,results_parallel,index_in_groups[idx]))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
                # group_results_in_group,AQP_result_in_group,time_cost_in_group=querySQLgroups(subgroup,DBEstClient,columnItem,func,x,y,lb,hb,p)
            # pool = ThreadPool(processes=4)
            # async_result = pool.apply_async(querySQLgroups,(groups,DBEstClient,columnItem,func,x,y,lb,hb,p))
            # return_val = async_result.get()
            # self.logger.logger.info(return_val)
            # return

            # querySQLgroups(groups,DBEstClient,columnItem,func,x,y,lb,hb,p,results_parallel,index_in_groups)

            # for grp_name in self.group_names[str([tbl, grp])]:
            #     columnItem = str([x, y])
            #     columnItem = str(columnItem) + "-" + str(grp_name)
            #     DBEstClient = self.DBEstClients[tbl]
            #     result, time=querySQL(DBEstClient,columnItem,func,lb,hb,p)
            #     AQP_results.append(result)
            #     time_costs.append(time)
            #     groups_results.append(grp_name)
            groups_results = results_parallel[0]
            AQP_results = results_parallel[1]
            time_costs = results_parallel[2]

            end_time=datetime.now()
            

            self.logger.logger.info(groups_results)
            self.logger.logger.info(AQP_results)
            self.logger.logger.info(time_costs)

            time_cost = (end_time - start_time).total_seconds()
            self.logger.logger.info(
                "Query response time is (%.4fs)"
                % time_cost)

            if output is not None:
                with open(output, 'w+') as file:
                    for i in range(len(groups_results)):
                        file.write('%s, %s\n' % (
                            str(int(groups_results[i])), str(AQP_results[i])))


    def read_num_of_points_per_group(self, file):
        num_of_points = {}
        with open(file) as fin:
            for line in fin:
                group_count = line.replace(
                    "(", " ").replace(")", " ").replace(";", "")
                group_count = re.split('\s+', group_count)
                num_of_points[group_count[0]] = group_count[1]
                # self.logger.logger.debug(group_count[0]+","+group_count[1])

        return num_of_points

    def clear_training_data(self):
        if self.df is not None:
            del self.df
        gc.collect()

    # def del_client(self, table, columnItem):
    #     # self.DBEstClients[table][str(columnItem)]=None
    #     del self.DBEstClients[table][str(columnItem)]
    #     gc.collect()

    def get_size(self, b_models_only=True):
        if b_models_only:
            size = 0.0
            idx=0
            filenames=[]

            for table in self.DBEstClients:
                for columnItem in self.DBEstClients[table]:
                    filename="models/"+str(idx)+".txt"
                    filenames.append(filename)
                    self.warehouse.append(self.DBEstClients[table][columnItem])
                    size = size + self.DBEstClients[table][columnItem].get_size()
            start_time=datetime.now()
            for model in self.warehouse:
                self.warehouse_serielize.append(pickle.dumps(model))
            end_time=datetime.now()
            
            for model in self.warehouse_serielize:
                model=pickle.loads(model)
            end_time1=datetime.now()
            t1=(end_time-start_time).total_seconds()
            t2=(end_time1-end_time).total_seconds()
            self.logger.logger.info("size is "+str(size))
            self.logger.logger.info("time to serielize : " +str(t1))
            self.logger.logger.info("time to deserielize : " +str(t2))



            # for disk io 
            start_time=datetime.now()
            for idx,model in enumerate(self.warehouse):
                pickle_out = open(filenames[idx],"wb")
                pickle.dump(model,pickle_out)
                pickle_out.close()
            end_time=datetime.now()
            
            for filename in filenames:
                pickle_in = open(filename,"rb")
                model=pickle.load(pickle_in)
            end_time1=datetime.now()
            t1=(end_time-start_time).total_seconds()
            t2=(end_time1-end_time).total_seconds()
            self.logger.logger.info("size is "+str(size))
            self.logger.logger.info("time to serielize : " +str(t1))
            self.logger.logger.info("time to deserielize : " +str(t2))

            # time_before_write=datetime.now()
            # for idx,model in enumerate(self.warehouse_serielize):
            #     filename=str(idx)+"txt"
            #     with open()
        else:
            start_time=datetime.now()
            str_size=pickle.dumps(self)
            end_time=datetime.now()
            size = sys.getsizeof(str_size)
            model=pickle.loads(str_size)
            end_time1=datetime.now()
            t1=(end_time-start_time).total_seconds()
            t2=(end_time1-end_time).total_seconds()
            self.logger.logger.info("size is "+str(size))
            self.logger.logger.info("time to serielize : " +str(t1))
            self.logger.logger.info("time to deserielize : " +str(t2))           
        return size

    def show_tables(self):
        self.logger.logger.info(
            "DBEst holds " + str(len(self.DBEstClients)) + " models:")
        for model in self.DBEstClients:
            self.logger.logger.info(model)

    def show_clients(self):
        self.logger.logger.info(
            "DBEst holds " + str(len(self.DBEstClients)) + " models:")
        for model in self.DBEstClients:
            self.logger.logger.info(model)
            for client in self.DBEstClients[model]:
                self.logger.logger.info(client)
    def serialize_model(self,model):
        str_=pickle.dumps(model)
        # print(sys.getsizeof(str_))
        return str_
    def deserialize_model(self,model_str):
        return pickle.loads(model_str)

def querySQLgroups(groups,DBEstClient,columnItem,func,x,y,lb,hb,p,results_parallel, index):
    # AQP_results=[]
    # time_costs=[]
    # groups_results=[]
    for idx,grp_name in enumerate(groups):
        columnItem = str([x, y])
        columnItem = str(columnItem) + "-" + str(grp_name)
        result, time=querySQL(DBEstClient,columnItem,func,lb,hb,p)
        # AQP_results.append(result)
        # time_costs.append(time)
        # groups_results.append(grp_name)
        results_parallel[0][index[idx]]=grp_name
        results_parallel[1][index[idx]]=result
        results_parallel[2][index[idx]]=time


    # print(results_parallel)
    return results_parallel[0],results_parallel[1],results_parallel[2]

def querySQL(DBEstClient,columnItem,func,lb=None,hb=None,p=None):
    if func.lower() == "avg":
        # DBEstClient = self.DBEstClients[tbl]
        # print(float(lb))
        # print(float(hb))
        # print("_________")
        result, time = DBEstClient[columnItem].approximate_avg_from_to(
            float(lb), float(hb), 0, epsabs=epsabs, epsrel=epsrel, limit=limit)
        # AQP_results.append(result)
        # time_costs.append(time)
        # groups_results.append(grp_name)
    elif func.lower() == "sum":
        # DBEstClient = self.DBEstClients[tbl]
        result, time = DBEstClient[columnItem].approximate_sum_from_to(
            float(lb), float(hb), 0, epsabs=epsabs, epsrel=epsrel, limit=limit)
        # AQP_results.append(result)
        # time_costs.append(time)
        # groups_results.append(grp_name)
    elif func.lower() == "count":
        # print(float(lb))
        # print(float(hb))
        # print("_________")
        result, time = DBEstClient[columnItem].approximate_count_from_to(
            float(lb), float(hb), 0, epsabs=epsabs, epsrel=epsrel, limit=limit)
        # AQP_results.append(result)
        # time_costs.append(time)
        # groups_results.append(grp_name)
    elif func.lower() == 'variance_x':
        result, time = DBEstClient[columnItem].approximate_variance_x_from_to(
            float(lb), float(hb), 0, epsabs=epsabs, epsrel=epsrel, limit=limit)
        # AQP_results.append(result)
        # time_costs.append(time)
        # groups_results.append(grp_name)
    elif func.lower() == 'min':
        result, time = DBEstClient[columnItem].approximate_min_from_to(
            float(lb), float(hb), 0, ci=ci, confidence=confidence)
        # AQP_results.append(result)
        # time_costs.append(time)
        # groups_results.append(grp_name)
    elif func.lower() == 'max':
        result, time = DBEstClient[columnItem].approximate_max_from_to(
            float(lb), float(hb), 0, ci=ci, confidence=confidence)
        # AQP_results.append(result)
        # time_costs.append(time)
        # groups_results.append(grp_name)
    elif func.lower() == 'covar':
        result, time = DBEstClient[columnItem].approximate_covar_from_to(
            float(lb), float(hb), 0, epsabs=epsabs, epsrel=epsrel, limit=limit)
        # AQP_results.append(result)
        # time_costs.append(time)
        # groups_results.append(grp_name)
    elif func.lower() == 'corr':
        result, time = DBEstClient[columnItem].approximate_corr_from_to(
            float(lb), float(hb), 0)
        # AQP_results.append(result)
        # time_costs.append(time)
        # groups_results.append(grp_name)
    elif func.lower() == 'percentile':
        result, time = DBEstClient[columnItem].approximate_percentile_from_to(
            float(p))
        # AQP_results.append(result)
        # time_costs.append(time)
        # groups_results.append(grp_name)
    else:
        self.logger.logger.warning(
            func + " is currently not supported!")
        return
    return result,time

def querySQLsWithReturnValues(queries,DBEstClients,CSinTable,predictions, times, time):
    start_time=datetime.now()
    predictions, times=querySQLs(queries,DBEstClients,CSinTable)
    end_time=datetime.now()
    times=(end_time-start_time).total_seconds()
    print("______________________________________________________")
    print(times)
    print("______________________________________________________")


def querySQLs(queries,DBEstClients,CSinTable):

    AQP_results=[]
    time_costs=[]
    for line in queries:
        query_list = line.replace(
            "(", " ").replace(")", " ").replace(";", "").replace(",", "")
        # self.logger.logger.info(query_list)
        query_list = re.split('\s+', query_list)
        # remove empty strings caused by sequential blank spaces.
        query_list = list(filter(None, query_list))
        func = query_list[1]

        if func.lower() != "percentile":
            x = query_list[6]
            y = query_list[2]
            lb = query_list[8]
            hb = query_list[10]
            tbl = query_list[4]
            p=None

        else:
            x = query_list[2]
            y = "*"
            lb=None
            hb=None
            p = query_list[3]
            tbl = query_list[5]

        if y == "*":
            columnSetsInTable = CSinTable[tbl]
            for cs in columnSetsInTable:
                if x == cs[0]:
                    print("Find a column \
                        set in table " + tbl + " to replace *: " +
                                             str(cs))
                    y = cs[1]
                    break
        if y == "*":
            print(
                "There is no model to predict percentile!")
            AQP_results.append("Null")
            time_costs.append("Null")
            break

        columnItem = str([x, y])
        DBEstClient = DBEstClients[tbl]

        result,time=querySQL(DBEstClient,columnItem,func,lb,hb,p)
        AQP_results.append(result)
        time_costs.append(time)
    return AQP_results,time_costs


def run_add_pair_client():
    log_file = "../results/DBEsti_tpcds_100k_all.log"
    db = DBEst(dataset="tpcds", logger_file=log_file)
    db.add_pair_client(file='../data/tpcDs10k/store_sales.csv', table_name="store_sales",
        columnPair=["ss_quantity", "ss_ext_sales_price"],num_of_points=2685596178)
    db.add_pair_client(file='../data/tpcDs10k/store_sales.csv', table_name="store_sales",
        columnPair=["ss_quantity", "ss_ext_list_price"],num_of_points=2685596178)
    db.add_pair_client(file='../data/tpcDs10k/store_sales.csv', table_name="store_sales",
        columnPair=["ss_quantity", "ss_ext_sales_price"],num_of_points=2685596178)

    db.del_pair_client(table_name="store_sales",
        columnPair=["ss_quantity", "ss_ext_sales_price"])
    db.add_pair_client(file='../data/tpcDs10k/store_sales.csv', table_name="store_sales",
        columnPair=["ss_quantity", "ss_ext_sales_price"],num_of_points=2685596178)

def test_serialize():
    log_file = "../results/DBEsti_tpcds_100k_all.log"
    db = DBEst(dataset="tpcds", logger_file=log_file)
    model=db.add_pair_client(file='../data/5m.csv', table_name="store_sales",
        columnPair=["ss_list_price","ss_wholesale_cost"],num_of_points=2685596178)
    print(model.cregression.predict([20]))
    model_str=db.serialize_model(model)
    model=db.deserialize_model(model_str)
    print(model.cregression.predict([20]))

def test_batch_query():
    log_file="../results/pm25.log"
    dataset="tpcds"

    filess='../data/pm25/10k.csv'
    db = DBEst(dataset=dataset,logger_file=log_file,
                       base_models=[tools.app_boosting],
                       n_jobs=1)
    db.init_whole_range(file=filess,
                            table="pm_10k",
                            columnItems=[
                                ['TEMP','pm25'],
                                ['DEWP','pm25'],
                                ['PRES','pm25'],
                            ],
                            num_of_points={'pm_10k':'100000000'})#115203420#110022652
    times=[]
    for core in [1,2,4]:#[1,2,3,4,5,6,7,8]:
        db.n_jobs=core
        times.append(db.mass_query_simple_parallel(file="../query/pm25/10k.sql"))
        db.logger.logger.info(str(core))
        db.logger.logger.info("_____________********************_________________________*****************")
    print(core)
    print(times)





    # db.init_whole_range(file='../data/tpcDs10k/store_sales.csv',
    #                     table="store_sales",
    #                     columnItems=[
    #                         ["ss_quantity", "ss_ext_sales_price"],
    #                         ["ss_quantity", "ss_ext_list_price"],
    #                         ["ss_quantity", "ss_ext_tax"],
    #                         ["ss_quantity", "ss_net_paid"],
    #                         ["ss_quantity", "ss_net_paid_inc_tax"],
    #                         ["ss_quantity", "ss_net_profit"],
    #                         ["ss_quantity", "ss_list_price"],
    #                         ["ss_list_price", "ss_list_price"],
    #                         ["ss_coupon_amt", "ss_list_price"],
    #                         ["ss_wholesale_cost", "ss_list_price"],
    #                         ["ss_sales_price", "ss_quantity"],
    #                         ["ss_net_profit", "ss_quantity"],
    #                         ["ss_quantity", "ss_ext_discount_amt"],
    #                     ],
    #                     num_of_points={'store_sales': '2685596178'})



def run_sample_whole_range():

    log_file = "../results/DBEsti_tpcds_100k_all.log"
    db = DBEst(dataset="tpcds", logger_file=log_file)

    table = "store_sales"
    file = '../data/tpcDs10k/store_sales.csv'  # tpcDs100k/store_sales.csv
    num_of_points = {'store_sales': '2685596178'}
    tableColumnSets = [["ss_list_price", "ss_wholesale_cost"]]

    db.init_whole_range(file=file,  # "../data/tpcDs10k/store_sales.csv",    #
                        table=table,
                        columnItems=tableColumnSets,
                        num_of_points=num_of_points)
    db.clear_training_data()

    # table="web_page"
    # file = '../data/tpcDs10k/web_page.csv'
    # num_of_points={'web_page':'3000'}
    # tableColumnSets = [["wp_char_count", "wp_link_count"]]

    # db.init_whole_range(file=file,  # "../data/tpcDs10k/store_sales.csv",    #
    #                 table=table,
    #                 columnItems=tableColumnSets,
    #                 num_of_points=num_of_points)
    # db.clear_training_data()

    db.logger.logger.info("Total size of DBEst is " +
                          str(db.get_size()) + " bytes.")

def get_join_size():
    log_file="../results/tpcds40g_10k_join.log"
    dataset="tpcds"
    db = DBEst(dataset=dataset,logger_file=log_file,
                       base_models=[tools.app_xgboost])

    filess='../data/tpcds40g/join/10k.csv'
    db.init_whole_range(file=filess,
                            table="ss_s_10k",
                            columnItems=[
                                ['s_number_employees','ss_wholesale_cost'],
                                ['s_number_employees','ss_net_profit'],
                            ],
                            num_of_points={'ss_s_10k':'115203420'})#115203420#110022652
    db.mass_query_simple(file="../query/tpcds/hiveql/join/10k.sql")

    db.logger.logger.info("Total size of DBEst is " +
                          str(db.get_size()) + " bytes.")
    db.logger.logger.info("___________________**********************______________________")



    db.init_whole_range(file=filess,
                            table="ss_s_100k",
                            columnItems=[
                                ['s_number_employees','ss_wholesale_cost'],
                                ['s_number_employees','ss_net_profit'],
                            ],
                            num_of_points={'ss_s_100k':'115203420'})#115203420#110022652
    db.mass_query_simple(file="../query/tpcds/hiveql/join/100k.sql")

    db.logger.logger.info("Total size of DBEst is " +
                          str(db.get_size()) + " bytes.")
    db.logger.logger.info("___________________**********************______________________")



    db.init_whole_range(file=filess,
                            table="ss_s_1m",
                            columnItems=[
                                ['s_number_employees','ss_wholesale_cost'],
                                ['s_number_employees','ss_net_profit'],
                            ],
                            num_of_points={'ss_s_1m':'115203420'})#115203420#110022652
    db.mass_query_simple(file="../query/tpcds/hiveql/join/1m.sql")

    db.logger.logger.info("Total size of DBEst is " +
                          str(db.get_size()) + " bytes.")
    db.logger.logger.info("___________________**********************______________________")


def run_sample_group_by():
    log_file = "../results/DBEsti_tpcds_100k_all.log"
    db = DBEst(dataset="tpcds", logger_file=log_file)
    file = "../data/tpcds_1g_100k/ss_100k.csv"
    table = "store_sales"
    group = "ss_store_sk"
    columnItem = ["ss_wholesale_cost", "ss_list_price"]
    num_of_points_per_group = db.read_num_of_points_per_group(
        "../data/tpcds_1g_100k/counts.txt")

    db.init_groupby(file=file,  # "../data/tpcDs10k/store_sales.csv",    #
                    table=table, group=group,
                    columnItem=columnItem,
                    num_of_points_per_group=num_of_points_per_group)
    db.query_simple_groupby(
        query="select sum(ss_list_price) from store_sales where ss_wholesale_cost between 20 and 30  group by ss_store_sk",
        epsabs=10, epsrel=1E-1, limit=20)
    db.logger.logger.info("Total size of DBEst is " +
                          str(db.get_size()) + " bytes.")


def run_tpcds_multi_columns():
    db = DBEst(dataset="tpcds", base_models=[
               tools.app_xgboost, tools.app_boosting])
    db.init_whole_range(file='../data/tpcDs10k/web_page.csv',
                        table="web_page",
                        columnItems=[["wp_char_count", "wp_link_count"]],
                        num_of_points={"web_page": 3000})
    db.init_whole_range(file='../data/tpcDs10k/time_dim.csv',
                        table="time_dim",
                        columnItems=[["t_minute", "t_hour"]],
                        num_of_points={"time_dim": 86400})
    # db.init_whole_range(file='../data/tpcDs10k/web_sales.csv',
    #                     table="web_sales",
    #                     columnItems=[["ws_sales_price", "ws_quantity"]],
    #                     num_of_points={"web_sales": 720000376})
    # db.init_whole_range(file='../data/tpcDs10k/store_sales.csv',
    #                     table="store_sales",
    #                     columnItems=[
    #                         ["ss_quantity", "ss_ext_sales_price"],
    #                         ["ss_quantity", "ss_ext_list_price"],
    #                         ["ss_quantity", "ss_ext_tax"],
    #                         ["ss_quantity", "ss_net_paid"],
    #                         ["ss_quantity", "ss_net_paid_inc_tax"],
    #                         ["ss_quantity", "ss_net_profit"],
    #                         ["ss_quantity", "ss_list_price"],
    #                         ["ss_list_price", "ss_list_price"],
    #                         ["ss_coupon_amt", "ss_list_price"],
    #                         ["ss_wholesale_cost", "ss_list_price"],
    #                         ["ss_sales_price", "ss_quantity"],
    #                         ["ss_net_profit", "ss_quantity"],
    #                         ["ss_quantity", "ss_ext_discount_amt"],
    #                     ],
    #                     num_of_points={'store_sales': '2685596178'})

    print(db.get_size(b_models_only=False))
    # print(db.get_size(b_models_only=False))

def run_powerplant_multi_columns():
    db = DBEst(dataset="pp")
    db.init_whole_range(file='../data/pp10k/powerplant.csv',
                        table="powerplant",
                        columnItems=[
                            ["T", "EP"],
                            ["AP", "EP"],
                            ["RH", "EP"]
                        ],
                        num_of_points={'powerplant': '26000000000'})
    db.clear_training_data()

    print(db.get_size())


def run_8_groupby():
    log_file = "../results/DBEsti_tpcd_groupby_1m_all.log"
    db = DBEst(dataset="tpcds", logger_file=log_file,n_jobs=4)
    file = "../data/tpcds_groupby_few_groups/ss_100k_group.csv"
    table = "store_sales_group_d"
    group = "ss_store_sk"
    columnItem = ["ss_sold_date_sk", "ss_sales_price"]
    num_of_points_per_group = db.read_num_of_points_per_group(
        "../data/tpcds_groupby_few_groups/group_count8.csv")

    db.init_groupby(file=file,  # "../data/tpcDs10k/store_sales.csv",    #
                    table=table, group=group,
                    columnItem=columnItem,
                    num_of_points_per_group=num_of_points_per_group)
    # db.query_simple_groupby(
    #     query="select count(ss_sales_price)   from store_sales_group_d where ss_sold_date_sk between 2451484 and 2451849 group by ss_store_sk",
    #     epsabs=10, epsrel=1E-1, limit=20)
    
    # db.query_simple_groupby(
    #     query="select avg(ss_sales_price)   from store_sales_group_d where ss_sold_date_sk between 2451484 and 2451849 group by ss_store_sk",
    #     epsabs=10, epsrel=1E-1, limit=20)
    db.logger.logger.info("Total size of DBEst is " +
                          str(db.get_size()) + " bytes.")


def run_501_group_by():
    log_file = "../results/DBEsti_tpcd_groupby_1m_all.log"
    db = DBEst(dataset="tpcds", logger_file=log_file)
    file = "../data/tpcds5m/ss_5m.csv"
    table = "store_sales"
    group = "ss_store_sk"
    columnItem = ["ss_sold_date_sk", "ss_sales_price"]
    num_of_points_per_group = db.read_num_of_points_per_group(
        "../data/tpcds5m/num_of_points.csv")

    db.init_groupby(file=file,  # "../data/tpcDs10k/store_sales.csv",    #
                    table=table, group=group,
                    columnItem=columnItem,
                    num_of_points_per_group=num_of_points_per_group)
    # db.query_simple_groupby(
    #     query="select sum(ss_sales_price)   from store_sales_group_d where ss_sold_date_sk between 2451484 and 2451849 group by ss_store_sk",
    #     epsabs=10, epsrel=1E-1,limit=20)
    db.logger.logger.info("Total size of DBEst is " +
                          str(db.get_size()) + " bytes.")

def run_60_groupby():
    file ="../data/tpcds40g/ss_600k.csv"
    table = "store_sales"
    group = "ss_store_sk"
    columnItem=["ss_wholesale_cost", "ss_list_price"]
    log_file="../results/DBEsti_tpcds_groupby_60groups_all.log"
    

    db = DBEst(dataset="tpcds",logger_file=log_file,
                       base_models=[tools.app_xgboost])

    num_of_points_per_group = db.read_num_of_points_per_group(
            "../data/tpcds40g/num_of_points.csv")

    db.init_groupby(file=file,  # "../data/tpcDs10k/store_sales.csv",    #
                    table=table, group=group,
                    columnItem=columnItem,
                    num_of_points_per_group=num_of_points_per_group)
    db.logger.logger.info("Total size of DBEst is " +
                          str(db.get_size()) + " bytes.")
    db.n_jobs=1
    db.query_simple_groupby('select count(ss_list_price)  from store_sales where ss_wholesale_cost between 10     and 11   group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds40g/dbest/count1.txt")
    # db.query_simple_groupby('select count(ss_list_price)  from store_sales where ss_wholesale_cost between 20     and 21   group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds40g/dbest/count2.txt")
    db.query_simple_groupby('select sum(ss_list_price)    from store_sales where ss_wholesale_cost between 10     and 11   group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds40g/dbest/sum1.txt")
    # db.query_simple_groupby('select sum(ss_list_price)    from store_sales where ss_wholesale_cost between 20     and 21   group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds40g/dbest/sum2.txt")
    db.query_simple_groupby('select avg(ss_list_price)    from store_sales where ss_wholesale_cost between 10     and 11   group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds40g/dbest/avg1.txt")
    # db.query_simple_groupby('select avg(ss_list_price)    from store_sales where ss_wholesale_cost between 20     and 21   group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds40g/dbest/avg2.txt")

    db.n_jobs=2
    db.query_simple_groupby('select count(ss_list_price)  from store_sales where ss_wholesale_cost between 10     and 11   group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds40g/dbest/count1.txt")
    # db.query_simple_groupby('select count(ss_list_price)  from store_sales where ss_wholesale_cost between 20     and 21   group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds40g/dbest/count2.txt")
    db.query_simple_groupby('select sum(ss_list_price)    from store_sales where ss_wholesale_cost between 10     and 11   group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds40g/dbest/sum1.txt")
    # db.query_simple_groupby('select sum(ss_list_price)    from store_sales where ss_wholesale_cost between 20     and 21   group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds40g/dbest/sum2.txt")
    db.query_simple_groupby('select avg(ss_list_price)    from store_sales where ss_wholesale_cost between 10     and 11   group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds40g/dbest/avg1.txt")
    # db.query_simple_groupby('select avg(ss_list_price)    from store_sales where ss_wholesale_cost between 20     and 21   group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds40g/dbest/avg2.txt")

    db.n_jobs=4
    db.query_simple_groupby('select count(ss_list_price)  from store_sales where ss_wholesale_cost between 10     and 11   group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds40g/dbest/count1.txt")
    # db.query_simple_groupby('select count(ss_list_price)  from store_sales where ss_wholesale_cost between 20     and 21   group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds40g/dbest/count2.txt")
    db.query_simple_groupby('select sum(ss_list_price)    from store_sales where ss_wholesale_cost between 10     and 11   group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds40g/dbest/sum1.txt")
    # db.query_simple_groupby('select sum(ss_list_price)    from store_sales where ss_wholesale_cost between 20     and 21   group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds40g/dbest/sum2.txt")
    db.query_simple_groupby('select avg(ss_list_price)    from store_sales where ss_wholesale_cost between 10     and 11   group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds40g/dbest/avg1.txt")
    # db.query_simple_groupby('select avg(ss_list_price)    from store_sales where ss_wholesale_cost between 20     and 21   group by   ss_store_sk',epsabs=10, epsrel=1E-1,limit=20,output="../data/tpcds40g/dbest/avg2.txt")


if __name__ == "__main__":
    mp.set_start_method('forkserver')
    run_501_group_by()
    # warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    # run_add_pair_client()
    # test_serialize()
    # run_8_groupby()
    # get_join_size()
    # test_batch_query()
    # run_sample_whole_range()
    # log_file= "../results/DBEsti_tpcds_100k_all.log"
    # db = DBEst(dataset="tpcds",logger_file=log_file)
    # file = "../data/tpcds5m/ss_5m.csv"
    # table = "store_sales"
    # group = "ss_store_sk"
    # columnItem=["ss_sold_date_sk", "ss_sales_price"]
    # num_of_points_per_group = db.read_num_of_points_per_group(
    #     "../data/tpcds5m/num_of_points.csv")

    # db.init_groupby(file=file,  # "../data/tpcDs10k/store_sales.csv",    #
    #                 table=table, group=group,
    #                 columnItem=columnItem,
    #                 num_of_points_per_group=num_of_points_per_group)
    # # db.query_simple_groupby(
    # #     query="select sum(ss_list_price) from store_sales where ss_wholesale_cost between 20 and 30  group by ss_store_sk",
    # #     epsabs=10, epsrel=1E-1,limit=20)
    # db.logger.logger.info("Total size of DBEst is "+str(db.get_size()) +" bytes.")

    # run_sample_whole_range()
    # modify_df("../data/tpcds5m/groupby_test.txt","../data/tpcds5m/new_groupby.txt")

    # run_powerplant_multi_columns()

    # db.mass_query_simple(file="../query/tpcds/qreg/avg.qreg")

    # num_of_points_per_group = db.read_num_of_points_per_group(
    #     "../data/tpcds_1g_100k/counts.txt")
    # db.init_groupby(file="../data/tpcds_1g_100k/ss_100k.csv",  # "../data/tpcDs10k/store_sales.csv",    #
    #                 table="store_sales", group="ss_store_sk",
    #                 columnItem=["ss_wholesale_cost", "ss_list_price"],
    #                 num_of_points_per_group=num_of_points_per_group)
    # db.query_simple_groupby(
    #     query="select sum(ss_list_price) from store_sales where ss_wholesale_cost between 20 and 30  group by ss_store_sk",
    #     epsabs=1E-1, epsrel=1E-3,limit=30)
    # db.logger.logger.info("--------------------------------------------------------------------------------")
    # db.query_simple_groupby(
    #     query="select sum(ss_list_price) from store_sales where ss_wholesale_cost between 20 and 30  group by ss_store_sk",
    #     epsabs=1E-1, epsrel=1E-2,limit=30)
    # db.logger.logger.info("--------------------------------------------------------------------------------")
    # db.query_simple_groupby(
    #     query="select sum(ss_list_price) from store_sales where ss_wholesale_cost between 20 and 30  group by ss_store_sk",
    # epsabs=1E-1, epsrel=1E-1,limit=30)
    # db.logger.logger.info("--------------------------------------------------------------------------------")
    # db.query_simple_groupby(
    #     query="select sum(ss_list_price) from store_sales where ss_wholesale_cost between 20 and 30  group by ss_store_sk",
    #     epsabs=1E-1, epsrel=1E-1,limit=20,output="haha.txt")
    # db.logger.logger.info("--------------------------------------------------------------------------------")
    # db.query_simple_groupby(
    #     query="select sum(ss_list_price) from store_sales where ss_wholesale_cost between 20 and 30  group by ss_store_sk",
    #     epsabs=1E-1, epsrel=1E-1,limit=10)
    # db.mass_query_simple_groupby(
    #     query="select avg(ss_list_price) from store_sales where ss_wholesale_cost between 20 and 30  group by ss_store_sk")
    # db.mass_query_simple_groupby(
    # query="select sum(ss_list_price) from store_sales where
    # ss_wholesale_cost between 20 and 30  group by ss_store_sk")

    # tableColumnSets = [["ss_wholesale_cost","ss_list_price"],["ss_list_price","ss_wholesale_cost"]]
    # log_file= "../results/DBEsti_tpcds_100k_all.log"
    # file = '../data/tpcDs100k/store_sales.csv'
    # num_of_points={'store_sales':'2685596178'}
    # db.init_whole_range(file=file,  # "../data/tpcDs10k/store_sales.csv",    #
    #                     table="store_sales",
    #                     columnItems=tableColumnSets,
    #                     num_of_points=num_of_points)
    # db.clear_training_data()

    # # db.mass_query_simple(file="../query/tpcds/qreg/avg.qreg")

    # client=db.DBEstClients["store_sales"][str(["ss_wholesale_cost","ss_list_price"])]

    # print(client.get_size())
    # print(db.get_size())

    # import dill
    # str1=dill.dumps(client)
    # import sys
    # print(sys.getsizeof(str1))
