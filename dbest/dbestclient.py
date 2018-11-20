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

logger_file = "../results/deletable.log"


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

    def __init__(self, dataset="tpcds", logger_file=logger_file, base_models=[tools.app_xgboost]):
        self.dataset = dataset
        self.logger = logs.QueryLogs(log=logger_file)
        self.logger.set_level("INFO")
        self.logger.logger.info("Initialising DBEst...")
        self.group_names = {}
        self.num_of_points_per_group_tbls = {}
        self.num_of_points = {}
        self.df = {}
        self.DBEstClients = {}
        self.modelCatalog = {}
        self.tableColumnSets = []  # store all QeuryEngines, for each table
        self.base_models = base_models
        # warnings.filterwarnings(action='ignore', category=DeprecationWarning)

    def generate_model_catalog_string(self, client, table_name, columnPairs, groupbyID=None, groupby_value=None):
        """Generate the string holding the model catalog information, each model has a unique model catalog

        Args:
            client (TYPE):  DBEst client, consists of a regrssion model and a density estimator
            table_name (String): the table name
            columnPairs (List): the coolumn pair for a query
            groupbyID (None, String): groupby attribute
            groupby_value (None, String): groupby value, for example, group by month, you should input 'January', etc.
        Returns:
            String: the model catalog
        """
        client_identifier = str(columnPairs)
        if groupbyID != None:
            # Check if the groupby_value is provided.
            if groupby_value == None:
                self.logger.logger.warning("Please provide the groupby_value!")
                return
            client_identifier += "-groupby-"
            client_identifier += str([groupbyID, groupby_value])
        return client_identifier

    def check_and_add_client(self, client, table_name, columnPairs, groupbyID=None, groupby_value=None):
        """Summary

        Args:
            client (TYPE): DBEst client, consists of a regrssion model and a density estimator
            table_name (String): the table name
            columnPairs (List): the coolumn pair for a query
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
            client, table_name, columnPairs, groupbyID, groupby_value)
        if client_identifier not in self.DBEstClients:
            self.DBEstClients[table_name][client_identifier]=client
        else:
            self.logger.logger.info(client_identifier+" exists in DBEst, so ignore adding it.")

    def del_client(self,client, table_name, columnPairs, groupbyID=None, groupby_value=None):
        """ delete a DBEst client
        
        Args:
            client (TYPE): DBEst client, consists of a regrssion model and a density estimator
            table_name (String): the table name
            columnPairs (List): the coolumn pair for a query
            groupbyID (None, String): groupby attribute
            groupby_value (None, String): groupby value, for example, group by month, you should input 'January', etc.
        """
        client_identifier = self.generate_model_catalog_string(
            client, table_name, columnPairs, groupbyID, groupby_value)
        if client_identifier in self.DBEstClients:
            del self.DBEstClients[client_identifier]
        else:
            self.logger.logger.info("No such model in DBEst, so could not operate the deletion operation.")

    def init_whole_range(self, file, table, columnItems, num_of_points=None):
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

        if not hasattr(self, 'df'):
            self.df = {}

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
            self.logger.logger.info(grp_name)
            # self.logger.logger.info(group)
            qe = QueryEngine(
                cRegression, logger_object=self.logger,
                num_training_points=int(
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

                else:
                    x = query_list[2]
                    y = "*"
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
                if func.lower() == "avg":

                    result, time = DBEstClient[columnItem].\
                        approximate_avg_from_to(float(lb), float(
                            hb), 0, epsabs=epsabs, epsrel=epsrel, limit=limit)
                    AQP_results.append(result)
                    time_costs.append(time)
                elif func.lower() == "sum":
                    # DBEstClient = self.DBEstClients[tbl]
                    result, time = DBEstClient[columnItem].\
                        approximate_sum_from_to(float(lb), float(
                            hb), 0, epsabs=epsabs, epsrel=epsrel, limit=limit)
                    AQP_results.append(result)
                    time_costs.append(time)
                elif func.lower() == "count":
                    # DBEstClient = self.DBEstClients[tbl]
                    # self.logger.logger.info("table "+ str(tbl))
                    # self.logger.logger.info("lb "+ str(lb))
                    # self.logger.logger.info("hb "+ str(hb))
                    result, time = DBEstClient[columnItem].\
                        approximate_count_from_to(float(lb), float(
                            hb), 0, epsabs=epsabs, epsrel=epsrel, limit=limit)
                    AQP_results.append(result)
                    time_costs.append(time)
                elif func.lower() == 'variance_x':
                    result, time = DBEstClient[columnItem].approximate_variance_x_from_to(
                        float(lb), float(hb), 0, epsabs=epsabs, epsrel=epsrel, limit=limit)
                    AQP_results.append(result)
                    time_costs.append(time)

                elif func.lower() == 'min':
                    result, time = DBEstClient[columnItem].approximate_min_from_to(
                        float(lb), float(hb), 0, ci=ci, confidence=confidence)
                    AQP_results.append(result)
                    time_costs.append(time)

                elif func.lower() == 'max':
                    result, time = DBEstClient[columnItem].approximate_max_from_to(
                        float(lb), float(hb), 0, ci=ci, confidence=confidence)
                    AQP_results.append(result)
                    time_costs.append(time)

                elif func.lower() == 'covar':
                    result, time = DBEstClient[columnItem].approximate_covar_from_to(
                        float(lb), float(hb), 0, epsabs=epsabs, epsrel=epsrel, limit=limit)
                    AQP_results.append(result)
                    time_costs.append(time)

                elif func.lower() == 'corr':
                    result, time = DBEstClient[columnItem].approximate_corr_from_to(
                        float(lb), float(hb), 0)
                    AQP_results.append(result)
                    time_costs.append(time)
                elif func.lower() == 'percentile':
                    result, time = DBEstClient[columnItem].approximate_percentile_from_to(
                        float(p))
                    AQP_results.append(result)
                    time_costs.append(time)

                else:
                    self.logger.logger.warning(
                        func + " is currently not supported!")
            self.logger.logger.info(AQP_results)
            self.logger.logger.info(time_costs)

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
        else:
            x = query_list[2]
            y = "*"
            p = query_list[3]
            tbl = query_list[5]

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

            for grp_name in self.group_names[str([tbl, grp])]:
                columnItem = str([x, y])
                columnItem = str(columnItem) + "-" + str(grp_name)
                DBEstClient = self.DBEstClients[tbl]
                if func.lower() == "avg":
                    # DBEstClient = self.DBEstClients[tbl]
                    result, time = DBEstClient[columnItem].approximate_avg_from_to(
                        float(lb), float(hb), 0, epsabs=epsabs, epsrel=epsrel, limit=limit)
                    AQP_results.append(result)
                    time_costs.append(time)
                    groups_results.append(grp_name)
                elif func.lower() == "sum":
                    # DBEstClient = self.DBEstClients[tbl]
                    result, time = DBEstClient[columnItem].approximate_sum_from_to(
                        float(lb), float(hb), 0, epsabs=epsabs, epsrel=epsrel, limit=limit)
                    AQP_results.append(result)
                    time_costs.append(time)
                    groups_results.append(grp_name)
                elif func.lower() == "count":
                    result, time = DBEstClient[columnItem].approximate_count_from_to(
                        float(lb), float(hb), 0, epsabs=epsabs, epsrel=epsrel, limit=limit)
                    AQP_results.append(result)
                    time_costs.append(time)
                    groups_results.append(grp_name)
                elif func.lower() == 'variance_x':
                    result, time = DBEstClient[columnItem].approximate_variance_x_from_to(
                        float(lb), float(hb), 0, epsabs=epsabs, epsrel=epsrel, limit=limit)
                    AQP_results.append(result)
                    time_costs.append(time)
                    groups_results.append(grp_name)
                elif func.lower() == 'min':
                    result, time = DBEstClient[columnItem].approximate_min_from_to(
                        float(lb), float(hb), 0, ci=ci, confidence=confidence)
                    AQP_results.append(result)
                    time_costs.append(time)
                    groups_results.append(grp_name)
                elif func.lower() == 'max':
                    result, time = DBEstClient[columnItem].approximate_max_from_to(
                        float(lb), float(hb), 0, ci=ci, confidence=confidence)
                    AQP_results.append(result)
                    time_costs.append(time)
                    groups_results.append(grp_name)
                elif func.lower() == 'covar':
                    result, time = DBEstClient[columnItem].approximate_covar_from_to(
                        float(lb), float(hb), 0, epsabs=epsabs, epsrel=epsrel, limit=limit)
                    AQP_results.append(result)
                    time_costs.append(time)
                    groups_results.append(grp_name)
                elif func.lower() == 'corr':
                    result, time = DBEstClient[columnItem].approximate_corr_from_to(
                        float(lb), float(hb), 0)
                    AQP_results.append(result)
                    time_costs.append(time)
                    groups_results.append(grp_name)
                elif func.lower() == 'percentile':
                    result, time = DBEstClient[columnItem].approximate_percentile_from_to(
                        float(p))
                    AQP_results.append(result)
                    time_costs.append(time)
                    groups_results.append(grp_name)
                else:
                    self.logger.logger.warning(
                        func + " is currently not supported!")
            self.logger.logger.info(groups_results)
            self.logger.logger.info(AQP_results)
            self.logger.logger.info(time_costs)
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
            for table in self.DBEstClients:
                for columnItem in self.DBEstClients[table]:
                    size = size + self.DBEstClients[table][columnItem].get_size()
        else:
            str_size=pickle.dumps(self)
            size = sys.getsizeof(str_size)            
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


def run_sample_whole_range():

    log_file = "../results/DBEsti_tpcds_100k_all.log"
    db = DBEst(dataset="tpcds", logger_file=log_file)

    table = "store_sales"
    file = '../data/1_percent.csv'  # tpcDs100k/store_sales.csv
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


def run_8_group_by():
    log_file = "../results/DBEsti_tpcd_groupby_1m_all.log"
    db = DBEst(dataset="tpcds", logger_file=log_file)
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
    db.query_simple_groupby(
        query="select sum(ss_sales_price)   from store_sales_group_d where ss_sold_date_sk between 2451484 and 2451849 group by ss_store_sk",
        epsabs=10, epsrel=1E-1, limit=20)
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


if __name__ == "__main__":
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    run_tpcds_multi_columns()
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
