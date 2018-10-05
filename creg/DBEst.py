#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function, division
import logs
import pandas as pd
from tools import DataSource
from core import CRegression
from query_engine import QueryEngine
from datetime import datetime
import re
import sys

logger_file = "../results/deletable.log"


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

    def __init__(self, dataset="tpcds", logger_file=logger_file):
        self.dataset = dataset
        self.logger = logs.QueryLogs(log=logger_file)
        self.logger.set_level("INFO")
        self.logger.logger.info("Initialising DBEst...")
        self.group_names = {}
        self.num_of_points_per_group_tbls = {}
        self.num_of_points = {}
        self.df = {}
        self.DBEstClients = {}
        self.tableColumnSets = []  # store all QeuryEngines, for each table

    def init_whole_range(self, file, tableColumnSets=None, num_of_points=None):
        """Summary

        Args:
            file (TYPE): Description
            tableColumnSets (None, optional): Description
            num_of_points (dict, optional): Description
        """
        start_time = datetime.now()

        if self.dataset is "tpcds":
            # initialise the column set, tables in TPC-DS dataset.
            if num_of_points is None:
                num_of_points = {}
                num_of_points["store_sales"] = 2685596178
                num_of_points["web_page"] = 3000
                num_of_points["time_dim"] = 86400
                num_of_points["web_sales"] = 720000376
            # merge dict num_of_points to self.num_of_points
            self.num_of_points = {**self.num_of_points, **num_of_points}

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

            self.tableColumnSets = self.tableColumnSets+tableColumnSets
        if self.dataset is "pp":
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
            self.tableColumnSets = self.tableColumnSets+tableColumnSets

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

                    cRegression = CRegression(logger_object=self.logger)
                    cRegression.fit(data)

                    qe = QueryEngine(
                        cRegression, logger_object=self.logger,
                        num_training_points=self.num_of_points[uniqueTable])
                    qe.density_estimation()
                    DBEstiClient[str(columnItem)] = qe
                    self.logger.logger.info("Finish training Qeury Engine " +
                                            "for Table " + uniqueTable +
                                            ", Column Set: " + str(columnItem))
                    self.logger.logger.info(
                        "--------------------------------------------------")
                    self.logger.logger.debug(DBEstiClient)
                    self.DBEstClients[uniqueTable] = DBEstiClient
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
        self.logger.logger.info("")
        self.logger.logger.info("Start building GROUP BY for Table " + table)
        self.df[table] = pd.read_csv(file)
        self.df[table] = self.df[table].dropna()
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
                logger_object=self.logger, b_cross_validation=True)
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
                    num_of_points_per_group[str(int(grp_name))]))
            qe.density_estimation()
            self.DBEstClients[table][columnItemGroup] = qe
            self.logger.logger.info(
                "Start building groupy by for " + columnItemGroup)
            self.logger.logger.info(
                "--------------------------------------------------")

            # result, time = self.query_2d_percentile(float(line))

    def mass_query_simple(self, file):
        AQP_results = []
        time_costs = []
        index = 1
        with open(file) as fin:
            for line in fin:
                self.logger.logger.info("Starting Query " + str(index) + ":")
                self.logger.logger.info(line)
                index = index + 1
                query_list = line.replace(
                    "(", " ").replace(")", " ").replace(";", "")
                # self.logger.logger.info(query_list)
                query_list = re.split('\s+', query_list)
                # remove empty strings caused by sequential blank spaces.
                query_list = list(filter(None, query_list))
                func = query_list[1]
                tbl = query_list[4]
                x = query_list[6]
                y = query_list[2]
                lb = query_list[8]
                hb = query_list[10]

                if y == "*":
                    columnSetsInTable = self.CSinTable[tbl]
                    for cs in columnSetsInTable:
                        if x == cs[0]:
                            self.logger.logger.debug("Find a column \
                                set in table " + tbl + " to replace *: " +
                                                     str(cs))
                            y = cs[1]
                            break
                columnItem = str([x, y])
                if func.lower() == "avg":
                    DBEstClient = self.DBEstClients[tbl]
                    result, time = DBEstClient[columnItem].\
                        approximate_avg_from_to(float(lb), float(hb), 0)
                    AQP_results.append(result)
                    time_costs.append(time)
                elif func.lower() == "sum":
                    DBEstClient = self.DBEstClients[tbl]
                    result, time = DBEstClient[columnItem].\
                        approximate_sum_from_to(float(lb), float(hb), 0)
                    AQP_results.append(result)
                    time_costs.append(time)
                elif func.lower() == "count":
                    DBEstClient = self.DBEstClients[tbl]
                    result, time = DBEstClient[columnItem].\
                        approximate_count_from_to(float(lb), float(hb), 0)
                    AQP_results.append(result)
                    time_costs.append(time)
                else:
                    self.logger.logger.warning(
                        func + " is currently not supported!")
            self.logger.logger.info(AQP_results)
            self.logger.logger.info(time_costs)

    def query_simple_groupby(self, query):
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
        tbl = query_list[4]
        x = query_list[6]
        y = query_list[2]
        lb = query_list[8]
        hb = query_list[10]
        grp = query_list[13]

        for grp_name in self.group_names[str([tbl, grp])]:
            columnItem = str([x, y])
            columnItem = str(columnItem) + "-" + str(grp_name)
            if func.lower() == "avg":
                DBEstClient = self.DBEstClients[tbl]
                result, time = DBEstClient[columnItem].approximate_avg_from_to(
                    float(lb), float(hb), 0)
                AQP_results.append(result)
                time_costs.append(time)
                groups_results.append(grp_name)
            elif func.lower() == "sum":
                DBEstClient = self.DBEstClients[tbl]
                result, time = DBEstClient[columnItem].approximate_sum_from_to(
                    float(lb), float(hb), 0)
                AQP_results.append(result)
                time_costs.append(time)
                groups_results.append(grp_name)
            elif func.lower() == "count":
                DBEstClient = self.DBEstClients[tbl]
                result, time = DBEstClient[columnItem].\
                    approximate_count_from_to(float(lb), float(hb), 0)
                AQP_results.append(result)
                time_costs.append(time)
                groups_results.append(grp_name)
            else:
                self.logger.logger.warning(
                    func + " is currently not supported!")
        self.logger.logger.info(groups_results)
        self.logger.logger.info(AQP_results)
        self.logger.logger.info(time_costs)

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


if __name__ == "__main__":
    db = DBEst(dataset="tpcds")
    # db.init_whole_range(file='../data/tpcDs10k/store_sales.csv')
    # db.init_whole_range(file='../data/tpcDs10k/store_sales.csv',
    #                     tableColumnSets=[
    #                         ["store_sales", "ss_quantity", "ss_ext_sales_price"],
    #                         ["store_sales", "ss_quantity", "ss_ext_list_price"],
    #                         ["store_sales", "ss_quantity", "ss_ext_tax"],
    #                         ["store_sales", "ss_quantity", "ss_net_paid"],
    #                         ["store_sales", "ss_quantity", "ss_net_paid_inc_tax"],
    #                         ["store_sales", "ss_quantity", "ss_net_profit"],
    #                         ["store_sales", "ss_quantity", "ss_list_price"],
    #                         ["store_sales", "ss_list_price", "ss_list_price"],
    #                         ["store_sales", "ss_coupon_amt", "ss_list_price"],
    #                         ["store_sales", "ss_wholesale_cost", "ss_list_price"],
    #                         ["store_sales", "ss_sales_price", "ss_quantity"],
    #                         ["store_sales", "ss_net_profit", "ss_quantity"] #,
    #                         #["web_page", "wp_char_count", "wp_link_count"],# *
    #                         #["time_dim", "t_minute", "t_hour"],            # *
    #                         # ["web_sales", "ws_sales_price", "ws_quantity"]
    #                     ],
    #                     num_of_points={"web_page": 2685596178})
    # db.init_whole_range(file='../data/tpcDs10k/web_page.csv',
    #                     tableColumnSets=[
    #                         ["web_page", "wp_char_count", "wp_link_count"]],
    #                     num_of_points={"web_page": 3000})
    # db.init_whole_range(file='../data/tpcDs10k/time_dim.csv',
    #                     tableColumnSets=[["time_dim", "t_minute", "t_hour"]],
    #                     num_of_points={"time_dim": 86400})
    # db.init_whole_range(file='../data/tpcDs10k/web_sales.csv',
    #                     tableColumnSets=[
    #                         ["web_sales", "ws_sales_price", "ws_quantity"]],
    #                     num_of_points={"web_sales": 720000376})

    # db.mass_query_simple(file="../query/tpcds/qreg/avg.qreg")

    num_of_points_per_group = db.read_num_of_points_per_group(
        "../data/tpcds_1g_100k/counts.txt")
    db.init_groupby(file="../data/tpcds_1g_100k/ss_100k.csv",  # "../data/tpcDs10k/store_sales.csv",    #
                    table="store_sales", group="ss_store_sk",
                    columnItem=["ss_wholesale_cost", "ss_list_price"],
                    num_of_points_per_group=num_of_points_per_group)

    db.query_simple_groupby(
        query="select count(ss_list_price) from store_sales where ss_wholesale_cost between 20 and 30  group by ss_store_sk")
    # db.mass_query_simple_groupby(
    #     query="select avg(ss_list_price) from store_sales where ss_wholesale_cost between 20 and 30  group by ss_store_sk")
    # db.mass_query_simple_groupby(
    #     query="select sum(ss_list_price) from store_sales where ss_wholesale_cost between 20 and 30  group by ss_store_sk")
