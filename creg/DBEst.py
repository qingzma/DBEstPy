#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function, division
import numpy as np
import logs

logger_file = "../results/deletable.log"


class DBEst:

    def __init__(self, dataset="tpcds", logger_file=logger_file):
        self.logger = logs.QueryLogs(log=logger_file)
        self.logger.set_level("INFO")
        self.logger.logger.info("Initialising DBEst...")
        if dataset is "tpcds":
            # initialise the column set, tables in TPC-DS dataset.
            tableColumnSets = [
                # ["store_sales", "ss_quantity", "*"],
                ["store_sales", "ss_quantity", "ss_ext_discount_amt"],
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
                ["store_sales", "ss_net_profit", "ss_quantity"],
                # ["web_page", "wp_char_count", "*"],
                # ["time_dim", "t_minute", "*"],
                ["web_sales", "ws_sales_price", "ws_quantity"]]
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
            # load data


if __name__ == "__main__":
    db = DBEst(dataset="tpcds")
