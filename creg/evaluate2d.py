#!/usr/bin/env python
# coding=utf-8
from core import CRegression
import data_loader as dl
from query_engine import QueryEngine
import logs
import random
import subprocess
# from pyhive import hive
# import subprocess
import os
import pyhs2
# import MySQLdb
import pymysql
pymysql.install_as_MySQLdb()
import generate_random

from pyhive import hive

from datetime import datetime
import warnings
import sys
import errno

default_mass_query_number = 5
logger_file = "../results/deletable.log"


class Query_Engine_2d:

    def __init__(self, dataID, b_allow_repeated_value=True, logger_file=logger_file,
                 num_of_points=None):
        self.logger = logs.QueryLogs(log=logger_file)
        # self.logger.set_no_output()
        self.data = dl.load2d(dataID)
        if not b_allow_repeated_value:
            self.data.remove_repeated_x_1d()
        self.cRegression = CRegression(logger_object=self.logger)
        self.cRegression.fit(self.data)
        # self.logger.set_logging(file_name=logger_file)
        if num_of_points is None:
            self.qe = QueryEngine(self.cRegression, logger_object=self.logger)
        else:
            self.qe = QueryEngine(
                self.cRegression, logger_object=self.logger,
                num_training_points=num_of_points)
        self.qe.density_estimation()
        self.q_min = min(self.data.features)
        self.q_max = max(self.data.features)
        self.dataID = dataID

        #warnings.filterwarnings(action='ignore', category=DeprecationWarning)

    def query_2d_avg(self, l=0, h=100):
        """query to 2d data sets.

        Args:
            l (int, optional): query lower boundary
            h (int, optional): query higher boundary
        """
        avgs, time = self.qe.approximate_avg_from_to(l, h, 0)  # 0.05E8,0.1E8,
        return avgs, time

    def query_2d_sum(self, l=0, h=100):
        """query to 2d data sets.

        Args:
            l (int, optional): query lower boundary
            h (int, optional): query higher boundary
        """
        sums, time = self.qe.approximate_sum_from_to(l, h, 0)
        return sums, time

    def query_2d_count(self, l=0, h=100):
        count, time = self.qe.approximate_count_from_to(l, h, 0)
        return count, time

    def query_2d_variance_x(self, l=0, h=100):
        variance_x, time = self.qe.approximate_variance_x_from_to(l, h, 0)
        return variance_x, time

    def query_2d_variance_y(self, l=0, h=100):
        variance_y, time = self.qe.approximate_variance_y_from_to(l, h, 0)
        return variance_y, time

    def query_2d_covariance(self, l=0, h=100):
        covariance, time = self.qe.approximate_covar_from_to(l, h, 0)
        return covariance, time

    def query_2d_correlation(self, l=0, h=100):
        correlation, time = self.qe.approximate_corr_from_to(l, h, 0)
        return correlation, time

    def query_2d_percentile(self, p):
        percentile, time = self.qe.approximate_percentile_from_to(
            p, self.q_min, self.q_max)
        return percentile, time

    def query_2d_min(self, l=0, h=100, ci=True, confidence=0.95):
        _min, time = self.qe.approximate_min_from_to(
            l, h, ci=ci, confidence=confidence)
        return _min, time

    def query_2d_max(self, l=0, h=100, ci=True, confidence=0.95):
        _max, time = self.qe.approximate_max_from_to(
            l, h, ci=ci, confidence=confidence)
        return _max, time

    def mass_query_sum(self, table, x="ss_list_price", y="ss_wholesale_cost",
                       percent=5, number=default_mass_query_number, b_random_queries=True):
        q_range_half_length = (self.q_max - self.q_min) * percent / 100.0 / 2.0
        self.logger.logger.info("Start generating queries")
        if b_random_queries:
            random.seed(1.0)
            query_centres = []
            for i in range(number):
                query_centres.append(random.uniform(self.q_min, self.q_max)[0])
        else:
            query_centres = generate_random.make_user_distribution(
                self.qe.kde, self.q_min, self.q_max, n=number)
        self.logger.logger.info(
            "Finish generating " + str(number) + " queries, the center points are:")
        self.logger.logger.info(str(query_centres))
        exact_results = []
        exact_times = []
        approx_results = []
        approx_times = []
        for i in range(number):
            self.logger.logger.info(
                "start query No." + str(i + 1) + " out of " + str(number))
            # random.uniform(self.q_min, self.q_max)
            q_centre = query_centres[i]
            q_left = q_centre - q_range_half_length
            q_right = q_centre + q_range_half_length

            sqlStr = "SELECT SUM(" + y + ") FROM " + str(table) + " WHERE  " + \
                x + " BETWEEN " + str(q_left[0]) + " AND " + str(q_right[0])
            self.logger.logger.info(sqlStr)

            approx_result, approx_time = self.query_2d_sum(l=q_left, h=q_right)
            self.logger.logger.info(approx_result)
            exact_result, exact_time = self.query2mysql(sql=sqlStr)

            self.logger.logger.info(exact_result)

            if (exact_result is not None) and (exact_result is not 0):
                exact_results.append(exact_result)
                exact_times.append(exact_time)
                approx_results.append(approx_result)
                approx_times.append(approx_time)
            else:
                self.logger.logger.warning(
                    "MYSQL returns None, so this record is ignored.")
        self.logger.logger.warning(
            "MYSQL query results: " + str(exact_results))
        self.logger.logger.warning(
            "MYSQL query time cost: " + str(exact_times))
        self.logger.logger.warning(
            "Approximate query results: " + str(approx_results))
        self.logger.logger.warning(
            "Approximate query time cost: " + str(approx_times))
        return exact_results, approx_results, exact_times, approx_times

    def mass_query_avg(self, table, x="ss_list_price", y="ss_wholesale_cost",
                       percent=5, number=default_mass_query_number, b_random_queries=True):
        q_range_half_length = (self.q_max - self.q_min) * percent / 100.0 / 2.0
        self.logger.logger.info("Start generating queries")
        if b_random_queries:
            random.seed(1.0)
            query_centres = []
            for i in range(number):
                query_centres.append(random.uniform(self.q_min, self.q_max)[0])
        else:
            query_centres = generate_random.make_user_distribution(
                self.qe.kde, self.q_min, self.q_max, n=number)
        self.logger.logger.info(
            "Finish generating " + str(number) + " queries, the center points are:")
        self.logger.logger.info(str(query_centres))
        exact_results = []
        exact_times = []
        approx_results = []
        approx_times = []
        for i in range(number):
            self.logger.logger.info(
                "start query No." + str(i + 1) + " out of " + str(number))
            # random.uniform(self.q_min, self.q_max)
            q_centre = query_centres[i]
            q_left = q_centre - q_range_half_length
            q_right = q_centre + q_range_half_length

            sqlStr = "SELECT AVG(" + y + ") FROM " + str(table) + " WHERE  " + \
                x + " BETWEEN " + str(q_left[0]) + " AND " + str(q_right[0])
            self.logger.logger.info(sqlStr)

            approx_result, approx_time = self.query_2d_avg(l=q_left, h=q_right)
            self.logger.logger.info(approx_result)
            exact_result, exact_time = self.query2mysql(sql=sqlStr)

            self.logger.logger.info(exact_result)
            if (exact_result is not None) and (exact_result is not 0):
                exact_results.append(exact_result)
                exact_times.append(exact_time)
                approx_results.append(approx_result)
                approx_times.append(approx_time)
            else:
                self.logger.logger.warning(
                    "MYSQL returns None, so this record is ignored.")
        self.logger.logger.warning(
            "MYSQL query results: " + str(exact_results))
        self.logger.logger.warning(
            "MYSQL query time cost: " + str(exact_times))
        self.logger.logger.warning(
            "Approximate query results: " + str(approx_results))
        self.logger.logger.warning(
            "Approximate query time cost: " + str(approx_times))
        return exact_results, approx_results, exact_times, approx_times

    def mass_query_count(self, table, x="ss_list_price", y="ss_wholesale_cost",
                         percent=5, number=default_mass_query_number, b_random_queries=True):
        q_range_half_length = (self.q_max - self.q_min) * percent / 100.0 / 2.0
        self.logger.logger.info("Start generating queries")
        if b_random_queries:
            random.seed(1.0)
            query_centres = []
            for i in range(number):
                query_centres.append(random.uniform(self.q_min, self.q_max)[0])
        else:
            query_centres = generate_random.make_user_distribution(
                self.qe.kde, self.q_min, self.q_max, n=number)
        self.logger.logger.info(
            "Finish generating " + str(number) + " queries, the center points are:")
        self.logger.logger.info(str(query_centres))
        random_left_boundary = self.q_min + q_range_half_length
        random_right_boundary = self.q_max - q_range_half_length

        exact_results = []
        exact_times = []
        approx_results = []
        approx_times = []
        for i in range(number):
            self.logger.logger.info(
                "start query No." + str(i + 1) + " out of " + str(number))
            # random.uniform(self.q_min, self.q_max)
            q_centre = query_centres[i]
            q_left = q_centre - q_range_half_length
            q_right = q_centre + q_range_half_length

            sqlStr = "SELECT COUNT(" + y + ") FROM " + str(table) + " WHERE  " + \
                x + " BETWEEN " + str(q_left[0]) + " AND " + str(q_right[0])
            self.logger.logger.info(sqlStr)

            approx_result, approx_time = self.query_2d_count(
                l=q_left, h=q_right)
            self.logger.logger.info(approx_result)
            exact_result, exact_time = self.query2mysql(sql=sqlStr)

            self.logger.logger.info(exact_result)
            if (exact_result is not None) and (exact_result is not 0):
                exact_results.append(exact_result)
                exact_times.append(exact_time)
                approx_results.append(approx_result)
                approx_times.append(approx_time)
            else:
                self.logger.logger.warning(
                    "MYSQL returns None, so this record is ignored.")
        self.logger.logger.warning(
            "MYSQL query results: " + str(exact_results))
        self.logger.logger.warning(
            "MYSQL query time cost: " + str(exact_times))
        self.logger.logger.warning(
            "Approximate query results: " + str(approx_results))
        self.logger.logger.warning(
            "Approximate query time cost: " + str(approx_times))
        return exact_results, approx_results, exact_times, approx_times

    def mass_query_variance_x(self, table, x="ss_list_price",
                              y="ss_wholesale_cost", percent=5,
                              number=default_mass_query_number,
                              b_random_queries=True):
        q_range_half_length = (self.q_max - self.q_min) * percent / 100.0 / 2.0
        self.logger.logger.info("Start generating queries")
        if b_random_queries:
            random.seed(1.0)
            query_centres = []
            for i in range(number):
                query_centres.append(random.uniform(self.q_min, self.q_max)[0])
        else:
            query_centres = generate_random.make_user_distribution(
                self.qe.kde, self.q_min, self.q_max, n=number)
        self.logger.logger.info(
            "Finish generating " + str(number) + " queries, the center points are:")
        self.logger.logger.info(str(query_centres))
        random_left_boundary = self.q_min + q_range_half_length
        random_right_boundary = self.q_max - q_range_half_length

        exact_results = []
        exact_times = []
        approx_results = []
        approx_times = []
        for i in range(number):
            self.logger.logger.info(
                "start query No." + str(i + 1) + " out of " + str(number))
            # random.uniform(self.q_min,self.q_max)
            q_centre = query_centres[i]
            q_left = q_centre - q_range_half_length
            q_right = q_centre + q_range_half_length

            sqlStr = "SELECT VARIANCE(" + x + ") FROM " + str(table) + " WHERE  " + \
                x + " BETWEEN " + str(q_left[0]) + " AND " + str(q_right[0])
            self.logger.logger.info(sqlStr)

            approx_result, approx_time = self.query_2d_variance_x(
                l=q_left, h=q_right)
            self.logger.logger.info(approx_result)
            exact_result, exact_time = self.query2mysql(sql=sqlStr)

            self.logger.logger.info(exact_result)
            if (exact_result is not None) and (exact_result is not 0):
                exact_results.append(exact_result)
                exact_times.append(exact_time)
                approx_results.append(approx_result)
                approx_times.append(approx_time)
            else:
                self.logger.logger.warning(
                    "MYSQL returns None, so this record is ignored.")
        self.logger.logger.warning(
            "MYSQL query results: " + str(exact_results))
        self.logger.logger.warning(
            "MYSQL query time cost: " + str(exact_times))
        self.logger.logger.warning(
            "Approximate query results: " + str(approx_results))
        self.logger.logger.warning(
            "Approximate query time cost: " + str(approx_times))
        return exact_results, approx_results, exact_times, approx_times

    def mass_query_variance_y(self, table, x="ss_list_price", y="ss_wholesale_cost",
                              percent=5, number=default_mass_query_number, b_random_queries=True):
        q_range_half_length = (self.q_max - self.q_min) * percent / 100.0 / 2.0
        # random.seed(1.0)
        random_left_boundary = self.q_min + q_range_half_length
        random_right_boundary = self.q_max - q_range_half_length
        self.logger.logger.info("Start generating queries")
        if b_random_queries:
            random.seed(1.0)
            query_centres = []
            for i in range(number):
                query_centres.append(random.uniform(self.q_min, self.q_max)[0])
        else:
            query_centres = generate_random.make_user_distribution(
                self.qe.kde, self.q_min, self.q_max, n=number)
        self.logger.logger.info(
            "Finish generating " + str(number) + " queries, the center points are:")
        self.logger.logger.info(str(query_centres))
        exact_results = []
        exact_times = []
        approx_results = []
        approx_times = []
        for i in range(number):
            self.logger.logger.info(
                "start query No." + str(i + 1) + " out of " + str(number))
            # random.uniform(self.q_min,self.q_max)
            q_centre = query_centres[i]
            q_left = q_centre - q_range_half_length
            q_right = q_centre + q_range_half_length

            sqlStr = "SELECT VARIANCE(" + y + ") FROM " + str(table) + " WHERE  " + \
                x + " BETWEEN " + str(q_left[0]) + " AND " + str(q_right[0])
            self.logger.logger.info(sqlStr)

            approx_result, approx_time = self.query_2d_variance_y(
                l=q_left, h=q_right)
            self.logger.logger.info(approx_result)
            exact_result, exact_time = self.query2mysql(sql=sqlStr)

            self.logger.logger.info(exact_result)
            if (exact_result is not None) and (exact_result is not 0):
                exact_results.append(exact_result)
                exact_times.append(exact_time)
                approx_results.append(approx_result)
                approx_times.append(approx_time)
            else:
                self.logger.logger.warning(
                    "MYSQL returns None, so this record is ignored.")
        self.logger.logger.warning(
            "MYSQL query results: " + str(exact_results))
        self.logger.logger.warning(
            "MYSQL query time cost: " + str(exact_times))
        self.logger.logger.warning(
            "Approximate query results: " + str(approx_results))
        self.logger.logger.warning(
            "Approximate query time cost: " + str(approx_times))
        return exact_results, approx_results, exact_times, approx_times

    def mass_query_covariance(self, table, x="ss_list_price", y="ss_wholesale_cost",
                              percent=5, number=default_mass_query_number, b_random_queries=True):
        q_range_half_length = (self.q_max - self.q_min) * percent / 100.0 / 2.0
        # random.seed(1.0)
        random_left_boundary = self.q_min + q_range_half_length
        random_right_boundary = self.q_max - q_range_half_length
        self.logger.logger.info("Start generating queries")
        if b_random_queries:
            random.seed(1.0)
            query_centres = []
            for i in range(number):
                query_centres.append(random.uniform(self.q_min, self.q_max)[0])
        else:
            query_centres = generate_random.make_user_distribution(
                self.qe.kde, self.q_min, self.q_max, n=number)
        self.logger.logger.info(
            "Finish generating " + str(number) + " queries, the center points are:")
        self.logger.logger.info(str(query_centres))
        exact_results = []
        exact_times = []
        approx_results = []
        approx_times = []
        for i in range(number):
            self.logger.logger.info(
                "start query No." + str(i + 1) + " out of " + str(number))
            # random.uniform(self.q_min,self.q_max)
            q_centre = query_centres[i]
            q_left = q_centre - q_range_half_length
            q_right = q_centre + q_range_half_length

            sqlStr = "SELECT COVARIANCE(" + x + ", " + y + ") FROM " + str(
                table) + " WHERE  " + x + " BETWEEN " + str(q_left[0]) + " AND " + str(
                q_right[0])
            self.logger.logger.info(sqlStr)

            approx_result, approx_time = self.query_2d_covariance(
                l=q_left, h=q_right)
            self.logger.logger.info(approx_result)
            exact_result, exact_time = self.query2mysql(sql=sqlStr)

            self.logger.logger.info(exact_result)
            if (exact_result is not None) and (exact_result is not 0):
                exact_results.append(exact_result)
                exact_times.append(exact_time)
                approx_results.append(approx_result)
                approx_times.append(approx_time)
            else:
                self.logger.logger.warning(
                    "MYSQL returns None, so this record is ignored.")
        self.logger.logger.warning(
            "MYSQL query results: " + str(exact_results))
        self.logger.logger.warning(
            "MYSQL query time cost: " + str(exact_times))
        self.logger.logger.warning(
            "Approximate query results: " + str(approx_results))
        self.logger.logger.warning(
            "Approximate query time cost: " + str(approx_times))
        return exact_results, approx_results, exact_times, approx_times

    def mass_query_correlation(self, table, x="ss_list_price", y="ss_wholesale_cost",
                               percent=5, number=default_mass_query_number,
                               b_random_queries=True):
        q_range_half_length = (self.q_max - self.q_min) * percent / 100.0 / 2.0
        # random.seed(1.0)
        random_left_boundary = self.q_min + q_range_half_length
        random_right_boundary = self.q_max - q_range_half_length
        self.logger.logger.info("Start generating queries")
        if b_random_queries:
            random.seed(1.0)
            query_centres = []
            for i in range(number):
                query_centres.append(random.uniform(self.q_min, self.q_max)[0])
        else:
            query_centres = generate_random.make_user_distribution(
                self.qe.kde, self.q_min, self.q_max, n=number)
        self.logger.logger.info(
            "Finish generating " + str(number) + " queries, the center points are:")
        self.logger.logger.info(str(query_centres))
        exact_results = []
        exact_times = []
        approx_results = []
        approx_times = []
        for i in range(number):
            self.logger.logger.info(
                "start query No." + str(i + 1) + " out of " + str(number))
            # random.uniform(self.q_min,self.q_max)
            q_centre = query_centres[i]
            q_left = q_centre - q_range_half_length
            q_right = q_centre + q_range_half_length

            sqlStr = "SELECT CORR(" + x + ", " + y + ") FROM " + str(
                table) + " WHERE  " + x + " BETWEEN " + str(q_left[0]) + " AND " + str(
                q_right[0])
            self.logger.logger.info(sqlStr)

            approx_result, approx_time = self.query_2d_correlation(
                l=q_left, h=q_right)
            self.logger.logger.info(approx_result)
            exact_result, exact_time = self.query2mysql(sql=sqlStr)

            self.logger.logger.info(exact_result)
            if (exact_result is not None) and (exact_result is not 0):
                exact_results.append(exact_result)
                exact_times.append(exact_time)
                approx_results.append(approx_result)
                approx_times.append(approx_time)
            else:
                self.logger.logger.warning(
                    "MYSQL returns None, so this record is ignored.")
        self.logger.logger.warning(
            "MYSQL query results: " + str(exact_results))
        self.logger.logger.warning(
            "MYSQL query time cost: " + str(exact_times))
        self.logger.logger.warning(
            "Approximate query results: " + str(approx_results))
        self.logger.logger.warning(
            "Approximate query time cost: " + str(approx_times))
        self.logger.logger.info("")
        return exact_results, approx_results, exact_times, approx_times

    def mass_query_percentile(self, table, x="ss_list_price", y="ss_wholesale_cost",
                              percent=5, number=default_mass_query_number,
                              b_random_queries=True):
        q_range_half_length = (self.q_max - self.q_min) * percent / 100.0 / 2.0
        random.seed(1.0)

        exact_results = []
        exact_times = []
        approx_results = []
        approx_times = []
        for i in range(number):
            self.logger.logger.info(
                "start query No." + str(i + 1) + " out of " + str(number))
            q_centre = random.uniform(0, 1)
            sqlStr = "SELECT percentile_cont(" + x + \
                ", " + str(q_centre) + ") FROM " + str(table)
            self.logger.logger.info(sqlStr)

            approx_result, approx_time = self.query_2d_percentile(q_centre)
            self.logger.logger.info(approx_result)

            exact_result, exact_time = self.query2mysql(sql=sqlStr)

            self.logger.logger.info(exact_result)
            if (exact_result is not None) and (exact_result is not 0):
                exact_results.append(exact_result)
                exact_times.append(exact_time)
                approx_results.append(approx_result)
                approx_times.append(approx_time)
            else:
                self.logger.logger.warning(
                    "MYSQL returns None, so this record is ignored.")
        self.logger.logger.warning(
            "MYSQL query results: " + str(exact_results))
        self.logger.logger.warning(
            "MYSQL query time cost: " + str(exact_times))
        self.logger.logger.warning(
            "Approximate query results: " + str(approx_results))
        self.logger.logger.warning(
            "Approximate query time cost: " + str(approx_times))
        self.logger.logger.info("")
        return exact_results, approx_results, exact_times, approx_times

    def mass_query_min(self, table, x="ss_list_price", y="ss_wholesale_cost",
                       percent=5, number=default_mass_query_number, b_random_queries=True):
        q_range_half_length = (self.q_max - self.q_min) * percent / 100.0 / 2.0
        random.seed(1.0)
        exact_results = []
        exact_times = []
        approx_results = []
        approx_times = []
        for i in range(number):
            self.logger.logger.info(
                "start query No." + str(i + 1) + " out of " + str(number))
            q_centre = random.uniform(self.q_min, self.q_max)
            q_left = q_centre - q_range_half_length
            q_right = q_centre + q_range_half_length

            sqlStr = "SELECT MIN(" + y + ") FROM " + str(table) + " WHERE  " + \
                x + " BETWEEN " + str(q_left[0]) + " AND " + str(q_right[0])
            self.logger.logger.info(sqlStr)

            approx_result, approx_time = self.query_2d_min(
                l=q_left[0], h=q_right[0])
            self.logger.logger.info(approx_result)
            exact_result, exact_time = self.query2mysql(sql=sqlStr)

            self.logger.logger.info(exact_result)

            if (exact_result is not None) and (exact_result is not 0):
                exact_results.append(exact_result)
                exact_times.append(exact_time)
                approx_results.append(approx_result)
                approx_times.append(approx_time)
            else:
                self.logger.logger.warning(
                    "MYSQL returns None, so this record is ignored.")
        self.logger.logger.warning(
            "MYSQL query results: " + str(exact_results))
        self.logger.logger.warning(
            "MYSQL query time cost: " + str(exact_times))
        self.logger.logger.warning(
            "Approximate query results: " + str(approx_results))
        self.logger.logger.warning(
            "Approximate query time cost: " + str(approx_times))
        return exact_results, approx_results, exact_times, approx_times

    def mass_query_max(self, table, x="ss_list_price", y="ss_wholesale_cost",
                       percent=5, number=default_mass_query_number, b_random_queries=True):
        q_range_half_length = (self.q_max - self.q_min) * percent / 100.0 / 2.0
        random.seed(1.0)
        exact_results = []
        exact_times = []
        approx_results = []
        approx_times = []
        for i in range(number):
            self.logger.logger.info(
                "start query No." + str(i + 1) + " out of " + str(number))
            q_centre = random.uniform(self.q_min, self.q_max)
            q_left = q_centre - q_range_half_length
            q_right = q_centre + q_range_half_length

            sqlStr = "SELECT MAX(" + y + ") FROM " + str(table) + " WHERE  " + \
                x + " BETWEEN " + str(q_left[0]) + " AND " + str(q_right[0])
            self.logger.logger.info(sqlStr)

            approx_result, approx_time = self.query_2d_max(
                l=q_left[0], h=q_right[0])
            self.logger.logger.info(approx_result)
            exact_result, exact_time = self.query2mysql(sql=sqlStr)

            self.logger.logger.info(exact_result)

            if (exact_result is not None) and (exact_result is not 0):
                exact_results.append(exact_result)
                exact_times.append(exact_time)
                approx_results.append(approx_result)
                approx_times.append(approx_time)
            else:
                self.logger.logger.warning(
                    "MYSQL returns None, so this record is ignored.")
        self.logger.logger.warning(
            "MYSQL query results: " + str(exact_results))
        self.logger.logger.warning(
            "MYSQL query time cost: " + str(exact_times))
        self.logger.logger.warning(
            "Approximate query results: " + str(approx_results))
        self.logger.logger.warning(
            "Approximate query time cost: " + str(approx_times))
        return exact_results, approx_results, exact_times, approx_times

    def relative_error(self, exact_results, approx_results):
        # abs_errors = [abs(i - j) for i, j in zip(exact_results,approx_results) ]
        rel_errors = [abs(i - j) * 1.0 / i for i,
                      j in zip(exact_results, approx_results)]
        # abs_time_reduction = [(j - i) for i, j in zip(exact_times, approx_times) ]
        result = sum(rel_errors) / len(rel_errors)
        self.logger.logger.warning("Relative error is : " + str(result))
        return result

    def time_ratio(self, exact_times, approx_times):
        result = sum(approx_times) / sum(exact_times)
        self.logger.logger.warning("Time ratio is : " + str(result))
        return result

    # def query2hive(self, sql="SHOW TABLES", use_server=True):
    #     if use_server:
    #         host = "137.205.118.65"
    #     else:
    #         host = "localhost"

    #     with pyhs2.connect(host=host,
    #                        port=10000,
    #                        authMechanism="NOSASL",
    #                        user='hiveuser',
    #                        password='bayern',
    #                        database='default') as conn:
    #         with conn.cursor() as cur:
    #             # Show databases
    #             # print cur.getDatabases()

    #             # Execute query
    #             # cur.execute("select * from src")
    #             start = datetime.now()
    #             cur.execute(sql)
    #             end = datetime.now()
    #             time_cost = (end - start).total_seconds()
    #             # Return column info from query
    #             # print cur.getSchema()

    #             # Fetch table results
    #             self.logger.logger.info(
    #                 "Time spent for HIVE query: %.4fs." % time_cost)
    #             for i in cur.fetch():
    #                 self.logger.logger.info(i)
    #     return i[0], time_cost

    def query2hive2(self, sql="SHOW TABLES", use_server=True):
        if use_server:
            host = "137.205.118.65"
        else:
            host = "localhost"

        conn = hive.connect(host=host, port=10000,
                            username='hiveuser', auth='NOSASL')

        cursor = conn.cursor()
        start = datetime.now()
        cursor.execute(sql)
        end = datetime.now()
        time_cost = (end - start).total_seconds()
        for result in cursor.fetchall():
            self.logger.logger.info(result)
        return result[0], time_cost

    def query2mysql(self, sql="SHOW TABLES", use_server=True):
        # Open database connection
        # db = MySQLdb.connect("127.0.0.1","hiveuser","bayern","hivedb" )
        if use_server:
            db = pymysql.connect("137.205.118.65", "u1796377",
                                 "bayern", "hivedb", port=3306)
        else:
            db = pymysql.connect("127.0.0.1", "hiveuser",
                                 "bayern", "hivedb", port=3306)

        # prepare a cursor object using cursor() method
        cursor = db.cursor()

        # Drop table if it already exist using execute() method.
        # cursor.execute("DROP TABLE IF EXISTS EMPLOYEE")

        # Create table as per requirement
        # sql = sql
        start = datetime.now()
        cursor.execute(sql)
        results = cursor.fetchall()

        end = datetime.now()
        time_cost = (end - start).total_seconds()
        self.logger.logger.info(
            "Time spent for MYSQL query: %.4fs." % time_cost)
        for row in results:
            self.logger.logger.info(row)

        # disconnect from server
        db.close()
        return row[0], time_cost

    def mass_query(self, file, agg_func='avg', ci=True, confidence=0.95):
        AQP_results = []
        time_costs = []
        index = 0
        with open(file) as fin:
            for line in fin:
                self.logger.logger.info("Starting Query " + str(index) + ":")
                index = index + 1
                if agg_func is 'percentile':
                    # print(line)
                    result, time = self.query_2d_percentile(float(line))
                    AQP_results.append(result)
                    time_costs.append(time)
                    # print(result)
                if agg_func is 'avg':
                    lh = line.split(",")
                    l = float(lh[0])
                    h = float(lh[1])
                    result, time = self.query_2d_avg(l, h)
                    AQP_results.append(result)
                    time_costs.append(time)
                if agg_func is 'sum':
                    lh = line.split(",")
                    l = float(lh[0])
                    h = float(lh[1])
                    result, time = self.query_2d_sum(l, h)
                    AQP_results.append(result)
                    time_costs.append(time)
                if agg_func is 'count':
                    lh = line.split(",")
                    l = float(lh[0])
                    h = float(lh[1])
                    result, time = self.query_2d_count(l, h)
                    AQP_results.append(result)
                    time_costs.append(time)
                if agg_func is 'variance_x':
                    lh = line.split(",")
                    l = float(lh[0])
                    h = float(lh[1])
                    result, time = self.query_2d_variance_x(l, h)
                    AQP_results.append(result)
                    time_costs.append(time)
                if agg_func is 'min':
                    lh = line.split(",")
                    l = float(lh[0])
                    h = float(lh[1])
                    result, time = self.query_2d_min(
                        l, h, ci=ci, confidence=confidence)
                    AQP_results.append(result)
                    time_costs.append(time)
                if agg_func is 'max':
                    lh = line.split(",")
                    l = float(lh[0])
                    h = float(lh[1])
                    result, time = self.query_2d_max(
                        l, h, ci=ci, confidence=confidence)
                    AQP_results.append(result)
                    time_costs.append(time)
                if agg_func is 'covar':
                    lh = line.split(",")
                    l = float(lh[0])
                    h = float(lh[1])
                    result, time = self.query_2d_covariance(l, h)
                    AQP_results.append(result)
                    time_costs.append(time)
                if agg_func is 'corr':
                    lh = line.split(",")
                    l = float(lh[0])
                    h = float(lh[1])
                    result, time = self.query_2d_correlation(l, h)
                    AQP_results.append(result)
                    time_costs.append(time)

        self.logger.logger.info(AQP_results)
        self.logger.logger.info(time_costs)
        return AQP_results

    def generate_queries(self, table, x="ss_list_price", y="ss_wholesale_cost",
                         percent=5, number=default_mass_query_number,
                         b_random_queries=False,
                         mode=2, file_pre="etrade_"):
        """Summary

        Args:
            table (TYPE): TABLE name
            x (str, optional): x
            y (str, optional): y
            percent (int, optional): query width, or query domain
            number (TYPE, optional): number of queries generated
            b_random_queries (bool, optional): Description
            mode (str, optional): 1 for percentile queries, QRegQL;
                                  2 for other queries, QRegQL;
                                  1 for percentile queries, SQL or HIVEQL;
                                  2 for other queries, SQL or HIVEQL;
        """
        if mode is 1:
            file_name_qreg = file_pre + "percentile" + str(percent) + ".qreg"
            file_name_hiveql = file_pre + \
                "percentile" + str(percent) + ".hiveql"
        elif mode is 2:
            file_name_qreg = file_pre + "queries" + str(percent) + ".qreg"
            file_name_hiveql = file_pre + "queries" + str(percent) + ".hiveql"
        else:
            self.logger.logger.error(
                "Failed to generate queries, no mode selected!")
            sys.exit(errno.EPERM)
        q_range_half_length = (self.q_max - self.q_min) * percent / 100.0 / 2.0
        self.logger.logger.info("Start generating queries")
        if b_random_queries:
            random.seed(1.0)
            query_centres = []
            for i in range(number):
                if mode is 1:
                    # _start = (self.q_max - self.q_min) * 0.4
                    # query_centres.append(random.uniform(
                    #     self.q_min + _start, self.q_max)[0])
                    query_centres.append(random.uniform(
                        0.4, 1))
                else:
                    query_centres.append(
                        random.uniform(self.q_min, self.q_max)[0])
        else:
            query_centres = generate_random.make_user_distribution(
                self.qe.kde, self.q_min, self.q_max, n=number)
        self.logger.logger.info(
            "Finish generating " + str(number) + " queries, the center points are:")
        self.logger.logger.info(str(query_centres))
        exact_results = []
        exact_times = []
        approx_results = []
        approx_times = []

        if mode is 1:
            with open(file_name_hiveql, 'w+') as f_hiveql:
                with open(file_name_qreg, 'w+') as f_qreg:
                    for i in range(number):
                        q_centre = query_centres[i]
                        qregStr = str(q_centre)
                        sqlStr = "SELECT percentile_cont(" + x + \
                            ", " + str(q_centre) + ") FROM " + str(table)
                        f_hiveql.write(sqlStr + "\n")
                        f_qreg.write(qregStr + "\n")
        else:
            with open(file_name_hiveql, 'w+') as f_hiveql:
                with open(file_name_qreg, 'w+') as f_qreg:
                    aggregates = ["COUNT", "SUM",
                                  "AVG", "MIN", "MAX"]
                    for aggregate in aggregates:
                        for i in range(number):
                            q_centre = query_centres[i]
                            q_left = q_centre - q_range_half_length
                            q_right = q_centre + q_range_half_length
                            qregStr = str(q_left[0]) + ", " + str(q_right[0])
                            sqlStr = "SELECT " + aggregate + "(" + y + ") FROM " + str(table) + " WHERE  " + \
                                x + " BETWEEN " + \
                                    str(q_left[0]) + " AND " + str(q_right[0])

                            f_hiveql.write(sqlStr + "\n")
                            if aggregate is "COUNT":
                                f_qreg.write(qregStr + "\n")
                    for i in range(number):
                        q_centre = query_centres[i]
                        q_left = q_centre - q_range_half_length
                        q_right = q_centre + q_range_half_length
                        qregStr = str(q_left[0]) + ", " + str(q_right[0])
                        sqlStr = "SELECT " + "VARIANCE" + "(" + x + ") FROM " + str(table) + " WHERE  " + \
                            x + " BETWEEN " + \
                                str(q_left[0]) + " AND " + str(q_right[0])

                        f_hiveql.write(sqlStr + "\n")

            self.logger.logger.info(
                "Queries are written to file " + file_name_qreg + " and " + file_name_hiveql)

    def mass_query2mysql(self, file):
        AQP_results = []
        time_costs = []
        index = 1
        with open(file) as fin:
            for line in fin:
                self.logger.logger.info("Starting Query " + str(index) + ":")
                index = index + 1
                self.logger.logger.info(line)
                result, time = self.query2mysql(sql=line)
                AQP_results.append(result)
                time_costs.append(time)
        self.logger.logger.info(AQP_results)
        self.logger.logger.info(time_costs)
        return AQP_results

    def mass_query2hive(self, file):
        AQP_results = []
        time_costs = []
        index = 1
        with open(file) as fin:
            for line in fin:
                try:
                    self.logger.logger.info("Starting Query " + str(index) + ":")
                    index = index + 1
                    self.logger.logger.info(line)
                    result, time = self.query2hive2(sql=line)
                    AQP_results.append(result)
                    time_costs.append(time)
                except Exception as e:
                    import smtplib
 
                    server = smtplib.SMTP('smtp.gmail.com', 587)
                    server.starttls()
                    server.login("qingzma@gmail.com", "jszuo1991228")
                     
                    msg = "failed!"
                    server.sendmail("qingzma@gmail.com", "qingzma@gmail.com", msg)
                    server.quit()

                    import sys
                    sys.exit("Error message")

        self.logger.logger.info(AQP_results)
        self.logger.logger.info(time_costs)
        return AQP_results
if __name__ == '__main__':

    qe2d = Query_Engine_2d("10k", num_of_points=10000,
                           logger_file="../results/1m.log")
    qe2d.logger.set_level("DEBUG")
    # print("haha")
    # qe2d.mass_query2hive("../query/hiveql/queries10.hiveql")
    # qe2d.generate_queries(table="price_cost_1t",
    #                       number=100, percent=1, mode=1, b_random_queries=True)
    # qe2d.mass_query(file="avg.hiveql", agg_func="avg")

    # exact_results, approx_results, exact_times, approx_times = qe2d.mass_query_sum()
    #
    # exact_results, approx_results, exact_times, approx_times = qe2d.mass_query_avg(table="price_cost_1t_sorted",number=5,percent=1)
    # qe2d.relative_error(exact_results, approx_results)
    # exact_results, approx_results, exact_times, approx_times = qe2d.mass_query_count(table="price_cost_1t_sorted",number=5,percent=1)
    # qe2d.relative_error(exact_results, approx_results)
    # exact_results, approx_results, exact_times, approx_times = qe2d.mass_query_sum(table="price_cost_1t_sorted",number=5,percent=1)
    # qe2d.relative_error(exact_results, approx_results)
    # exact_results, approx_results, exact_times, approx_times = qe2d.mass_query_variance_x(table="price_cost_100k",number=5,percent=1)
    # exact_results, approx_results, exact_times, approx_times = qe2d.mass_query_variance_y(table="price_cost_100k",number=5,percent=1)
    # exact_results, approx_results, exact_times, approx_times = qe2d.mass_query_covariance(table="price_cost_100k",number=5,percent=1)
    # exact_results, approx_results, exact_times, approx_times = qe2d.mass_query_correlation(table="price_cost_100k",number=5,percent=1)
    # exact_results, approx_results, exact_times, approx_times = qe2d.mass_query_percentile(
    # table="price_cost_100k", number=5, percent=1)
    # qe2d.relative_error(exact_results, approx_results)
    # exact_results, approx_results, exact_times, approx_times = qe2d.mass_query_min(table="price_cost_100k",number=5,percent=1)
    # qe2d.relative_error(exact_results, approx_results)
    # exact_results, approx_results, exact_times, approx_times = qe2d.mass_query_max(table="price_cost_100k",number=5,percent=1)
    # qe2d.relative_error(exact_results, approx_results)
