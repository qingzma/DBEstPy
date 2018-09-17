#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function, division
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import random

from sklearn import preprocessing

# import pyspark
# from pyspark.sql import SparkSession
# from pyspark.mllib.regression import LabeledPoint

##import findspark
# findspark.init()
import logging

# from vispy.color import ColorArray

# ----------------------------------------------------------------------------------------------------------------#
# r = ColorArray('red')
# g = ColorArray((0, 1, 0, 1))
# blue = ColorArray('blue')
color1 = (0.1, 0.3, 0.1)
color2 = (0.4, 0.4, 0.3)
color3 = (0.6, 0.7, 0.5)
color4 = (0.8, 0.9, 0.7)
color5 = (0.9, 0.1, 0.2)
colors_matploblib = ['g', 'r', 'b', 'c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y',
                     'g', 'r', 'b', 'c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y',
                     'g', 'r', 'b', 'c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y']

markers_matplotlib = ['*', '1', 'v', 'o', 'h', '.', ',', '^', '<', '>', '2', '3', '4', '8', 's', 'p', 'H', '+', 'D',
                      'd', '|', '_',
                      '*', '1', 'v', 'o', 'h', '.', ',', '^', '<', '>', '2', '3', '4', '8', 's', 'p', 'H', '+', 'D',
                      'd', '|', '_', '*', '1', 'v', 'o', 'h', '.', ',', '^', '<', '>', '2', '3', '4', '8', 's', 'p',
                      'H', '+', 'D', 'd', '|', '_']
# ----------------------------------------------------------------------------------------------------------------#
# parameters for the ClientClass
app_linear = "LR"  # "sklearn_linear"
#model_name0 = "sklearn_linear_model"

app_poly = "PR"  # "sklearn_poly"
#model_name1 = "sklearn_poly_model"

app_knn = "sklearn_knn"
#model_name2 = "sklearn_knn_model"

app_rbf = "sklearn_svr_rbf"
#model_name3 = "sklearn_svr_rbf_model"

app_mllib = "mllib_regression"
#model_name4 = "mllib_lrm_SGD_model"

app_gaussian = "sklearn_gaussian_process"
#model_name5 = "sklearn_gaussian_process_model"

app_adaboost = "sklearn_adaboost"
#model_name6 = "sklearn_ensemble_adaboost_model"

app_boosting = "GBoost"  # "sklearn_gradient_tree_boosting"
#model_name7 = "sklearn_ensemble_gradient_tree_boosting_model"

app_decision_tree = "DTR"  # "sklearn_decision_tree"
#model_name8 = "sklearn_decision_tree_model"

app_xgboost = "XGboost"
#model_name9 = "xgboost_model"

app_pwlf = "piecewise linear"

CPM_name = "CRegression"
# base_model_library = ["sklearn_linear", "sklearn_poly", "sklearn_decision_tree", "sklearn_knn",
#                       "sklearn_svr_rbf", "mllib_regression", "sklearn_gaussian_process",
#                       "sklearn_adaboost", "sklearn_gradient_tree_boosting", "xgboost"]
# ensemble_model_library = ["sklearn_adaboost", "sklearn_gradient_tree_boosting", "xgboost"]

classifier_linear_name = 'linear'
classifier_rbf_name = 'rbf'
classifier_xgboost_name = 'xgboost'


# logger_name = '../results/result.log'
# ----------------------------------------------------------------------------------------------------------------#
class CPMstatistics:
    '''Store the final prediction results, the NRMSEs of different models, etc.'''

    def __init__(self, logger_name=None):
        self.file_name = None
        self.s_model_headers = []
        # include training time for both base models and ensemble methods
        self.s_training_time_all_models = []
        # self.model_names_deployed = []
        self.model_names_for_classifier = []
        self.NRMSE = []
        self.NRMSE_training_models = []
        self.NRMSE_training_classifier = []
        self.NRMSE_ideal = None
        self.classifier_selection_names = []
        self.classifier_selection_NRMSEs = []
        self.classifier_name = 'xgboost classifier (default).'
        self.classifier_accuracy = None
        # self.time_training_models = None
        self.time_training_ensemble_models = None
        self.time_training_classifiers = "None! The classifier selection process is not enabled! "
        # time cost of the best classifier
        self.time_training_classifier = "None! The classifier selection process is not enabled! "
        self.time_query_execution_on_classifier = []
        self.time_query_processing_all_models = []
        self.time_average_query_processing_of_all_models = []
        self.num_of_instances_in_testing_dataset = None
        self.num_of_instances = None
        self.time_program = None
        self.logger_name = logger_name
        if logger_name is not None:
            self.logger = logging.getLogger(logger_name)
            # else:
            #     self.logger = logging
            #     # self.logger.basicConfig(level=logging.DEBUG,
            #     #                         format='%(levelname)s - %(message)s')

    def ratio(self):
        index = self.s_model_headers.index(CPM_name)
        ratios = list(self.NRMSE)
        for i in range(len(self.NRMSE)):
            ratios[i] = ratios[i] / self.NRMSE[index]
        return ratios

    def print_summary(self):
        if self.logger_name is not None:
            self.logger.critical("")
            self.logger.critical("")
            self.logger.critical("")
            self.logger.critical(
                '-----------------------------------------------------------------------------------------------------------')
            self.logger.critical(
                "-----------------------------------------------------------------------------------------------------------")
            self.logger.critical(
                "Dataset: " + self.file_name + ", classifier is: " + self.classifier_name)
            self.logger.critical('Calculation Summary:')
            self.logger.critical(
                "-----------------------------------------------------------------------------------------------------------")
            self.logger.critical("Model: " + str(self.s_model_headers))
            self.logger.critical("NRMSE: " + str(self.NRMSE))
            self.logger.critical("Normalised NRMSE: " + str(self.ratio()))
            self.logger.critical(
                "The lower boundary of the NRMSE is: " + str(self.NRMSE_ideal))
            self.logger.critical("")

            self.logger.critical(
                "Time cost (seconds) to train the models: " + str(self.s_training_time_all_models))
            if self.time_training_classifiers != "None! The classifier selection process is not enabled! ":
                self.logger.critical(
                    "Time cost (seconds) to train the classifiers: " + str(self.time_training_classifiers))
                self.logger.critical(
                    "The best classifier is: " + self.classifier_name)
                self.logger.critical(
                    "Time cost (seconds) to train the best classifier is: " + str(self.time_training_classifier))
            else:
                self.logger.critical(
                    "Time cost (seconds) to train the linear classifier is: " + str(self.time_training_classifier))
                self.logger.critical(
                    "Classifier accuracy is " + str(self.classifier_accuracy))
                self.logger.critical("")

            self.logger.critical(
                "Average time cost(ms)  on the classifier is: " + str(self.time_query_execution_on_classifier))
            self.logger.critical(
                "Average time (ms) to process the queries is: " + str(self.time_average_query_processing_of_all_models))
            self.logger.critical(
                "Total time cost(s) to process queries of each model: " + str(self.time_query_processing_all_models))
            self.logger.critical(
                "Number of instances in the testing dataset is " + str(self.num_of_instances_in_testing_dataset))
            self.logger.critical(
                "Number of instances in the dataset is " + str(self.num_of_instances))
            self.logger.critical("")

            self.logger.critical(
                "Program ended! Time cost is " + str(self.time_program) + " s.")
            self.logger.critical(
                "-----------------------------------------------------------------------------------------------------------")
            self.logger.critical(
                "-----------------------------------------------------------------------------------------------------------")
            self.logger.critical("")
            self.logger.critical("")
            self.logger.critical("")
        else:
            print(
                "-----------------------------------------------------------------------------------------------------------")
            print(
                "-----------------------------------------------------------------------------------------------------------")
            print("Dataset: " + self.file_name +
                  ", classifier is: " + self.classifier_name)
            print('Calculation Summary:')
            print(
                "-----------------------------------------------------------------------------------------------------------")
            print("Model: " + str(self.s_model_headers))
            print("NRMSE: " + str(self.NRMSE))
            print("Normalised NRMSE: " + str(self.ratio()))
            print("The lower boundary of the NRMSE is: " + str(self.NRMSE_ideal))
            print()

            print("Time cost (seconds) to train the models: " +
                  str(self.s_training_time_all_models))
            if self.time_training_classifiers != "None! The classifier selection process is not enabled! ":
                print("Time cost (seconds) to train the classifiers: " +
                      str(self.time_training_classifiers))
                print("The best classifier is: " + self.classifier_name)
                print("Time cost (seconds) to train the best classifier is: " +
                      str(self.time_training_classifier))
            else:
                print("Time cost (seconds) to train the linear classifier is: " +
                      str(self.time_training_classifier))
            print("Classifier accuracy is " + str(self.classifier_accuracy))
            print()

            print("Average time cost(ms)  on the classifier is: " +
                  str(self.time_query_execution_on_classifier))
            print(
                "Average time (ms) to process the queries is: " + str(self.time_average_query_processing_of_all_models))
            print("Total time cost(s) to process queries of each model: " +
                  str(self.time_query_processing_all_models))
            print("Number of instances in the testing dataset is " +
                  str(self.num_of_instances_in_testing_dataset))
            print("Number of instances in the dataset is " +
                  str(self.num_of_instances))
            print()

            print("Program ended! Time cost is " +
                  str(self.time_program) + " s.")
            print(
                "-----------------------------------------------------------------------------------------------------------")
            print(
                "-----------------------------------------------------------------------------------------------------------")
        return


# ----------------------------------------------------------------------------------------------------------------#
class PredictionSummary:
    """ store the prediction results and the statistical summary."""

    def __init__(self):
        self.features = []
        self.predictions = []
        self.labels = []
        self.throughput = -1.0
        self.latency = 999999.99
        self.modelID = []
        self.bool_some_predictions_not_return_in_time = False
        self.status = []
        self.num_success = None
        self.num_defaults = None
        self.model_name = ""
        self.headers = None
        self.num_of_instances = None
        self.time_total = None
        self.time_query_execution_on_classifier = None

    def get_vispy_plot_data_2d(self, model_ID=None):
        a = np.array(self.features)
        b = np.array([np.array(self.predictions)])
        if model_ID != None:

            c = []
            for index, ID in enumerate(self.modelID):

                if ID == model_ID:
                    c.append([a[index, 0], b[0, index]])

            return np.array(c)

        c = np.concatenate((a, b.T), axis=1)
        return c

    def get_vispy_plot_data_3d(self, model_ID=None):
        x = np.array(self.features)
        y = np.array([np.array(self.predictions)])
        if model_ID != None:

            c = []
            for index, ID in enumerate(self.modelID):

                if ID == model_ID:
                    c.append([x[index, 0], x[index, 1], y[0, index]])

            return np.array(c)

        c = np.concatenate((x, y.T), axis=1)
        return c

    # Evaluate the model on training data
    def MSE(self, b_exclude_default_prediction=False):
        if len(self.predictions) != len(self.labels):
            print("Error occurs when calculating MSE or RMSE, number mismatch! ")
            sys.exit(0)

        if b_exclude_default_prediction:
            result = 0.0
            for i in range(len(self.labels)):
                if self.status[i] == 1:
                    result += (self.labels[i] - self.predictions[i]) ** 2
            result = result / self.num_success
            return result
        else:
            result = 0.0
            for i in range(len(self.labels)):
                result += (self.labels[i] - self.predictions[i]) ** 2
            result = result / len(self.labels)
            return result

    def RMSE(self, b_exclude_default_prediction=False):
        return np.sqrt(self.MSE(b_exclude_default_prediction))

    def NRMSE(self, b_exclude_default_prediction=False):
        return self.RMSE(b_exclude_default_prediction) / (np.amax(self.labels) - np.amin(self.labels))

    def plot(self):
        plt.plot(self.predictions, "r.", self.labels, "g.")
        plt.xlabel('Query ID')
        plt.ylabel('Predictions')
        plt.title('Real values (green) VS predictions (red) - %s.' %
                  (self.model_name))
        plt.show()

    def predict_precision(self, labs):
        preds = self.modelID
        # labs = self.labels
        if len(preds) != len(labs):
            print("Error! Size mismatch, can not calculate precision.")
            print(len(preds))
            print(len(labs))
            return -1.0
        else:
            size = len(preds)
            count = 0
            for i in range(len(preds)):
                if preds[i] == labs[i]:
                    count = count + 1
            return count * 1.0 / size


# ----------------------------------------------------------------------------------------------------------------#
class DataSource:
    """ This is the data structure of the input data"""

    def __init__(self):
        self.features = []
        self.labels = []
        self.headers = []
        self.file = None

    def __len__(self):
        return len(self.labels)

    # def toRDD(self, spark):
    #     # spark = SparkSession \
    #     #    .builder \
    #     #   .appName("no_meaning_but_a_name") \
    #     #    .getOrCreate()
    #     # print(np.array(self.features))
    #     # print(np.array(self.labels))
    #     data = np.concatenate((np.array(self.features), np.array([self.labels]).T), axis=1)
    #     # print(data)
    #     df = pd.DataFrame(data)
    #     s_df = spark.createDataFrame(df)
    #     train_dataset_RDD = s_df.rdd.map(lambda x: LabeledPoint(x[-1], x[:-1]))
    #     # spark.stop()
    #     return train_dataset_RDD

    def get_before(self, n):
        data = DataSource()
        data.features = self.features[:n]
        data.labels = self.labels[:n]
        data.headers = self.headers
        return data

    def sort1d(self):
        zipped = zip(self.features, self.labels)
        zipped.sort()

        list1 = [i for (i, j) in zipped]
        list2 = [j for (i, j) in zipped]
        self.features = list1
        self.labels = list2

    def scale(self):
        self.features = preprocessing.scale(self.features)

    def replace(self, from_=",", to_=""):
        for element in self.features:
            for ele in element:
                if type(ele) == str:
                    ele.replace(from_, to_)

        for element in self.labels:
            if type(element) == str:
                element.replace(from_, to_)

    def str2float(self):
        results = []
        results_label = []
        width = len(self.features[1, :])
        # print("width = "+str(width))
        for i in range(len(self.labels)):
            result = []
            result_label = []
            for j in range(width):
                # print("i is {0}, j is {1}".format(i,j))
                # print(self.features[i,j])
                # print(type(self.features[i,j]))
                if self.features[i, j] == "?":
                    result = []
                    result_label = []
                    # print("abnormal points detected and deleted!")
                    break

                else:
                    data = np.array(self.features[i, j]).astype(float).tolist()
                    data1 = np.array(self.labels[i])
                    # print(data)
                    result.append(data)
                    result_label = data1

                    # print(result)
            if result != []:
                results.append(result)
                results_label.append(result_label)

        self.features = np.asarray(results)
        self.labels = np.asarray(results_label)
        print(np.asarray(results).shape)
        print("Data Filtered successfully.")

    def remove_repeated_x_1d(self):
        from collections import OrderedDict
        tmp = OrderedDict()

        for point in zip(self.features[:, 0].tolist(), self.labels):
            # print(point)
            tmp.setdefault(point[:1], point)
        mypoints = np.array(tmp.values())
        # print(mypoints)
        # print(len(self.features))
        self.features = mypoints[:, 0].reshape(-1, 1)
        self.labels = mypoints[:, 1]
        # print(len(self.features))
        # exit(1)
        return

    def remove_repeated_x_2d(self):
        from collections import OrderedDict
        tmp = OrderedDict()

        for point in zip(self.features[:, 0].tolist(), self.features[:, 1].tolist(), self.labels):
            # print(point)
            tmp.setdefault(point[:2], point)
        mypoints = np.array(tmp.values())
        # print(mypoints)
        # print(len(self.features))
        self.features = mypoints[:, :2]  # .reshape(-1,1)
        self.labels = mypoints[:, 2]
        # print(len(self.features))
        # exit(1)
        # print(self.features)
        # print(self.labels)
        return

    def filter(self):
        import time
        results = []
        results_label = []
        width = len(self.features[1, :])
        # print("width = "+str(width))
        for i in range(len(self.labels)):
            result = []
            result_label = []
            for j in range(width):
                # print("i is {0}, j is {1}".format(i,j))
                # print(self.features[i,j])
                # print(type(self.features[i,j]))
                if self.features[i, j] == "?":
                    result = []
                    result_label = []
                    # print("abnormal points detected and deleted!")
                    break

                else:
                    data = self.features[i, j]
                    data1 = self.labels[i]
                    # print(data)
                    result.append(data)
                    result_label = data1

                    # print(result)
            if result != []:
                results.append(result)
                results_label.append(result_label)

        self.features = np.asarray(results)
        self.labels = np.asarray(results_label)
        # print(np.asarray(results).shape)
        print("Data Filtered successfully.")
        ts = []
        # tmp=[]
        for dt, tm in zip(self.labels, self.features[:, 0]):
            strs = dt + " " + tm
            tsi = time.mktime(time.strptime(strs, '%d/%m/%Y %H:%M:%S'))
            ts.append(tsi)
            # tmp.append(strs)
        return np.array(ts)  # ,tmp

    def disorder2d(self):
        """
        This function disorder the points, useful for time series data.
        Returns
        -------

        """
        import random
        l = range(len(self.labels))
        random.shuffle(l)
        features = [self.features[i, :] for i in l]
        labels = [self.labels[i] for i in l]
        features = np.asarray(features).reshape(-1, 1)
        labels = np.asarray(labels)  # .reshape(-1, 1)
        self.features = features
        self.labels = labels
        # print(self.features)
        # print(self.labels)
        # exit(1)

    def disorderNd(self):
        """
        This function disorder the points, useful for time series data.
        Returns
        -------

        """
        import random
        l = range(len(self.labels))
        random.shuffle(l)
        features = [self.features[i, :] for i in l]
        labels = [self.labels[i] for i in l]
        features = np.asarray(features)  # .reshape(1, -1)
        labels = np.asarray(labels)  # .reshape(-1, 1)
        self.features = features
        self.labels = labels


class Evaluation:

    def __init__(self, clients, logger_name):
        self.clients = clients
        if logger_name is not None:
            self.logger = logging.getLogger(logger_name)

    def print_evaluation_normalised_NRMSE(self):
        self.logger.critical('')
        self.logger.critical(
            "----------------------------------------------------------------------------------------")
        self.logger.critical("Normalised NRMSE:")
        for client in self.clients:
            self.logger.critical(np.array2string(np.array(client.summary.ratio()), separator=',', formatter={
                                 'float_kind': lambda x: "%.4f" % x}).replace('[', '').replace(']', '') + ', ' + str(client.summary.num_of_instances))

    def print_NRMSE(self):
        self.logger.critical('')
        self.logger.critical(
            "----------------------------------------------------------------------------------------")
        self.logger.critical("NRMSE:")
        for client in self.clients:
            self.logger.critical(np.array2string(np.array(client.summary.NRMSE), separator=',', formatter={'float_kind': lambda x: "%.4f" % x}).replace(
                '[', '').replace(']', '') + ', ' + str(client.summary.NRMSE_ideal) + ', ' + str(client.summary.classifier_accuracy))

    def print_time_train_models(self):
        self.logger.critical('')
        self.logger.critical(
            "----------------------------------------------------------------------------------------")
        self.logger.critical("Time to train base models (s):")
        for client in self.clients:
            self.logger.critical(
                np.array2string(np.array(client.summary.s_training_time_all_models), separator=',', formatter={'float_kind': lambda x: "%.4f" % x}).replace('[', '').replace(']', '') + ', ' + str(client.summary.time_training_classifier))

    def print_query_execution_time(self):
        self.logger.critical('')
        self.logger.critical(
            "----------------------------------------------------------------------------------------")
        self.logger.critical("Average query execution time (ms):")
        for client in self.clients:
            self.logger.critical(
                np.array2string(np.array(client.summary.time_average_query_processing_of_all_models), separator=',', formatter={'float_kind': lambda x: "%.4f" % x}).replace('[', '').replace(']', '') + ', ' + str(client.summary.time_query_execution_on_classifier))

    def print_summary(self):
        self.print_evaluation_normalised_NRMSE()
        self.print_NRMSE()
        self.print_time_train_models()
        self.print_query_execution_time()
        self.logger.critical(
            "----------------------------------------------------------------------------------------")
        self.logger.critical(
            "----------------------------------------------------------------------------------------")


def NRMSE(xs, labels):
    """
        Calculate the NRMSE
    Parameters
    ----------
    xs: floats
        the difference between predictions and labels
    labels: floats
        the labels
    Returns
    -------

    """
    result = 0.0
    for i in range(len(xs)):
        result += (xs[i]) ** 2
    result = result / len(xs)
    result = np.sqrt(result)
    return result / (np.amax(labels) - np.amin(labels))

    # return result


def NRMSE_with_range(errors, ranges):
    if len(errors) == 0:
        return []
    else:
        result = 0.0
        for i in range(len(errors)):
            result += (errors[i]) ** 2
        result = result / len(errors)
        result = np.sqrt(result)
        return result / ranges


def split_data(data_source):
    """ Split the data into 3 sub-datasets: training and testing."""
    training_data_model = DataSource()
    training_data_classifier = DataSource()
    testing_data = DataSource()

    training_data_model.features, training_data_classifier.features, testing_data.features \
        = np.array_split(data_source.features, 3)

    training_data_model.labels, training_data_classifier.labels, testing_data.labels \
        = np.array_split(data_source.labels, 3)

    training_data_model.headers = data_source.headers
    training_data_classifier.headers = data_source.headers
    testing_data.headers = data_source.headers

    if len(training_data_model.features) == len(training_data_model.labels) and \
            len(training_data_classifier.features) == len(training_data_classifier.labels) and \
            len(testing_data.features) == len(testing_data.labels):
        return training_data_model, training_data_classifier, testing_data
    else:
        print("Error splitting the data, split size mismatch! Size: ")
        print(len(training_data_model.features))
        print(len(training_data_model.labels))
        print(len(training_data_classifier.features))
        print(len(training_data_classifier.labels))
        print(len(testing_data.features))
        print(len(testing_data.labels))
        exit(-1)


def split_data_to_2(data_source, weights_=0.5):
    """ Split the data into 2 sub-datasets: training and testing."""
    training_data_model = DataSource()
    training_data_classifier = DataSource()

    cut_index = int(weights_ * len(data_source.features))
    # print("cut_index is: ")
    # print(cut_index)

    training_data_model.features, training_data_classifier.features \
        = np.array_split(data_source.features, [cut_index])

    training_data_model.labels, training_data_classifier.labels \
        = np.array_split(data_source.labels, [cut_index])

    training_data_model.headers = data_source.headers
    training_data_classifier.headers = data_source.headers

    if len(training_data_model.features) == len(training_data_model.labels) and \
            len(training_data_classifier.features) == len(training_data_classifier.labels):
        return training_data_model, training_data_classifier
    else:
        print("Error splitting the data, split size mismatch! Size: ")
        print(len(training_data_model.features))
        print(len(training_data_model.labels))
        print(len(training_data_classifier.features))
        print(len(training_data_classifier.labels))
        exit(1)


def load_csv(filename, fields=None, y_column=None, sep=','):
    """ Read the csv file."""
    input = pd.read_csv(filename, skipinitialspace=True,
                        usecols=fields, sep=sep, low_memory=False)
    # dtype={"ss_list_price": float, "ss_wholesale_cost": float}
    input_data = input.values
    data = DataSource()

    if y_column == None:
        data.features = input_data[:, :-1]
        data.labels = input_data[:, -1]
        data.headers = input.keys()

    else:
        data.features = np.delete(
            input_data, [y_column], axis=1)  # input_data[:, :-1]
        data.labels = input_data[:, y_column]
        headers = np.array(input.keys())
        data.headers = list(np.delete(headers, [y_column]))
        data.headers.append(input.keys()[y_column])
        # print(data.headers)
    try:
        data.file = filename.split("/")[-1]
    except Exception:
        data.file = filename

    return data


# --------------------------------------------------------------------------------------------
""" some usefull utilities"""


def join_vectors(*preds):
    model_num = len(preds)
    row_num = len(preds[0])
    # print(preds[0])
    values = np.ones((row_num, model_num))

    for j in range(model_num):
        for i in range(row_num):
            values[i, j] = preds[j][i]
    return values


def get_values_equal_to_(k, xs, ys):
    X = []
    Y = []
    for x, y in zip(xs, ys):
        if y == k:
            X.append(x)
            Y.append(y)

    if X == []:
        print("No values equals to " + str(k) + " !")
    return X, Y


def get_values_equal_to_3D(k, xs, ys):
    X = []
    Y = []
    Z = []
    for x, target_k in zip(xs, ys):
        if target_k == k:
            X.append(x[0])
            Y.append(x[1])
            Z.append(x[2])

    if X == []:
        print("Warning: No values equals to " + str(k) + " !")
    return X, Y, Z


def float2str(number):
    return str(('%.2f' % ((number) * 100.0))) + '%'


def mix_gaussian(n, k=3, weight=None, b_show_hist=False):
    """
    return n points from a Gaussian mixture model of k Gaussians.

    Parameters
    ----------
    n : int
        the number of points generated
    k : int
        the number of gaussian distributions.
    weight: list[[weight][weight][weight]]
        the weight of each gaussian distribution
    Returns
    ---------
    X : list[float]
        n samples.
    """

    from sklearn.mixture import GMM
    np.random.seed(1)
    gmm = GMM(k)
    means = np.array([[-1 + (i + 1) * 2.0 / (k + 1)] for i in range(k)])
    # print(means)
    gmm.means_ = means

    if weight == None:
        weight = [[1.0 / k] for _ in range(k)]
        # print(weight)
    gmm.weights_ = weight

    cov = np.array(weight) ** 8
    gmm.covars_ = cov

    X = gmm.sample(n)

    if b_show_hist:
        plt.hist(X, 1000)
        plt.show()
    return X


def scaleBack(x, x_min, x_max):
    """
    Scale the x back to  normal

    Parameters
    ----------
    x : float
        x
    x_min : float
        min value of x
    x_max : foat
        max value of x

    Returns
    -------
    output: float
        output X

    """
    return (x_max - x_min) * x + x_min


def scaleBacks(xs, x_min, x_max):
    return np.asarray([scaleBack(x, x_min, x_max) for x in xs])


def scale(xs):
    x_max = max(xs)
    x_min = min(xs)
    return [(element - x_min) / (x_max - x_min) for element in xs]


# -----------------------------------------------------------------------------------------------
# -- shared variables betweeen modules.


def prepare_data8():
    # Number 8 dataset
    # '''
    # load the data

    fields = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
              'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    # fields = ['Date',  'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
    #          'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    # should be the order in the input file, not in the "fields" order.
    y_column = 0
    data = load_csv("../data/8household_power_consumption.csv",
                    fields, y_column, sep=';')

    # print(data.features)
    # print(data.features.shape)
    # print(data.labels)
    # print(data.labels.shape)

    # '''

    # data.str2float()
    ts = data.filter()
    ts = np.array([(element - ts[0]) / 60.0 for element in ts])

    print(ts)

    # print(aa)
    fields = ['Date', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
              'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    # should be the order in the input file, not in the "fields" order.
    y_column = 0
    data1 = load_csv("../data/8household_power_consumption.csv",
                     fields, y_column, sep=';')
    data1.str2float()
    consumptions = np.asarray(data1.features[:, 0]) * (1000.0 / 60.0) - np.asarray(data1.features[:, 4]) - np.array(
        data1.features[:, 5]) - np.array(data1.features[:, 6])
    print(consumptions)

    with open('../data/8data.txt', 'a') as f:
        f.write(
            'timestamp,Global_active_power,Global_reactive_power,Voltage,Global_intensity,Sub_metering_1,Sub_metering_2,Sub_metering_3,energy\n')
        for t1, y0, y1, y2, y3, y4, y5, y6, t2 in zip(ts, data1.features[:, 0], data1.features[:, 1],
                                                      data1.features[:, 2], data1.features[
                                                          :, 3], data1.features[:, 4],
                                                      data1.features[:, 5], data1.features[:, 6], consumptions):
            f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(
                t1, y0, y1, y2, y3, y4, y5, y6, t2))


# http://erikerlandson.github.io/blog/2015/11/20/very-fast-reservoir-sampling/
def fast_reservoir_sampling(file_handle, N=1000000, callable=None):
    sample = []

    if callable is None:
        callable = lambda x: x

    j = N
    for n, line in enumerate(file_handle):
        if n < N:
            sample.append(callable(line))
        else:
            if n < j:
                continue
            p = N / n
            g = np.random.geometric(p)
            j = j + g
            replace = random.randint(0, N - 1)
            sample[replace] = callable(line)

    return sample


# def reservoir_sampling(file_handle, N=1000000, callable=None):
#     sample = []

#     if callable is None:
#         callable = lambda x: x

#     for n, line in enumerate(file_handle):
#         if n < N:
#             sample.append(callable(line))
#         elif n >= N and random.random() < N / float(n + 1):
#             replace = random.randint(0, len(sample) - 1)
#             sample[replace] = callable(line)
#     return sample


def t_distribution(region=0.95):
    return stats.t.ppf(region, 5)


def resorvoir_sampling():
    import subprocess
    import pyarrow as pa
    fs = pa.hdfs.connect("137.205.118.65", 50075, user="hduser")
    subprocess.call(['ls'])

if __name__ == "__main__":
    str = resorvoir_sampling()
    

