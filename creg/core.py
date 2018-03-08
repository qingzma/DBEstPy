#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function, division
import tools

# import random
# import socket
import time
import numpy as np
import os
import sys

# Path for spark source folder
#os.environ['SPARK_HOME'] = "/home/u1796377/Program/spark-2.1.0-bin-hadoop2.7"

# Append pyspark  to Python Path
sys.path.append("/home/u1796377/Program/spark-2.1.0-bin-hadoop2.7")
# import requests
# import json
from collections import Counter
from sklearn import svm
from sklearn.svm import SVC

#import findspark

# findspark.init()
#import pyspark

import subprocess32 as subprocess
import pprint

# from pyspark.mllib.classification import LogisticRegressionWithSGD
# from pyspark.mllib.classification import SVMWithSGD
# from pyspark.mllib.regression import LinearRegressionWithSGD
# from pyspark.mllib.tree import RandomForest
# from pyspark.mllib.regression import LabeledPoint
# from pyspark.ml.regression import GeneralizedLinearRegression
# from pyspark.sql import SparkSession

from datetime import datetime
from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline

# from clipper_admin import Clipper
# import ClassifierClient


# package for different classifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import vispy.plot as vp
from vispy.color import ColorArray
from mpl_toolkits.mplot3d import axes3d
from matplotlib import gridspec

from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging


r = ColorArray('red')
g = ColorArray((0, 1, 0, 1))
blue = ColorArray('blue')
color1 = (0.1, 0.3, 0.1)
color2 = (0.4, 0.4, 0.3)
color3 = (0.6, 0.7, 0.5)
color4 = (0.8, 0.9, 0.7)
color5 = (0.9, 0.1, 0.2)
colors_matploblib = ['b', 'c', 'y', 'm', 'r', 'g']
markers_matplotlib = ['*', '1', 'v', 'o', 'h', 'x']


class CRegression:
    def __init__(self, logger_object=None, base_models=None, ensemble_models=None,
                 classifier_type=tools.classifier_xgboost_name,
                 b_show_plot=False, b_disorder=False, b_select_classifier=False):
        """

        Parameters
        ----------
        logger_name : String
            The name of the logger. If not provided, default logging is used, otherwise logging will be dealt with  the provided logger.
        base_models : list(String)
            The names of the base models to be used. Should be among
                    "sklearn_linear","sklearn_poly","sklearn_decision_tree","sklearn_knn",
                    "sklearn_svr_rbf","mllib_regression","sklearn_gaussian_process",
                    "sklearn_adaboost","sklearn_gradient_tree_boosting","xgboost"
        ensemble_models : list(String)
            The names of the ensemble methods to be compared with, should be among
                    "sklearn_adaboost","sklearn_gradient_tree_boosting","xgboost"
        """
        self.app_names_deployed = []

        self.apps_deployed = []
        index_of_models_in_classifier = []

        # bool_time_term_in_classifier = False
        # self.num_model_in_classifier = 10
        # self.app_names_for_classifier = None
        # self.apps_for_classifier = None
        self.ensemble_method_names = []
        self.ensemble_models_deployed = []
        self.classifier = None
        self.time_cost_to_train_base_models = []
        self.time_cost_to_train_ensemble_models = []

        self.predictions_testing = None

        if not logger_object:
            import logs
            logger_object = logs.QueryLogs()
        self.logger = logger_object.logger
        self.logger_name = logger_object.logger_name

        self.input_base_models = base_models
        self.input_ensemble_models = ensemble_models

        self.classifier_type = classifier_type
        self.b_show_plot = b_show_plot

        self.b_disorder = b_disorder
        self.b_select_classifier = b_select_classifier
        self.num_total_training_points = None  # the number of training points, used for density estimation.
        self.training_data = None
        self.summary = tools.CPMstatistics(logger_name=self.logger_name)

        # logging.basicConfig(level=logging.ERROR)

    def _test_deployed_model(self, model, training_data):
        '''
        response = requests.post(
            "http://localhost:1337/%s/predict" % app,
            headers=headers,
            data=json.dumps({
                'input': list(get_test_point(training_data))
            }))
        result = response.json()
        '''
        result = model.predict(training_data)
        print(result)

    def get_test_point(self, training_data):
        # print(list(training_data_model.features[1]))
        # print(training_data)
        # print(training_data.features)
        # print(training_data.features[1])
        # global sklearn_poly_model
        # sklearn_poly_model = train_sklearn_poly_regression(training_data)
        # print(sklearn_poly_predict_fn(training_data.features[1]))
        # print(list(training_data[1].features))
        return training_data.features[1]  # [13.0,1073.0, 0.663]

    def get_predictions(self, app, xs):
        try:
            self.logger.info("Start querying to %s." % (self.app_names_deployed[self.apps_deployed.index(app)]))
        except ValueError:
            self.logger.info(
                "Start querying to %s." % (self.ensemble_method_names[self.ensemble_models_deployed.index(app)]))
        answer = tools.PredictionSummary()
        # xs = [[1.0 2.0],[2.0 3.0]]
        num_defaults = 0
        num_success = len(xs.features)
        results = []
        # print(len(xs))
        start = datetime.now()
        for element in xs.features:
            # print(element)

            # print(element)
            results.append(app.predict([element])[0])
            answer.status.append(1)

        end = datetime.now()
        latency = (end - start).total_seconds() * 1000.0 / len(xs)
        # throughput = 1000 / latency
        self.logger.debug("Finish %d queries, average latency is %f ms. " % (len(xs), latency))
        if num_defaults > 0:
            self.logger.warning("Warning: %d of %d quries returns the default value -1." % (num_defaults, len(xs)))

        self.logger.info("Total time spent: %.4f s." % (end - start).total_seconds())
        self.logger.info("--------------------------------------------------------------------------------------------")

        answer.predictions = results
        answer.latency = latency
        # answer.throughput = throughput
        answer.labels = xs.labels
        answer.num_defaults = num_defaults
        answer.num_success = num_success
        try:
            answer.model_name = self.app_names_deployed[self.apps_deployed.index(app)]
        except ValueError:
            answer.model_name = self.ensemble_method_names[self.ensemble_models_deployed.index(app)]
        answer.features = xs.features
        answer.headers = xs.headers
        answer.time_total = (end - start).total_seconds()
        answer.num_of_instances = len(xs)

        return answer

    def get_predictions_from_models_for_testing(self, training_data):
        answers = []

        # print(training_data_classifier.__len__)
        for i in range(len(self.apps_for_classifier)):
            app_i = self.apps_for_classifier[i]
            answer_i = self.get_predictions(app_i, training_data)
            answers.append(answer_i)

        return answers

    def get_predictions_from_models(self, models, training_data):
        answers = []

        for i in range(len(models)):
            app_i = models[i]
            answer_i = self.get_predictions(app_i, training_data)
            answers.append(answer_i)

        return answers

    def get_classified_predictions(self, classifier, xs):

        self.logger.info("Start querying to Classified Prediction System.")
        answer = tools.PredictionSummary()
        # xs = [[1.0 2.0],[2.0 3.0]]
        num_defaults = 0
        num_success = len(xs)
        results = []
        time_classifier = []
        # print(len(xs))
        start = datetime.now()
        for element in xs.features:
            # print(element)

            start_i = datetime.now()
            model_number = classifier.predict(element.reshape(1, -1))
            end_i = datetime.now()
            time_classifier.append((end_i - start_i).total_seconds() * 1000.0)
            answer.modelID.append(model_number[0])

            # response = requests.post(
            #    "http://localhost:1337/%s/predict" % ClassifierClient.app_names_for_classifier[model_number[0]],
            #    headers=headers,
            #    data=json.dumps({
            #        'input': list(element)
            #    }))
            # result = response.json()

            # results.append(result["output"])
            answer.status.append(1)
            # print(model_number[0])
            # print(len(apps_deployed))
            # print(element)
            # print(list(element))
            # print(np.array(list(element)).reshape(1,-1))

            # print()
            value_tmp = self.apps_deployed[model_number[0]].predict(np.array(list(element)).reshape(1, -1))
            value = value_tmp[0]
            results.append(value)

        end = datetime.now()
        latency = (end - start).total_seconds() * 1000.0 / len(xs)
        # throughput = 1000 / latency

        answer.predictions = results
        answer.latency = latency
        # answer.throughput = throughput
        answer.labels = xs.labels
        answer.num_success = num_success
        answer.num_defaults = num_defaults
        answer.model_name = "classified model"
        answer.features = xs.features
        answer.headers = xs.headers
        answer.time_total = (end - start).total_seconds()
        answer.time_query_execution_on_classifier = (sum(time_classifier) / float(len(time_classifier)))
        answer.num_of_instances = len(time_classifier)
        # print(answer.modelID)

        # print statistics for the queries
        model_counts = []
        for i in range(self.num_model_in_classifier):
            model_counts.append(answer.modelID.count(i))
        model_counts_str = np.array_str(np.asarray(model_counts))

        self.logger.info("Queries are classified into %d categories:  " % (self.num_model_in_classifier))
        self.logger.info("Counts are: %s." % (model_counts_str))

        self.logger.debug("Finish %d queries, average latency is %f ms. " % (len(xs), latency))
        self.logger.debug(
            "Average time spent on the classifier is %f ms." % (sum(time_classifier) / float(len(time_classifier))))

        if num_defaults > 0:
            self.logger.warning("Warning: %d of %d quries returns the default value -1." % (num_defaults, len(xs)))

        self.logger.debug("Total time spent: %.2f s." % (end - start).total_seconds())
        self.logger.info("--------------------------------------------------------------------------------------------")

        return answer

    def get_classified_prediction(self, classifier, x):

        X = [x]
        model_number = classifier.predict(X)
        return self.apps_deployed[model_number[0]].predict(np.array(x).reshape(1, -1))[0]

    # # parse the data in a line
    # def parsePoint(self, line):
    #     values = [float(x) for x in line]
    #     return LabeledPoint(values[3], values[0:3])

    # return the traing data and tesing data, RDD values
    def load_data(self, sc):
        data = sc.textFile("OnlineNewsPopularity.csv")
        filteredData = data.map(lambda x: x.replace(',', ' ')).map(lambda x: x.split()).map(
            lambda x: (x[2], x[3], x[4], x[6]))
        parsedData = filteredData.map(self.parsePoint)
        query_training_data, trainingData, testingData = parsedData.randomSplit([0.3, 0.3, 0.4])
        return query_training_data, trainingData, testingData

    # -------------------------------------------------------------------------------------------------
    # def train_mllib_linear_regression_withSGD(self, trainingDataRDD):
    #     return LinearRegressionWithSGD.train(trainingDataRDD, iterations=500, step=0.0000000000000001,
    #                                          convergenceTol=0.0001, intercept=True)  # ,initialWeights=np.array[1.0])

    def deploy_model_sklearn_linear_regression(self, training_data):
        def train_sklearn_linear_regression(trainingData):
            start = datetime.now()
            X = trainingData.features
            y = trainingData.labels
            reg = linear_model.LinearRegression()
            reg.fit(X, y)
            end = datetime.now()
            time_train = (end - start).total_seconds()
            self.logger.debug("Sucessfully deployed " + tools.app_linear)
            self.logger.debug("Time cost to train the model is : %.5f s." % time_train)

            return reg, time_train

        def sklearn_lr_predict_fn(inputs):
            return sklearn_linear_model.predict(inputs)

        sklearn_linear_model, time_train = train_sklearn_linear_regression(training_data)
        return sklearn_linear_model, tools.app_linear, time_train

    def deploy_model_sklearn_poly_regression(self, training_data):
        def train_sklearn_poly_regression(trainingData):
            start = datetime.now()
            X = trainingData.features
            y = trainingData.labels
            model = make_pipeline(PolynomialFeatures(5), Ridge())

            model.fit(X, y)
            end = datetime.now()
            time_train = (end - start).total_seconds()
            self.logger.debug("Sucessfully deployed " + tools.app_poly)
            self.logger.debug("Time cost to train the model is : %.5f s." % time_train)

            return model, time_train

        def sklearn_poly_predict_fn(inputs):
            return sklearn_poly_model.predict(inputs)

        sklearn_poly_model, time_train = train_sklearn_poly_regression(training_data)
        return sklearn_poly_model, tools.app_poly, time_train

    def deploy_model_sklearn_knn_regression(self, training_data):
        def train_sklearn_knn_regression(trainingData):
            start = datetime.now()
            n_neighbors = 5
            weights = 'distance'  # or 'uniform'
            X = trainingData.features
            y = trainingData.labels
            knn = neighbors.KNeighborsRegressor(weights=weights, n_jobs=1, n_neighbors=n_neighbors)
            knn.fit(X, y)

            end = datetime.now()
            time_train = (end - start).total_seconds()
            self.logger.debug("Sucessfully deployed " + tools.app_knn)
            self.logger.debug("Time cost to train the model is : %.5f s." % time_train)

            return knn, time_train

        def sklearn_knn_predict_fn(inputs):
            return sklearn_knn_model.predict(inputs)

        # global sklearn_knn_model
        sklearn_knn_model, time_train = train_sklearn_knn_regression(training_data)
        return sklearn_knn_model, tools.app_knn, time_train

    def deploy_model_sklearn_svr_rbf_regression(self, training_data):
        def train_sklearn_rbf_regression(trainingData):
            start = datetime.now()
            X = trainingData.features
            y = trainingData.labels
            svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1, cache_size=10000)

            svr_rbf.fit(X, y)
            end = datetime.now()
            time_train = (end - start).total_seconds()
            self.logger.debug("Sucessfully deployed " + tools.app_rbf)
            self.logger.debug("Time cost to train the model is : %.5f s." % time_train)

            return svr_rbf, time_train

        def sklearn_rbf_predict_fn(inputs):
            return sklearn_rbf_model.predict(inputs)

        sklearn_rbf_model, time_train = train_sklearn_rbf_regression(training_data)

        return sklearn_rbf_model, tools.app_rbf, time_train

    def deploy_model_sklearn_gaussion_process_regression(self, training_data):
        def train_sklearn_gp_regression(trainingData):
            X = trainingData.features
            y = trainingData.labels
            # Instanciate a Gaussian Process model
            kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
            # Fit to data using Maximum Likelihood Estimation of the parameters
            start = datetime.now()
            gp.fit(X, y)
            end = datetime.now()
            time_train = (end - start).total_seconds()
            self.logger.debug("Sucessfully deployed " + tools.app_gaussian)
            self.logger.debug("Time cost to train the model is : %.5f s." % time_train)

            return gp, time_train

        def sklearn_gp_predict_fn(inputs):
            return sklearn_gp_model.predict(inputs)

        # global sklearn_gp_models
        sklearn_gp_model, time_train = train_sklearn_gp_regression(training_data)

        return sklearn_gp_model, tools.app_gaussian, time_train

    def deploy_model_sklearn_ensemble_adaboost(self, training_data):
        def train_sklearn_ensemble_adaboost(trainingData):
            X = trainingData.features
            y = trainingData.labels
            start = datetime.now()
            reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                    n_estimators=300)  # , random_state=rng)
            reg.fit(X, y)
            end = datetime.now()
            time_train = (end - start).total_seconds()
            self.logger.debug("Sucessfully deployed " + tools.app_adaboost)
            self.logger.debug("Time cost to train the model is : %.5f s." % (end - start).total_seconds())

            return reg, time_train

        def sklearn_ensemble_adaboost_predict_fn(inputs):
            return sklearn_adaboost_model.predict(inputs)

        sklearn_adaboost_model, time_train = train_sklearn_ensemble_adaboost(training_data)
        return sklearn_adaboost_model, tools.app_adaboost, time_train

    def deploy_model_sklearn_ensemble_gradient_tree_boosting(self, training_data):
        def train_sklearn_ensemble_gradient_tree_boosting(trainingData):
            X = trainingData.features
            y = trainingData.labels
            start = datetime.now()

            reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
            reg.fit(X, y)

            end = datetime.now()
            time_train = (end - start).total_seconds()
            self.logger.debug("Sucessfully deployed " + tools.app_boosting)
            self.logger.debug("Time cost to train the model is : %.5f s." % time_train)

            return reg, time_train

        def sklearn_ensemble_gradient_tree_boosting_predict_fn(inputs):
            return sklearn_gradient_tree_boosting_model.predict(inputs)

        # global sklearn_linear_model
        sklearn_gradient_tree_boosting_model, time_train = train_sklearn_ensemble_gradient_tree_boosting(training_data)

        return sklearn_gradient_tree_boosting_model, tools.app_boosting, time_train

    def deploy_model_sklearn_decision_tree_regression(self, training_data):
        def train_sklearn_decision_tree_regression(trainingData):
            X = trainingData.features
            y = trainingData.labels
            start = datetime.now()
            reg = DecisionTreeRegressor(max_depth=4)
            reg.fit(X, y)

            end = datetime.now()
            time_train = (end - start).total_seconds()
            self.logger.debug("Sucessfully deployed " + tools.app_decision_tree)
            self.logger.debug("Time cost to train the model is : %.5f s." % time_train)

            return reg, time_train

        def sklearn_decision_tree_predict_fn(inputs):
            return sklearn_decision_tree_model.predict(inputs)

        # global sklearn_linear_model
        sklearn_decision_tree_model, time_train = train_sklearn_decision_tree_regression(training_data)
        return sklearn_decision_tree_model, tools.app_decision_tree, time_train
        # -------------------------------------------------------------------------------------------------

    def deploy_xgboost_regression(self, trainingData):
        X = trainingData.features
        y = trainingData.labels
        start = datetime.now()
        reg = XGBRegressor()
        reg.fit(X, y)
        end = datetime.now()

        time_train = (end - start).total_seconds()
        self.logger.debug("Sucessfully deployed " + tools.app_xgboost)
        self.logger.debug("Time cost to train the model is : %.5f s." % time_train)

        return reg, tools.app_xgboost, time_train

        # self.apps_deployed.append(sklearn_decision_tree_model)
        # self.app_names_deployed.append(app_name8)
        # return sklearn_decision_tree_model
        # -------------------------------------------------------------------------------------------------

    def deploy_all_models(self, training_data):
        self.app_names_deployed = []
        self.apps_deployed = []
        self.time_cost_to_train_base_models = []

        if self.input_base_models is not None:
            self.input_base_models = list(self.input_base_models)
        else:
            self.input_base_models = [tools.app_linear, tools.app_xgboost]

        if self.input_ensemble_models is not None:
            self.input_ensemble_models = list(self.input_ensemble_models)
        else:
            self.input_ensemble_models = [tools.app_xgboost]

        if tools.app_linear in self.input_base_models:
            model, name, time = self.deploy_model_sklearn_linear_regression(training_data)
            self.apps_deployed.append(model)
            self.app_names_deployed.append(name)
            self.time_cost_to_train_base_models.append(time)

        if tools.app_poly in self.input_base_models:
            model, name, time = self.deploy_model_sklearn_poly_regression(training_data)
            self.apps_deployed.append(model)
            self.app_names_deployed.append(name)
            self.time_cost_to_train_base_models.append(time)

        if tools.app_knn in self.input_base_models:
            model, name, time = self.deploy_model_sklearn_knn_regression(training_data)
            self.apps_deployed.append(model)
            self.app_names_deployed.append(name)
            self.time_cost_to_train_base_models.append(time)

        if tools.app_rbf in self.input_base_models:
            model, name, time = self.deploy_model_sklearn_svr_rbf_regression(training_data)
            self.apps_deployed.append(model)
            self.app_names_deployed.append(name)
            self.time_cost_to_train_base_models.append(time)

        if tools.app_decision_tree in self.input_base_models:
            model, name, time = self.deploy_model_sklearn_decision_tree_regression(training_data)
            self.apps_deployed.append(model)
            self.app_names_deployed.append(name)
            self.time_cost_to_train_base_models.append(time)

        if tools.app_mllib in self.input_base_models:
            model, name, time = self.deploy_model_mllib_regression(training_data)
            self.apps_deployed.append(model)
            self.app_names_deployed.append(name)
            self.time_cost_to_train_base_models.append(time)

        if tools.app_gaussian in self.input_base_models:
            model, name, time = self.deploy_model_sklearn_gaussion_process_regression(training_data)
            self.apps_deployed.append(model)
            self.app_names_deployed.append(name)
            self.time_cost_to_train_base_models.append(time)

        if tools.app_adaboost in self.input_base_models:
            model, name, time = self.deploy_model_sklearn_ensemble_adaboost(training_data)
            self.apps_deployed.append(model)
            self.app_names_deployed.append(name)
            self.time_cost_to_train_base_models.append(time)

        if tools.app_boosting in self.input_base_models:
            model, name, time = self.deploy_model_sklearn_ensemble_gradient_tree_boosting(training_data)
            self.apps_deployed.append(model)
            self.app_names_deployed.append(name)
            self.time_cost_to_train_base_models.append(time)

        if tools.app_xgboost in self.input_base_models:
            model, name, time = self.deploy_xgboost_regression(training_data)
            self.apps_deployed.append(model)
            self.app_names_deployed.append(name)
            self.time_cost_to_train_base_models.append(time)

        return self.apps_deployed

    def deploy_ensemble_methods(self, training_data):
        self.time_cost_to_train_ensemble_models = []

        if tools.app_adaboost in self.input_ensemble_models:
            model, name, time = self.deploy_model_sklearn_ensemble_adaboost(training_data)
            self.ensemble_models_deployed.append(model)
            self.ensemble_method_names.append(name)
            self.time_cost_to_train_ensemble_models.append(time)

        if tools.app_boosting in self.input_ensemble_models:
            model, name, time = self.deploy_model_sklearn_ensemble_gradient_tree_boosting(training_data)
            self.ensemble_models_deployed.append(model)
            self.ensemble_method_names.append(name)
            self.time_cost_to_train_ensemble_models.append(time)

        if tools.app_xgboost in self.input_ensemble_models:
            model, name, time = self.deploy_xgboost_regression(training_data)
            self.ensemble_models_deployed.append(model)
            self.ensemble_method_names.append(name)
            self.time_cost_to_train_ensemble_models.append(time)

        return self.ensemble_models_deployed

    def set_app_names_deployed(self, names):
        self.app_names_deployed = names
        return True

    def get_app_names_deployed(self):

        return self.app_names_deployed

    # -----------------------------------------------------------------------------------------------

    # the code below is a  modified version of ClassifiedClient.py, adjusted for pure python implementation.
    def get_predictions_to_build_classifier(self, training_data_classifier):

        answers = []
        # print(training_data_classifier.__len__)
        for i in range(len(self.apps_deployed)):
            model_i = self.apps_deployed[i]
            answer_i = self.get_predictions(model_i, training_data_classifier)
            answers.append(answer_i)

        return answers

    def init_classifier_training_values(self, predictions, model_selection_index=None, factor=1):

        global index_of_models_in_classifier
        if model_selection_index != None:
            predictions = [predictions[i] for i in model_selection_index]
            self.app_names_for_classifier = [self.app_names_deployed[i] for i in model_selection_index]
            self.apps_for_classifier = [self.apps_deployed[i] for i in model_selection_index]
            index_of_models_in_classifier = model_selection_index
        else:
            self.app_names_for_classifier = self.app_names_deployed
            self.apps_for_classifier = self.apps_deployed
            index_of_models_in_classifier = range(len(self.app_names_for_classifier))

        # -----------------

        # ------------------
        dimension = len(predictions)

        self.num_model_in_classifier = dimension
        rankings = []
        minimum_errors = []
        num_predictions = len(predictions[0].predictions)

        for i in range(num_predictions):
            values = []
            for j in range(dimension):
                element_of_values = factor * abs(predictions[j].predictions[i] - predictions[j].labels[i]) + \
                    (1 - factor) * predictions[j].latency
                # proportion_of_default_prediction = predictions[j].num_defaults/(predictions[j].num_defaults+ \
                #                                                                predictions[j].num_success)
                # if b_default_prediction_influence:
                #    element_of_values = element_of_values * (1+proportion_of_default_prediction)
                values.append(element_of_values)
                # print(values)

            rankings.append(values.index(min(values)))
            minimum_errors.append(min(values))

        model_counts = []
        for i in range(dimension):
            model_counts.append(rankings.count(i))
        model_counts_str = np.array_str(np.asarray(model_counts))

        self.logger.debug("Queries are classified into %d categories:  " % (dimension))
        self.logger.debug("Counts are: %s." % (model_counts_str))

        return rankings, minimum_errors

    def build_classifier(self, training_data_classifier, y_classifier, C=100):

        distribution = Counter(y_classifier)
        start = datetime.now()
        if len(distribution.keys()) == 1:
            class classifier1:
                def predict(self, x):
                    return [y_classifier[0]]

            classifier = classifier1()
            self.logger.warning("Only one best model is found! New query will only go to this prediction model!")
            self.logger.warning("To use more models, please change the facotr of time term to be greater than 0.")
        else:

            classifier = svm.LinearSVC(C=C)
            classifier.fit(training_data_classifier.features, y_classifier)

        end = datetime.now()
        self.logger.debug("Total time to train linear classifier is: %.4f s." % (end - start).total_seconds())
        self.classifier_name = tools.classifier_linear_name
        return classifier, (end - start).total_seconds()

    def build_classifier_rbf(self, training_data_classifier, y_classifier, C=1):

        distribution = Counter(y_classifier)

        start = datetime.now()
        if len(distribution.keys()) == 1:
            class classifier1:
                def predict(self, x):
                    return [y_classifier[0]]

            classifier = classifier1()
            self.logger.warning("Only one best model is found! New query will only go to this prediction model!")
            self.logger.warning("To use more models, please change the facotr of time term to be greater than 0.")
        else:
            classifier = SVC(C=C, kernel='rbf')
            classifier.fit(training_data_classifier.features, y_classifier)

        end = datetime.now()
        self.logger.debug("Total time to train rbf classifier is: %.4f s." % (end - start).total_seconds())
        self.classifier_name = tools.classifier_rbf_name
        return classifier, (end - start).total_seconds()

    def build_classifier_xgboost(self, training_data_classifier, y_classifier):
        start = datetime.now()
        distribution = Counter(y_classifier)

        if len(distribution.keys()) == 1:
            class classifier1:
                def predict(self, x):
                    return [y_classifier[0]]

            classifier = classifier1()
            self.logger.warning(
                "Warning: Only one best model is found! New query will only go to this prediction model!")
            self.logger.warning("To use more models, please change the facotr of time term to be greater than 0.")
        else:
            classifier = XGBClassifier()
            classifier.fit(training_data_classifier.features, y_classifier)

        end = datetime.now()
        self.logger.debug("Total time to train xgboost classifier is: %.4f s." % (end - start).total_seconds())
        self.classifier_name = tools.classifier_xgboost_name
        return classifier, (end - start).total_seconds()

    def get_cluster_points(self, model_number, y_classifier, training_points_features):
        results = []
        for i, element in enumerate(y_classifier):
            if element == model_number:
                results.append(np.asarray(training_points_features[i]))
        return np.asarray(results)

    def predict(self, x):
        return self.get_classified_prediction(self.classifier, x)

    def predicts(self, xs):
        return [self.get_classified_prediction(self.classifier, x) for x in xs]

    def fit(self, training_data, testing_data=None, b_select_classifier=False):
        training_data_model, training_data_classifier = tools.split_data_to_2(training_data)

        models, time_cost_to_train_base_models = self.deploy_all_models(training_data_model)

        # get predictions to build the classifier
        answers_for_classifier = self.get_predictions_to_build_classifier(training_data_classifier)
        y_classifier, errors = self.init_classifier_training_values(answers_for_classifier,
                                                                    # model_selection_index=index_models,
                                                                    factor=1)

        if not b_select_classifier:
            classifier, time_cost_to_train_the_best_classifier = self.build_classifier_xgboost(training_data_classifier,
                                                                                               y_classifier)
        else:
            classifier, NRMSE_classifier_selection, time_cost_to_select_classifiers, \
                time_cost_to_train_the_best_classifier = \
                self.select_classifiers(training_data_classifier, y_classifier, testing_data)

        self.classifier = classifier
        self.num_total_training_points = len(training_data_model.labels) + len(training_data_classifier.labels)
        self.training_data = training_data

    def select_classifiers(self, training_data_classifier, y_classifier, testing_data):
        # global classifier_names_candidate
        classifier_names_candidate = ["Nearest Neighbors", "Linear SVM",  # "RBF SVM",
                                      "Decision Tree", "Random Forest", "Neural Net",  # "AdaBoost",
                                      "Naive Bayes", "QDA"]
        start = datetime.now()
        distribution = Counter(y_lcassifier)
        time_costs = []

        if len(distribution.keys()) == 1:
            class classifier1:
                def predict(self, x):
                    return [y_classifier[0]]

            classifier = classifier1()
            self.logger.warning(
                "Warning: Only one best model is found! New query will only go to this prediction model!")
            self.logger.warning("To use more models, please change the facotr of time term to be greater than 0.")
            time_costs.append(0.0)
        else:

            classifiers = [
                KNeighborsClassifier(3),
                svm.LinearSVC(C=100),  # SVC(kernel="linear", C=0.025),
                # SVC(gamma=2, C=1),
                DecisionTreeClassifier(max_depth=5),
                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                MLPClassifier(alpha=1),
                # AdaBoostClassifier(),
                GaussianNB(),
                QuadraticDiscriminantAnalysis()]

            # iterate over classifiers
            NRMSEs = []
            scores = []
            self.logger.info("Start selecting the best classifier:")
            for name, clf in zip(classifier_names_candidate, classifiers):
                self.logger.info("Classifier: " + name)
                time0 = datetime.now()
                clf.fit(training_data_classifier.features, y_classifier)
                time1 = datetime.now()
                score = clf.score(training_data_classifier.features, y_classifier)
                predictions_classified = self.get_classified_predictions(clf, testing_data)
                NRMSEs.append(predictions_classified.NRMSE())
                scores.append(score)
                time_costs.append((time1 - time0).seconds)
                print("-----------------------------------------------------------")

            self.logger.info("Summary:")
            self.logger.info("NRMSEs of the classifiers:" + str(NRMSEs))
            self.logger.info("Scores of the classifiers:" + str(scores))

            index = NRMSEs.index(min(NRMSEs))
            classifier = classifiers[index]
            self.logger.info("The best classifier is: " + classifier_names_candidate[index])
            self.logger.info("The best NRMSE is: " + str(NRMSEs[index]))
            self.classifier_name = classifier_names_candidate[index]
            time_cost = time_costs[index]

        return classifier, NRMSEs, time_costs, time_cost  # time cost of the best classifier

    def plot_training_data_2d(self):
        fig, ax = plt.subplots()

        ax.plot(self.training_data.features[:, 0], self.training_data.labels,
                tools.markers_matplotlib[5], label='real data', linewidth=0.0)
        plt.show()

    def matplotlib_plot_2D(self, answers, b_show_division_boundary=True, b_show_god_classifier=False, y_classifier=None,
                           xmin=None, xmax=None):
        font_size = 35
        names = self.app_names_for_classifier
        symbols = ['*', '1', 'v', 'o', 'h', 'x']
        if b_show_division_boundary:
            if b_show_god_classifier:
                gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
            else:
                gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        else:
            if b_show_god_classifier:
                gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            else:
                gs = gridspec.GridSpec(1, 1)
        fig = plt.figure()
        ax1 = plt.subplot(gs[0])
        # ax.plot(a, c, 'k--', label='Model length')
        # ax.plot(a, d, 'k:', label='Data length')
        # ax.plot(a, c + d, 'k', label='Total message length')
        for i in range(len(self.app_names_for_classifier)):
            # print(answers.get_vispy_plot_data(i))
            if answers.get_vispy_plot_data_2d(i) != []:
                ax1.plot(answers.get_vispy_plot_data_2d(i)[:, 0],
                         answers.get_vispy_plot_data_2d(i)[:, 1],
                         symbols[i],
                         label=names[i],
                         linewidth=0.0)
        ax1.plot(answers.features[:, 0], answers.labels, symbols[5], label='real data', linewidth=0.0)

        # Now add the legend with some customizations.
        legend1 = ax1.legend(loc='upper right', shadow=True, fontsize=font_size)

        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend1.get_frame()
        frame.set_facecolor('0.90')

        ax1.set_xlabel(answers.headers[0], fontsize=font_size)
        ax1.set_ylabel(answers.headers[1], fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        #ax1.set_title("Classified Regression Curve")
        if xmin != None:
            ax1.set_xlim(xmin, xmax)

        if b_show_division_boundary:
            ax2 = plt.subplot(gs[1])
            for i in range(len(self.app_names_for_classifier)):
                X, Y = tools.get_values_equal_to_(i, answers.get_vispy_plot_data_2d()[:, 0], answers.modelID)
                # print(X)
                if X != []:
                    ax2.scatter(X, Y, s=2, label=names[i], color=colors_matploblib[i])
            legend2 = ax2.legend(loc='right', shadow=True)
            ax2.set_xlabel(answers.headers[0])
            ax2.set_ylabel("Model ID")
            ax2.set_title("Decision boundary of the classified prediction method")
            ax2.set_yticks(np.arange(-1, max(answers.modelID) + 2, 1.0))
            ax2.set_ylim(-1, max(answers.modelID) + 1)
            if xmin != None:
                ax2.set_xlim(xmin, xmax)

        if b_show_division_boundary:
            if b_show_god_classifier:
                ax3 = plt.subplot(gs[2])
        if not b_show_division_boundary:
            if b_show_god_classifier:
                ax3 = plt.subplot(gs[1])

        if b_show_god_classifier:
            if y_classifier == None:
                self.logger.critical("y_classifier values are not provided! Program ended!")
                exit(1)
            for i in range(len(self.app_names_for_classifier)):
                X, Y = tools.get_values_equal_to_(i, answers.get_vispy_plot_data_2d()[:, 0], y_classifier)
                if X != []:
                    ax3.scatter(X, Y, s=2, label=names[i], color=colors_matploblib[i])
            # ax3.scatter(answers.get_vispy_plot_data()[:, 0], y_classifier,s=1)
            legend3 = ax3.legend(loc='right', shadow=True)
            ax3.set_xlabel(answers.headers[0])
            ax3.set_ylabel("Model ID")
            ax3.set_title("Decision boundary of the god classifier")
            ax3.set_yticks(np.arange(-1, max(answers.modelID) + 2, 1.0))
            ax3.set_ylim(-1, len(self.app_names_for_classifier) + 1)
            if xmin != None:
                ax3.set_xlim(xmin, xmax)

        plt.show()

        return

    def matplotlib_plot_2D_all_models(self, answers, answers_from_all_models,
                                      xmin=None, xmax=None):
        font_size = 35
        names = self.app_names_for_classifier
        symbols = tools.markers_matplotlib

        fig = plt.figure()
        ax1 = plt.subplot()
        # ax.plot(a, c, 'k--', label='Model length')
        # ax.plot(a, d, 'k:', label='Data length')
        # ax.plot(a, c + d, 'k', label='Total message length')

        # for i in range(len(self.app_names_for_classifier)):
        #     # print(answers.get_vispy_plot_data(i))
        #     if answers.get_vispy_plot_data_2d(i) != []:
        #         ax1.plot(answers.get_vispy_plot_data_2d(i)[:, 0],
        #                  answers.get_vispy_plot_data_2d(i)[:, 1],
        #                  symbols[i],
        #                  label=names[i],
        #                  linewidth=0.0)

        for i in range(len(self.app_names_for_classifier)):
            ax1.plot(answers_from_all_models[i].features[:, 0],
                     answers_from_all_models[i].predictions,
                     symbols[i],
                     label=names[i],
                     linewidth=0.0
                     )

        ax1.plot(answers_from_all_models[i].features[:, 0],
                 answers_from_all_models[i].labels,
                 symbols[5],
                 label='real data',
                 linewidth=0.0
                 )

        # Now add the legend with some customizations.
        legend1 = ax1.legend(loc='upper right', shadow=True, fontsize=font_size)

        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend1.get_frame()
        frame.set_facecolor('0.90')

        ax1.set_xlabel(answers.headers[0], fontsize=font_size)
        ax1.set_ylabel(answers.headers[1], fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        # ax1.set_title("Regression Curves of Linear and Polynomial Models")
        if xmin != None:
            ax1.set_xlim(xmin, xmax)

        plt.show()

        return

    def plot_classified_prediction_curves_2D(self, answers, knn_neighbours=None):
        fig = vp.Fig(show=False)
        color = (0.8, 0.25, 0.)
        fig1 = fig[0:4, 0:4]
        # fig2 = fig[0:4, 4:6]

        names = self.app_names_for_classifier
        colors = [r, g, blue, color1, color2, color3, color4, color5]
        symbols = tools.markers_matplotlib
        for i in range(len(self.app_names_for_classifier)):
            # print(answers.headers)
            # print(answers.headers[0])
            # print(answers.headers[1])
            if answers.get_vispy_plot_data_2d(i) != []:
                fig1.plot(answers.get_vispy_plot_data_2d(i), symbol=symbols[i], width=0.0, marker_size=6.,
                          # color=colors[i],
                          # face_color=colors[i] ,
                          title='Classified Regression Curve',
                          xlabel=answers.headers[0],
                          ylabel=answers.headers[1])

        # for i in range(len(client.app_names_for_classifier)):
        #    fig2.plot([0,0],symbol=symbols[i],  marker_size=6.)
        fig.show(run=True)
        return

    def plot_training_data_3d(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(self.training_data.features[:, 0], self.training_data.features[:, 1], self.training_data.labels)
        plt.show()

    def matplotlib_plot_3D(self, answers, plot_region=[]):
        names = self.app_names_for_classifier
        symbols = tools.markers_matplotlib

        # gs = gridspec.GridSpec(3, 1, height_ratios=[3])

        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        # ax.plot(a, c, 'k--', label='Model length')
        # ax.plot(a, d, 'k:', label='Data length')
        # ax.plot(a, c + d, 'k', label='Total message length')

        for i in range(len(self.app_names_for_classifier)):
            # print(answers.get_vispy_plot_data(i))
            if answers.get_vispy_plot_data_3d(i) != []:
                data_to_plot = answers.get_vispy_plot_data_3d(i)
                ax1.plot(data_to_plot[:, 0],
                         data_to_plot[:, 1],
                         data_to_plot[:, 2],
                         symbols[i],
                         label=names[i],
                         linewidth=0.0)

        # Now add the legend with some customizations.
        legend1 = ax1.legend(loc='upper left', shadow=True)

        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend1.get_frame()
        frame.set_facecolor('0.90')

        ax1.set_xlabel(answers.headers[0])
        ax1.set_ylabel(answers.headers[1])
        ax1.set_zlabel(answers.headers[2])
        ax1.set_title("Classified Regression Query Space")
        if plot_region != []:
            ax1.set_xlim(plot_region[0:2])
            ax1.set_ylim(plot_region[2:4])

        plt.show()

        return

    def matplotlib_plot_3D_decision_boundary(self, answers, plot_region=[]):
        names = self.app_names_for_classifier
        symbols = tools.markers_matplotlib

        fig = plt.figure()
        ax2 = plt.subplot()
        for i in range(len(self.app_names_for_classifier)):
            X, Y, Z = tools.get_values_equal_to_3D(i, answers.get_vispy_plot_data_3d(), answers.modelID)
            # print(X)
            if X != []:
                ax2.plot(X, Y, symbols[i], label=names[i], color=tools.colors_matploblib[i], linewidth=0.0)
        legend2 = ax2.legend(loc='upper left', shadow=True)
        ax2.set_xlabel(answers.headers[0])
        ax2.set_ylabel(answers.headers[1])
        #ax2.set_title("Decision boundary of the classified prediction method")
        # ax2.set_yticks(np.arange(-1, max(answers.modelID) + 2, 1.0))
        # ax2.set_ylim(-1, max(answers.modelID) + 1)
        if plot_region != []:
            ax2.set_xlim(plot_region[0:2])
            ax2.set_ylim(plot_region[2:4])
        plt.show()

        return

    def matplotlib_plot_3D_distribution_of_best_model(self, answers, y_classifier, plot_region=[]):
        names = self.app_names_for_classifier
        symbols = tools.markers_matplotlib

        fig = plt.figure()
        ax2 = plt.subplot()
        for i in range(len(self.app_names_for_classifier)):
            X, Y, Z = tools.get_values_equal_to_3D(i, answers.get_vispy_plot_data_3d(), y_classifier)
            # print(X)
            if X != []:
                ax2.plot(X, Y, symbols[i], label=names[i], color=tools.colors_matploblib[i], linewidth=0.0)
        legend2 = ax2.legend(loc='upper left', shadow=True)
        ax2.set_xlabel(answers.headers[0])
        ax2.set_ylabel(answers.headers[1])
        #ax2.set_title("Decision boundary of the classified prediction method")
        # ax2.set_yticks(np.arange(-1, max(answers.modelID) + 2, 1.0))
        # ax2.set_ylim(-1, max(answers.modelID) + 1)
        if plot_region != []:
            ax2.set_xlim(plot_region[0:2])
            ax2.set_ylim(plot_region[2:4])
        plt.show()

        return

    def boxplot(self, predictions_from_base_models, classified_predictions, y_classifier):
        num_of_plot = len(predictions_from_base_models)+1
        labels = classified_predictions.labels
        data_to_plot = []
        variance = []
        xlabels = self.input_base_models
        # print(xlabels)
        for i in range(num_of_plot-1):
            data_to_plot.append(np.subtract(np.asarray(
                predictions_from_base_models[i].predictions), np.asarray(labels)))
            variance.append(
                np.var(np.subtract(np.asarray(predictions_from_base_models[i].predictions), np.asarray(labels))))

        data_to_plot.append(np.subtract(np.asarray(classified_predictions.predictions), np.asarray(labels)))
        variance.append(np.var(np.subtract(np.asarray(classified_predictions.predictions), np.asarray(labels))))
        # variance.append(np.var(np.subtract(np.asarray(classified_predictions.predictions),np.asarray(y_classifier))))
        xlabels.append("classified")
        fig = plt.figure(1)  # , figsize=(9, 6))
        ax = fig.add_subplot(111)
        # Create the boxplot
        bp = ax.boxplot(data_to_plot, showfliers=False, showmeans=True)
        ax.set_xticklabels(xlabels)
        ax.set_ylabel("absolute error")
        plt.show()
        return variance

    def run2d(self, data):

        data.remove_repeated_x_1d()
        if self.b_disorder:
            data.disorder2d()

        time_program_start = datetime.now()

        training_data_model, training_data_classifier, testing_data = tools.split_data(data)

        training_data_model = training_data_model  # .get_before(300000)
        training_data_classifier = training_data_classifier  # .get_before(300000)
        testing_data = testing_data  # .get_before(300000)

        statistics = self.summary
        statistics.file_name = data.file
        statistics.num_of_instances = len(data)

        # deploy all models
        models = self.deploy_all_models(training_data_model)
        statistics.s_training_time_all_models = list(self.time_cost_to_train_base_models)

        # get predictions to build the classifier
        answers_for_classifier = self.get_predictions_to_build_classifier(training_data_classifier)
        # save tempary results
        statistics.s_model_headers = list(self.app_names_deployed)
        for element in answers_for_classifier:
            statistics.NRMSE_training_classifier.append(element.NRMSE())
            # statistics.time_query_execution_on_classifier.append(element.time_query_execution_on_classifier)
            statistics.time_query_processing_all_models.append(element.time_total)
            statistics.time_average_query_processing_of_all_models.append(element.latency)

        # train and select the classifier
        # init training values to build the classifier
        # index_models = [0, 1, 2]
        y_classifier, errors = self.init_classifier_training_values(answers_for_classifier,
                                                                    # model_selection_index=index_models,
                                                                    factor=1)
        # select the best classifier

        if not self.b_select_classifier:
            if self.classifier_type is tools.classifier_xgboost_name:
                classifier, time_cost_to_train_the_best_classifier = self.build_classifier_xgboost(
                    training_data_classifier,
                    y_classifier)
            if self.classifier_type is tools.classifier_linear_name:
                classifier, time_cost_to_train_the_best_classifier = self.build_classifier(training_data_classifier,
                                                                                           y_classifier)
            if self.classifier_type is tools.classifier_rbf_name:
                classifier, time_cost_to_train_the_best_classifier = self.build_classifier_rbf(training_data_classifier,
                                                                                               y_classifier)
            statistics.classifier_name = self.classifier_type
        else:
            classifier, NRMSE_classifier_selection, time_cost_to_select_classifiers,\
                time_cost_to_train_the_best_classifier = self.select_classifiers(
                    training_data_classifier, y_classifier, testing_data)
            statistics.classifier_name = self.classifier_name

        time_train_CPM = datetime.now()

        statistics.s_training_time_all_models.append((
            time_train_CPM - time_program_start).total_seconds())  # time cost for our classified prediction method, \
        # will be updated later on
        #
        # save tempary results
        # statistics.classifier_selection_names = client.classifier_names_candidate
        # statistics.classifier_selection_NRMSEs = NRMSE_classifier_selection
        # index = NRMSE_classifier_selection.index(min(NRMSE_classifier_selection))
        # statistics.classifier_name = client.classifier_names_candidate[index]
        # statistics.time_training_classifiers = list(time_cost_to_select_classifiers)
        statistics.time_training_classifier = time_cost_to_train_the_best_classifier

        '''
        cc=ClientClass()
        cc.fit(training_data_model,training_data_classifier)
        test_point = testing_data.features[0]
        print(test_point)
        print(client.get_classified_prediction(classifier,test_point))
        return
        '''
        # get predictions of each base prediction model for the testing dataset, to evaludate
        answers_for_testing = self.get_predictions_from_models_for_testing(testing_data)
        # save temparary results
        statistics.model_names_for_classifier = list(self.app_names_for_classifier)
        for element in answers_for_testing:
            statistics.NRMSE.append(element.NRMSE())

        # query to the classified prediction method
        predictions_classified = self.get_classified_predictions(classifier, testing_data)
        # save temparary results
        statistics.s_model_headers.append(tools.CPM_name)
        statistics.NRMSE.append(predictions_classified.NRMSE())
        statistics.time_query_execution_on_classifier = predictions_classified.time_query_execution_on_classifier
        statistics.time_query_processing_all_models.append(predictions_classified.time_total)
        # statistics.s_training_time_all_models.append(predictions_classified.time_total)
        statistics.time_average_query_processing_of_all_models.append(predictions_classified.latency)
        statistics.num_of_instances_in_testing_dataset = predictions_classified.num_of_instances

        # get ensemble results

        ensemble_methods = self.deploy_ensemble_methods(training_data_model)
        answers_ensemble = self.get_predictions_from_models(ensemble_methods, testing_data)
        # ensemble_answers0 = client.get_predictions(ensemble_methods[0], testing_data)
        # ensemble_answers1 = client.get_predictions(ensemble_methods[1], testing_data)

        # save ensemble results
        for element in answers_ensemble:
            statistics.NRMSE.append(element.NRMSE())
            statistics.time_query_processing_all_models.append(element.time_total)
            statistics.time_average_query_processing_of_all_models.append(element.latency)
        for element in self.ensemble_method_names:
            statistics.s_model_headers.append(element)
        for element in self.time_cost_to_train_ensemble_models:
            statistics.s_training_time_all_models.append(element)

        # calculate classifier accuracy or precision
        y_classifier_testing, errors_ideal = self.init_classifier_training_values(answers_for_testing,
                                                                                  # model_selection_index=index_models,
                                                                                  factor=1)
        # save results of classifier accuracy
        statistics.classifier_accuracy = predictions_classified.predict_precision(y_classifier_testing)
        statistics.NRMSE_ideal = tools.NRMSE(errors_ideal, answers_for_testing[0].labels)

        time_program_end = datetime.now()
        statistics.time_program = (time_program_end - time_program_start).seconds
        # print summary
        statistics.print_summary()

        # vispy_plt.plot_classified_prediction_curves_2D(predictions_classified)
        # vispy_plt.matplotlib_plot_2D(predictions_classified, b_show_division_boundary=True,\
        # b_show_god_classifier=True,
        # y_classifier=y_classifier)

        if self.b_show_plot:
            self.matplotlib_plot_2D(predictions_classified, b_show_division_boundary=True,
                                    b_show_god_classifier=True, y_classifier=y_classifier_testing)

        self.predictions_testing = answers_for_testing

        # self.matplotlib_plot_2D_all_models(predictions_classified,answers_for_testing)
        return statistics

    def run3d(self, data):

        data.remove_repeated_x_2d()

        if self.b_disorder:
            data.disorderNd()

        time_program_start = datetime.now()

        training_data_model, training_data_classifier, testing_data = tools.split_data(data)

        training_data_model = training_data_model  # .get_before(300000)
        training_data_classifier = training_data_classifier  # .get_before(300000)
        testing_data = testing_data  # .get_before(300000)

        statistics = self.summary
        statistics.file_name = data.file
        statistics.num_of_instances = len(data)

        # deploy all models
        models = self.deploy_all_models(training_data_model)
        statistics.s_training_time_all_models = list(self.time_cost_to_train_base_models)

        # get predictions to build the classifier
        answers_for_classifier = self.get_predictions_to_build_classifier(training_data_classifier)
        # save tempary results
        statistics.s_model_headers = list(self.app_names_deployed)
        for element in answers_for_classifier:
            statistics.NRMSE_training_classifier.append(element.NRMSE())
            # statistics.time_query_execution_on_classifier.append(element.time_query_execution_on_classifier)
            statistics.time_query_processing_all_models.append(element.time_total)
            statistics.time_average_query_processing_of_all_models.append(element.latency)

        # train and select the classifier
        # init training values to build the classifier
        # index_models = [0, 1, 2]
        y_classifier, errors = self.init_classifier_training_values(answers_for_classifier,
                                                                    # model_selection_index=index_models,
                                                                    factor=1)

        # select the best classifier
        if not self.b_select_classifier:
            if self.classifier_type is tools.classifier_xgboost_name:
                classifier, time_cost_to_train_the_best_classifier = self.build_classifier_xgboost(
                    training_data_classifier,
                    y_classifier)
            if self.classifier_type is tools.classifier_linear_name:
                classifier, time_cost_to_train_the_best_classifier = self.build_classifier(training_data_classifier,
                                                                                           y_classifier)
            if self.classifier_type is tools.classifier_rbf_name:
                classifier, time_cost_to_train_the_best_classifier = self.build_classifier_rbf(training_data_classifier,
                                                                                               y_classifier)
            statistics.classifier_name = self.classifier_type
        else:
            classifier, NRMSE_classifier_selection, time_cost_to_select_classifiers,\
                time_cost_to_train_the_best_classifier = self.select_classifiers(
                    training_data_classifier, y_classifier, testing_data)
            statistics.classifier_name = self.classifier_name

        time_train_CPM = datetime.now()

        statistics.s_training_time_all_models.append((
            time_train_CPM - time_program_start).total_seconds())  # time cost for our classified prediction method,
        # will be updated later on
        #
        # save tempary results
        # statistics.classifier_selection_names = client.classifier_names_candidate
        # statistics.classifier_selection_NRMSEs = NRMSE_classifier_selection
        # index = NRMSE_classifier_selection.index(min(NRMSE_classifier_selection))
        # statistics.classifier_name = client.classifier_names_candidate[index]
        # statistics.time_training_classifiers = list(time_cost_to_select_classifiers)
        statistics.time_training_classifier = time_cost_to_train_the_best_classifier

        '''
        cc=ClientClass()
        cc.fit(training_data_model,training_data_classifier)
        test_point = testing_data.features[0]
        print(test_point)
        print(client.get_classified_prediction(classifier,test_point))
        return
        '''
        # get predictions of each base prediction model for the testing dataset, to evaludate
        answers_for_testing = self.get_predictions_from_models_for_testing(testing_data)
        # save temparary results
        statistics.model_names_for_classifier = list(self.app_names_for_classifier)
        for element in answers_for_testing:
            statistics.NRMSE.append(element.NRMSE())

        # query to the classified prediction method
        predictions_classified = self.get_classified_predictions(classifier, testing_data)
        # save temparary results
        statistics.s_model_headers.append(tools.CPM_name)
        statistics.NRMSE.append(predictions_classified.NRMSE())
        statistics.time_query_execution_on_classifier = predictions_classified.time_query_execution_on_classifier
        # statistics.s_training_time_all_models.append(predictions_classified.time_total)
        statistics.time_average_query_processing_of_all_models.append(predictions_classified.latency)
        statistics.num_of_instances_in_testing_dataset = predictions_classified.num_of_instances

        # get ensemble results

        ensemble_methods = self.deploy_ensemble_methods(training_data_model)
        answers_ensemble = self.get_predictions_from_models(ensemble_methods, testing_data)
        # ensemble_answers0 = client.get_predictions(ensemble_methods[0], testing_data)
        # ensemble_answers1 = client.get_predictions(ensemble_methods[1], testing_data)

        # save ensemble results
        for element in answers_ensemble:
            statistics.NRMSE.append(element.NRMSE())
            statistics.time_query_processing_all_models.append(element.time_total)
            statistics.time_average_query_processing_of_all_models.append(element.latency)
        for element in self.ensemble_method_names:
            statistics.s_model_headers.append(element)
        for element in self.time_cost_to_train_ensemble_models:
            statistics.s_training_time_all_models.append(element)

        # calculate classifier accuracy or precision
        y_classifier_testing, errors_ideal = self.init_classifier_training_values(answers_for_testing,
                                                                                  # model_selection_index=index_models,
                                                                                  factor=1)

        # save results of classifier accuracy
        statistics.classifier_accuracy = predictions_classified.predict_precision(y_classifier_testing)
        statistics.NRMSE_ideal = tools.NRMSE(errors_ideal, answers_for_testing[0].labels)

        # print(errors_ideal)
        # print(answers_for_testing[0].labels)

        time_program_end = datetime.now()
        statistics.time_program = (time_program_end - time_program_start).seconds
        # print summary
        statistics.print_summary()

        # vispy_plt.plot_classified_prediction_curves_2D(predictions_classified)
        # vispy_plt.matplotlib_plot_2D(predictions_classified, b_show_division_boundary=True, \
        #    b_show_god_classifier=True, y_classifier=y_classifier)

        if self.b_show_plot:
            self.matplotlib_plot_3D_distribution_of_best_model(predictions_classified, y_classifier_testing)
            self.matplotlib_plot_3D(predictions_classified)
            self.matplotlib_plot_3D_decision_boundary(predictions_classified)

        self.predictions_testing = answers_for_testing
        return statistics

    def run(self, data):
        if self.b_disorder:
            data.disorderNd()

        time_program_start = datetime.now()

        training_data_model, training_data_classifier, testing_data = tools.split_data(data)

        training_data_model = training_data_model  # .get_before(300000)
        training_data_classifier = training_data_classifier  # .get_before(300000)
        testing_data = testing_data  # .get_before(300000)

        statistics = self.summary
        statistics.num_of_instances = len(data)
        statistics.file_name = data.file

        # deploy all models
        models = self.deploy_all_models(training_data_model)
        statistics.s_training_time_all_models = list(self.time_cost_to_train_base_models)

        # get predictions to build the classifier
        answers_for_classifier = self.get_predictions_to_build_classifier(training_data_classifier)
        # save tempary results
        statistics.s_model_headers = list(self.app_names_deployed)
        for element in answers_for_classifier:
            statistics.NRMSE_training_classifier.append(element.NRMSE())
            # statistics.time_query_execution_on_classifier.append(element.time_query_execution_on_classifier)
            statistics.time_query_processing_all_models.append(element.time_total)
            statistics.time_average_query_processing_of_all_models.append(element.latency)

        # train and select the classifier
        # init training values to build the classifier
        # index_models = [0, 1, 2]
        y_classifier, errors = self.init_classifier_training_values(answers_for_classifier,
                                                                    # model_selection_index=index_models,
                                                                    factor=1)

        #########################################################

        # classifier, time_cost_to_train_the_best_classifier = self.build_classifier(
        #     training_data_classifier,
        #     y_classifier)
        #
        #
        #
        # classifier, time_cost_to_train_the_best_classifier = self.build_classifier_xgboost(
        #     training_data_classifier,
        #     y_classifier)

        # classifier, time_cost_to_train_the_best_classifier = self.build_classifier_rbf(
        #     training_data_classifier,
        #     y_classifier)
        #
        # exit(1)
        ########################################################
        # select the best classifier
        if not self.b_select_classifier:
            if self.classifier_type is tools.classifier_xgboost_name:
                classifier, time_cost_to_train_the_best_classifier = self.build_classifier_xgboost(
                    training_data_classifier,
                    y_classifier)
            if self.classifier_type is tools.classifier_linear_name:
                classifier, time_cost_to_train_the_best_classifier = self.build_classifier(training_data_classifier,
                                                                                           y_classifier)
            if self.classifier_type is tools.classifier_rbf_name:
                classifier, time_cost_to_train_the_best_classifier = self.build_classifier_rbf(training_data_classifier,
                                                                                               y_classifier)
            statistics.classifier_name = self.classifier_type
        else:
            classifier, NRMSE_classifier_selection, time_cost_to_select_classifiers, \
                time_cost_to_train_the_best_classifier = self.select_classifiers(
                    training_data_classifier, y_classifier, testing_data)
            statistics.classifier_name = self.classifier_name

        time_train_CPM = datetime.now()

        # time cost for our classified prediction method, will be updated later on
        statistics.s_training_time_all_models.append((
            time_train_CPM - time_program_start).total_seconds())
        # save tempary results
        # statistics.classifier_selection_names = client.classifier_names_candidate
        # statistics.classifier_selection_NRMSEs = NRMSE_classifier_selection
        # index = NRMSE_classifier_selection.index(min(NRMSE_classifier_selection))
        # statistics.classifier_name = client.classifier_names_candidate[index]
        # statistics.time_training_classifiers = list(time_cost_to_select_classifiers)
        statistics.time_training_classifier = time_cost_to_train_the_best_classifier

        '''
        cc=ClientClass()
        cc.fit(training_data_model,training_data_classifier)
        test_point = testing_data.features[0]
        print(test_point)
        print(client.get_classified_prediction(classifier,test_point))
        return
        '''
        # get predictions of each base prediction model for the testing dataset, to evaludate
        answers_for_testing = self.get_predictions_from_models_for_testing(testing_data)
        # save temparary results
        statistics.model_names_for_classifier = list(self.app_names_for_classifier)
        for element in answers_for_testing:
            statistics.NRMSE.append(element.NRMSE())

        # query to the classified prediction method
        predictions_classified = self.get_classified_predictions(classifier, testing_data)
        # save temparary results
        statistics.s_model_headers.append(tools.CPM_name)
        statistics.NRMSE.append(predictions_classified.NRMSE())
        statistics.time_query_execution_on_classifier = predictions_classified.time_query_execution_on_classifier
        # statistics.s_training_time_all_models.append(predictions_classified.time_total)
        statistics.time_average_query_processing_of_all_models.append(predictions_classified.latency)
        statistics.num_of_instances_in_testing_dataset = predictions_classified.num_of_instances

        # get ensemble results

        ensemble_methods = self.deploy_ensemble_methods(training_data_model)
        answers_ensemble = self.get_predictions_from_models(ensemble_methods, testing_data)
        # ensemble_answers0 = client.get_predictions(ensemble_methods[0], testing_data)
        # ensemble_answers1 = client.get_predictions(ensemble_methods[1], testing_data)

        # save ensemble results
        for element in answers_ensemble:
            statistics.NRMSE.append(element.NRMSE())
            statistics.time_query_processing_all_models.append(element.time_total)
            statistics.time_average_query_processing_of_all_models.append(element.latency)
        for element in self.ensemble_method_names:
            statistics.s_model_headers.append(element)
        for element in self.time_cost_to_train_ensemble_models:
            statistics.s_training_time_all_models.append(element)

        # calculate classifier accuracy or precision
        y_classifier_testing, errors_ideal = self.init_classifier_training_values(answers_for_testing,
                                                                                  # model_selection_index=index_models,
                                                                                  factor=1)
        # save results of classifier accuracy
        statistics.classifier_accuracy = predictions_classified.predict_precision(y_classifier_testing)
        statistics.NRMSE_ideal = tools.NRMSE(errors_ideal, answers_for_testing[0].labels)

        # print(errors_ideal)
        # print(answers_for_testing[0].labels)

        time_program_end = datetime.now()
        statistics.time_program = (time_program_end - time_program_start).seconds
        # print summary
        statistics.print_summary()

        # vispy_plt.plot_classified_prediction_curves_2D(predictions_classified)
        # vispy_plt.matplotlib_plot_2D(predictions_classified, b_show_division_boundary=True, \
        #   b_show_god_classifier=True, y_classifier=y_classifier)

        # client.matplotlib_plot_2D(predictions_classified)

        # client.matplotlib_plot_3D(predictions_classified)
        # client.matplotlib_plot_3D_decision_boundary(predictions_classified)

        self.predictions_testing = answers_for_testing

        print(self.boxplot(answers_for_testing, predictions_classified, y_classifier_testing))
        return statistics

    def get_NRMSE_for_clusters(self, answers_for_classifier, y_classifier):
        # print(answers_for_classifier[0].labels)
        # print(y_classifier)
        indexs = []
        xs = []
        error_models = []
        NRMSE_comparisons = []
        range_query = max(answers_for_classifier[0].labels) - min(answers_for_classifier[0].labels)
        # print(range_query)
        for i in range(len(self.app_names_deployed)):
            index_i = [j for j, x in enumerate(y_classifier) if x == i]
            indexs.append(index_i)
            # print(index_i)

            # get the points for each index
            xs_i = []
            error_i = []
            NRMSE_comparisons_i = []
            predictions_best_i = [answers_for_classifier[0].labels[j] for j in index_i]
            # print(predictions_best_i)
            for method_i in range(len(self.app_names_deployed)):
                xs_i_j = [answers_for_classifier[method_i].predictions[j] for j in index_i]
                # print(xs_i_j)
                error_i_j = [abs(predictions_best_i[j] - xs_i_j[j]) for j in range(len(predictions_best_i))]
                # print(xs_i_j)
                # print(error_i)
                xs_i.append(xs_i_j)
                error_i.append(error_i_j)
                NRMSE_comparisons_i.append(tools.NRMSE_with_range(error_i_j, range_query))

            xs.append(xs_i)
            error_models.append(error_i)
            NRMSE_comparisons.append(NRMSE_comparisons_i)
            # print(NRMSE_comparisons_i)

        # print(error_models)
        return NRMSE_comparisons


# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    fields = ['Relative_Humidity', 'energy_output']
    y_column = 1  # should be the order in the input file, not in the "fields" order.
    data = tools.load_csv("../data/5CCPP/5Folds5x2_pp.csv", fields, y_column)

    training_data, testing_data = tools.split_data_to_2(data, 0.66667)

    '''
    training_data_model = training_data_model.get_before(100)
    training_data_classifier = training_data_classifier.get_before(100)
    testing_data = testing_data.get_before(100)
    '''

    cs = CRegression()
    cs.fit(training_data, testing_data)

    #models = cs.deploy_all_models(training_data_model)

    # answers_for_classifier = get_predictions_to_build_classifier(training_data_classifier)
    predictions0 = cs.predicts(testing_data.features)
    # print(predictions0)
