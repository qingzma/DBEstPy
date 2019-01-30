from __future__ import print_function
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dbest import tools
from dbest import data_loader as dl
from dbest import logs
from dbest.qreg import CRegression





class Runner:
    def __init__(self,logger_object=None):

        self.num_datasetaset = 8

        if not logger_object:
            logger_object = logs.QueryLogs()
        self.logger = logger_object
        self.logger_name = logger_object.logger_name
        # else:
        #     self.logger = logging
        #     self.logger.basicConfig(level=logging.DEBUG,
        #                             format='%(levelname)s - %(message)s')
        #     self.logger_name = None



    # ----------------------------------------------------------------------------------------------------------------------#
    def run2d(self, dataID, base_models=None, ensemble_models=None, classifier_type=tools.classifier_xgboost_name, b_show_plot=False,b_disorder=False,b_select_classifier=False):
        data = dl.load2d(dataID)

        client = CRegression(logger_object=self.logger, base_models=base_models, ensemble_models=ensemble_models,
                             classifier_type=classifier_type, b_show_plot=b_show_plot,
                             b_select_classifier=b_select_classifier,b_disorder=b_disorder)

        client.run2d(data)
        return client

    def run3d(self, dataID, base_models=None, ensemble_models=None, classifier_type=tools.classifier_xgboost_name, b_show_plot=False,b_disorder=False,b_select_classifier=False):
        data = dt.load3d(dataID)


        client = CRegression(logger_object=self.logger, base_models=base_models, ensemble_models=ensemble_models,
                             classifier_type=classifier_type, b_show_plot=b_show_plot,
                             b_select_classifier=b_select_classifier,b_disorder=b_disorder)

        client.run3d(data)
        return client

    def run4d(self, dataID, base_models=None, ensemble_models=None, classifier_type=tools.classifier_xgboost_name, b_show_plot=False,b_disorder=False,b_select_classifier=False):
        data = dl.load4d(dataID)

        client = CRegression(logger_object=self.logger, base_models=base_models, ensemble_models=ensemble_models,
                             classifier_type=classifier_type, b_show_plot=b_show_plot,
                             b_select_classifier=b_select_classifier,b_disorder=b_disorder)



        client.run(data)
        return client

    def run5d(self, dataID, base_models=None, ensemble_models=None, classifier_type=tools.classifier_xgboost_name, b_show_plot=False, b_disorder=False,b_select_classifier=False):
        data = dl.load5d(dataID)

        client = CRegression(logger_object=self.logger, base_models=base_models, ensemble_models=ensemble_models,
                             classifier_type=classifier_type, b_show_plot=b_show_plot,
                             b_select_classifier=b_select_classifier,b_disorder=b_disorder)

        client.run(data)
        return client

    def runNd(self, dataID, base_models=None, ensemble_models=None, classifier_type=tools.classifier_xgboost_name, b_show_plot=False, b_disorder=False,b_select_classifier=False):
        data =dl.loadNd(dataID)

        client = CRegression(logger_object=self.logger, base_models=base_models, ensemble_models=ensemble_models,
                             classifier_type=classifier_type, b_show_plot=b_show_plot,
                             b_select_classifier=b_select_classifier,b_disorder=b_disorder)

        client.run(data)
        return client
        # ----------------------------------------------------------------------------------------------------------------------#

    # ----------------------------------------------------------------------------------------------------------------------#
    def run2d_all(self, base_models=None, ensemble_models=None,
                  classifier_type=tools.classifier_xgboost_name, b_show_plot=None, b_disorder=False,b_select_classifier=False):
        clients = []
        for i in range(self.num_dataset):
            client = self.run2d(i + 1, base_models=base_models, ensemble_models=ensemble_models,
                                classifier_type=classifier_type,
                                b_show_plot=b_show_plot,
                             b_select_classifier=b_select_classifier,b_disorder=b_disorder)
            clients.append(client)
        evaluation = tools.Evaluation(clients, logger_name)
        evaluation.print_summary()

    def run3d_all(self, base_models=None, ensemble_models=None,
                  classifier_type=tools.classifier_xgboost_name, b_show_plot=None, b_disorder=False,b_select_classifier=False):
        clients = []
        for i in range(self.num_dataset):
            client = self.run3d(i + 1, base_models=base_models, ensemble_models=ensemble_models,
                                classifier_type=classifier_type,
                                b_show_plot=b_show_plot,
                             b_select_classifier=b_select_classifier,b_disorder=b_disorder)
            clients.append(client)
        evaluation = tools.Evaluation(clients, logger_name)
        evaluation.print_summary()

    def run4d_all(self, base_models=None, ensemble_models=None,
                  classifier_type=tools.classifier_xgboost_name, b_show_plot=None, b_disorder=False,b_select_classifier=False):
        clients = []
        for i in range(self.num_dataset):
            client = self.run4d(i + 1, base_models=base_models, ensemble_models=ensemble_models,
                                classifier_type=classifier_type,
                                b_show_plot=b_show_plot,
                             b_select_classifier=b_select_classifier,b_disorder=b_disorder)
            clients.append(client)
        evaluation = tools.Evaluation(clients, logger_name)
        evaluation.print_summary()

    def run5d_all(self, base_models=None, ensemble_models=None,
                  classifier_type=tools.classifier_xgboost_name, b_show_plot=None, b_disorder=False,b_select_classifier=False):
        clients = []
        for i in range(self.num_dataset):
            client = self.run5d(i + 1, base_models=base_models, ensemble_models=ensemble_models,
                                classifier_type=classifier_type,
                                b_show_plot=b_show_plot,
                             b_select_classifier=b_select_classifier,b_disorder=b_disorder)
            clients.append(client)
        evaluation = tools.Evaluation(clients, logger_name)
        evaluation.print_summary()

    def runNd_all(self, base_models=None, ensemble_models=None,
                  classifier_type=tools.classifier_xgboost_name, b_show_plot=None, b_disorder=False,b_select_classifier=False):
        clients = []
        for i in range(self.num_dataset):
            client = self.run2d(i + 1, base_models=base_models, ensemble_models=ensemble_models,
                                classifier_type=classifier_type,
                                b_show_plot=b_show_plot,
                             b_select_classifier=b_select_classifier,b_disorder=b_disorder)
            clients.append(client)
        evaluation = tools.Evaluation(clients, logger_name)
        evaluation.print_summary()

    # ----------------------------------------------------------------------------------------------------------------------#
    def run2d_linear(self,base_models,ensemble_models):
        # 2D linear
        runner.set_logging('2d_linear')
        # base_models = [tools.app_xgboost, tools.app_boosting]
        # ensemble_models = [tools.app_xgboost]
        classifier_type = tools.classifier_linear_name
        runner.run2d_all(base_models=base_models, ensemble_models=ensemble_models, classifier_type=classifier_type)

    def run2d_xgb(self,base_models,ensemble_models):
        # 2D xgb
        runner.set_logging('2d_xgb')
        # base_models = [tools.app_linear, tools.app_poly, tools.app_decision_tree]
        # ensemble_models = [tools.app_adaboost, tools.app_boosting, tools.app_xgboost]
        classifier_type = tools.classifier_xgboost_name
        runner.run2d_all(base_models=base_models, ensemble_models=ensemble_models, classifier_type=classifier_type)

    def run2d_xgb_base_model(self,base_models,ensemble_models):
        # 2D xgb, using xgb boost as base model
        runner.set_logging('2d_xgb_base_model')
        # base_models = [tools.app_linear, tools.app_poly, tools.app_xgboost]
        # ensemble_models = [tools.app_adaboost, tools.app_boosting]
        classifier_type = tools.classifier_xgboost_name
        runner.run2d_all(base_models=base_models, ensemble_models=ensemble_models, classifier_type=classifier_type)

    # ----------------------------------------------------------------------------------------------------------------------#
    def run3d_linear(self,base_models,ensemble_models):
        # 2D linear
        runner.set_logging('3d_linear')
        # base_models = [tools.app_linear, tools.app_poly, tools.app_decision_tree]
        # ensemble_models = [tools.app_adaboost, tools.app_boosting, tools.app_xgboost]
        classifier_type = tools.classifier_linear_name
        runner.run3d_all(base_models=base_models, ensemble_models=ensemble_models, classifier_type=classifier_type)

    def run3d_xgb(self,base_models,ensemble_models):
        # 2D xgb
        runner.set_logging('3d_xgb')
        # base_models = [tools.app_linear, tools.app_poly, tools.app_decision_tree]
        # ensemble_models = [tools.app_adaboost, tools.app_boosting, tools.app_xgboost]
        classifier_type = tools.classifier_xgboost_name
        runner.run3d_all(base_models=base_models, ensemble_models=ensemble_models, classifier_type=classifier_type)

    def run3d_xgb_base_model(self,base_models,ensemble_models):
        # 2D xgb, using xgb boost as base model
        runner.set_logging('3d_xgb_base_model')
        # base_models = [tools.app_linear, tools.app_poly, tools.app_xgboost]
        # ensemble_models = [tools.app_adaboost, tools.app_boosting]
        classifier_type = tools.classifier_xgboost_name
        runner.run3d_all(base_models=base_models, ensemble_models=ensemble_models, classifier_type=classifier_type)

    # ----------------------------------------------------------------------------------------------------------------------#
    def run4d_linear(self,base_models,ensemble_models):
        # 2D linear
        runner.set_logging('4d_linear')
        # base_models = [tools.app_linear, tools.app_poly, tools.app_decision_tree]
        # ensemble_models = [tools.app_adaboost, tools.app_boosting, tools.app_xgboost]
        classifier_type = tools.classifier_linear_name
        runner.run4d_all(base_models=base_models, ensemble_models=ensemble_models, classifier_type=classifier_type)

    def run4d_xgb(self,base_models,ensemble_models):
        # 2D xgb
        runner.set_logging('4d_xgb')
        # base_models = [tools.app_linear, tools.app_poly, tools.app_decision_tree]
        # ensemble_models = [tools.app_adaboost, tools.app_boosting, tools.app_xgboost]
        classifier_type = tools.classifier_xgboost_name
        runner.run4d_all(base_models=base_models, ensemble_models=ensemble_models, classifier_type=classifier_type)

    def run4d_xgb_base_model(self,base_models,ensemble_models):
        # 2D xgb, using xgb boost as base model
        runner.set_logging('4d_xgb_base_model')
        # base_models = [tools.app_linear, tools.app_poly, tools.app_xgboost]
        # ensemble_models = [tools.app_adaboost, tools.app_boosting]
        classifier_type = tools.classifier_xgboost_name
        runner.run4d_all(base_models=base_models, ensemble_models=ensemble_models, classifier_type=classifier_type)

    # ----------------------------------------------------------------------------------------------------------------------#
    def runNd_linear(self,base_models,ensemble_models):
        # 2D linear
        runner.set_logging('Nd_linear')
        # base_models = [tools.app_linear, tools.app_poly, tools.app_decision_tree]
        # ensemble_models = [tools.app_adaboost, tools.app_boosting, tools.app_xgboost]
        classifier_type = tools.classifier_linear_name
        runner.runNd_all(base_models=base_models, ensemble_models=ensemble_models, classifier_type=classifier_type)

    def runNd_xgb(self,base_models,ensemble_models):
        # 2D xgb
        runner.set_logging('Nd_xgb')
        # base_models = [tools.app_linear, tools.app_poly, tools.app_decision_tree]
        # ensemble_models = [tools.app_adaboost, tools.app_boosting, tools.app_xgboost]
        classifier_type = tools.classifier_xgboost_name
        runner.runNd_all(base_models=base_models, ensemble_models=ensemble_models, classifier_type=classifier_type)

    def runNd_xgb_base_model(self,base_models,ensemble_models):
        # 2D xgb, using xgb boost as base model
        runner.set_logging('Nd_xgb_base_model')
        # base_models = [tools.app_linear, tools.app_poly, tools.app_xgboost]
        # ensemble_models = [tools.app_adaboost, tools.app_boosting]
        classifier_type = tools.classifier_xgboost_name
        runner.runNd_all(base_models=base_models, ensemble_models=ensemble_models, classifier_type=classifier_type)

    # ----------------------------------------------------------------------------------------------------------------------#
    def run5d_linear(self,base_models,ensemble_models):
        # 2D linear
        runner.set_logging('5d_linear')
        # base_models = [tools.app_linear, tools.app_poly, tools.app_decision_tree]
        # ensemble_models = [tools.app_adaboost, tools.app_boosting, tools.app_xgboost]
        classifier_type = tools.classifier_linear_name
        runner.run5d_all(base_models=base_models, ensemble_models=ensemble_models, classifier_type=classifier_type)

    def run5d_xgb(self,base_models,ensemble_models):
        # 2D xgb
        runner.set_logging('5d_xgb')
        # base_models = [tools.app_linear, tools.app_poly, tools.app_decision_tree]
        # ensemble_models = [tools.app_adaboost, tools.app_boosting, tools.app_xgboost]
        classifier_type = tools.classifier_xgboost_name
        runner.run5d_all(base_models=base_models, ensemble_models=ensemble_models, classifier_type=classifier_type)

    def run5d_xgb_base_model(self,base_models,ensemble_models):
        # 2D xgb, using xgb boost as base model
        runner.set_logging('5d_xgb_base_model')
        # base_models = [tools.app_linear, tools.app_poly, tools.app_xgboost]
        # ensemble_models = [tools.app_adaboost, tools.app_boosting]
        classifier_type = tools.classifier_xgboost_name
        runner.run5d_all(base_models=base_models, ensemble_models=ensemble_models, classifier_type=classifier_type)

    def evaluate(self,base_models,ensemble_models):

        self.run2d_linear(base_models,ensemble_models)
        self.run2d_xgb(base_models,ensemble_models)
        self.run2d_xgb_base_model(base_models,ensemble_models)

        self.run3d_linear(base_models,ensemble_models)
        self.run3d_xgb(base_models,ensemble_models)
        self.run3d_xgb_base_model(base_models,ensemble_models)

        self.run4d_linear(base_models,ensemble_models)
        self.run4d_xgb(base_models,ensemble_models)
        self.run4d_xgb_base_model(base_models,ensemble_models)

        self.run5d_linear(base_models, ensemble_models)
        self.run5d_xgb(base_models, ensemble_models)
        self.run5d_xgb_base_model(base_models, ensemble_models)



if __name__ == "__main__":
    runner = Runner()
    #runner.run3d_linear()

    # base_models = [tools.app_xgboost, tools.app_boosting]
    # ensemble_models = [tools.app_xgboost]
    # runner.evaluate(base_models,ensemble_models)


    # base_models = [tools.app_boosting,tools.app_xgboost]#,tools.app_decision_tree]
    base_models = [tools.app_boosting, tools.app_xgboost]
    ensemble_models = [tools.app_xgboost]
    runner.run2d(5,base_models,ensemble_models, tools.classifier_xgboost_name)

    #runner.run2d_all(base_models,ensemble_models,classifier_type=tools.classifier_rbf_name,b_show_plot=False)
    #runner.run3d_xgb(base_models,ensemble_models)


