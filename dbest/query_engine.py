#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function, division
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dbest.qreg import CRegression

from dbest import logs
from dbest import data_loader as dl
from dbest import generate_random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy import integrate
from datetime import datetime
import warnings
import gc
import pickle


epsabs = 10         #1E-3
epsrel = 1E-01      #1E-1
mesh_grid_num = 20  #30
limit =30
opts = {'epsabs': epsabs, 'epsrel': epsrel, 'limit': 100}
# opts = {'epsabs': 1.49e-03, 'epsrel': 1.49e-03, 'limit': 100}

# variable_names=[a1,b1,c1,d1,e1,f1,g1,h1,i1,j1,k1,l1,m1,n1,o1,p1,q1,r1,s1,t1,u1,v1,w1,x1,y1,z1,
# a2,b2,c2,d2,e2,f2,g2,h2,i2,j2,k2,l2,m2,n2,o2,p2,q2,r2,s2,t2,u2,v2,w2,x2,y2,z2]


class QueryEngine:

    """Summary

    Attributes:
        cregression (TYPE): Description
        dimension (TYPE): Description
        kde (TYPE): Description
        log_dens (TYPE): Description
        logger (TYPE): Description
        num_training_points (TYPE): Description
        training_data (TYPE): Description
    """

    def __init__(self, cregression, logger_object=None, b_print_time_cost=True, num_training_points=None):
        if num_training_points is None:
            self.num_training_points = cregression.num_total_training_points
        else:
            self.num_training_points = num_training_points
        self.log_dens = None
        self.training_data = cregression.training_data
        self.kde = None  # kernel density object
        self.dimension = self.training_data.features.shape[1]
        self.cregression = cregression
        if logger_object:
            self.logger = logger_object.logger
        else:
            self.logger = logs.QueryLogs().logger
        self.b_print_time_cost = b_print_time_cost
        self.__sizeof__=None
        self.x_min = min(self.training_data.features)
        self.x_max = max(self.training_data.features)

        # warnings.filterwarnings(action='ignore', category=DeprecationWarning)

    def density_estimation(self, kernel=None):
        """Estimate the density of points.

        Args:
            kernel (None, optional): Should be one of [‘gaussian’,’tophat’,’epanechnikov’,’exponential’,’linear’,’cosine’]
            Default is ‘gaussian’.

        Returns:
            TYPE: Description
        """
        if kernel is None:
            kernel = 'gaussian'
        self.kde = KernelDensity(kernel=kernel).fit(
            self.training_data.features)
        # remove unecessary memory usage
        del self.training_data
        # self.cregression.clear_training_data()
        
        gc.collect()
        return self.kde

    def desngity_estimation_plt2d(self):
        """ plot the density distribution

        Returns:
            TYPE: Ture if the plot is shown
        """
        X_plot = np.linspace(min(self.training_data.features), max(
            self.training_data.features), mesh_grid_num)[:, np.newaxis]
        self.log_dens = self.kde.score_samples(X_plot)

        ax = plt.subplot(111)

        ax.plot(X_plot[:, 0], np.exp(self.log_dens), '-')
        ax.plot(self.training_data.features, -0.001 - 0.001 *
                np.random.random(self.training_data.features.shape[0]), '+k')
        ax.set_xlabel(self.training_data.headers[0])
        ax.set_ylabel("Probability")
        ax.set_title("2D Density Estimation")
        plt.show()
        return True

    def desngity_estimation_plt3d(self):
        x = np.linspace(min(self.training_data.features[:, 0]), max(
            self.training_data.features[:, 0]), mesh_grid_num)
        y = np.linspace(min(self.training_data.features[:, 1]), max(
            self.training_data.features[:, 1]), mesh_grid_num)
        X, Y = np.meshgrid(x, y)

        X1d = X.reshape(mesh_grid_num * mesh_grid_num)
        Y1d = Y.reshape(mesh_grid_num * mesh_grid_num)
        Z1d = [[X1d[i], Y1d[i]] for i in range(len(X1d))]
        Z = self.kde.score_samples(Z1d)
        Z_plot = Z.reshape((mesh_grid_num, mesh_grid_num))
        # print(Z_plot)

        self.log_dens = Z_plot

        ax = plt.axes(projection='3d')

        ax.plot_surface(X, Y, np.exp(Z_plot), rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none', alpha=0.8)
        # ax.plot(self.training_data.features,-0.001- 0.001 * np.random.random(self.training_data.features.shape[0]), '+k')
        ax.scatter(self.training_data.features[:, 0], self.training_data.features[:, 1], -0.0005 -
                   0.0001 * np.random.random(self.training_data.features[:, 0].shape[0]), '+k', alpha=0.8, s=1)
        ax.set_title("3D Density Estimation")
        ax.set_xlabel(self.training_data.headers[0])
        ax.set_ylabel(self.training_data.headers[1])
        ax.set_zlabel("Probability")
        plt.show()
        return True

    def approximate_avg_from_to(self, x_min, x_max, x_columnID,epsabs=epsabs, epsrel=epsrel,limit=limit):
        """ calculate the approximate average value between x_min and x_max

        Args:
            x_min (TYPE): lower bound
            x_max (TYPE): upper bound
            x_columnID (TYPE): the index of the x to be interated

        Returns:
            TYPE: the integeral value
        """
        start = datetime.now()
        if self.dimension is 1:
            def f_pRx(x):
                # print(self.cregression.predict(x))
                return np.exp(self.kde.score_samples(x)) * self.cregression.predict(x)

            def f_p(x):
                return np.exp(self.kde.score_samples(x))
            a = integrate.quad(f_pRx, x_min, x_max,
                               epsabs=epsabs, epsrel=epsrel)[0]
            b = integrate.quad(f_p, x_min, x_max,
                               epsabs=epsabs, epsrel=epsrel)[0]

            if b:
                result = a / b
            else:
                result = None

        if self.dimension > 1:
            data_range_length_half = [(max(self.training_data.features[:, i]) - min(
                self.training_data.features[:, i])) * 0.5 for i in range(self.dimension)]
            data_range = [[min(self.training_data.features[:, i]) - data_range_length_half[i], max(
                self.training_data.features[:, i]) + data_range_length_half[i]] for i in range(self.dimension)]

            # generate the integral bounds
            bounds = []
            for i in range(x_columnID):
                bounds.append(data_range[i])
            bounds.append([x_min, x_max])
            # print(bounds)
            for i in range(x_columnID + 1, self.dimension):
                bounds.append(data_range[i])

            def f_p(*args):
                # print(np.exp(self.kde.score_samples(np.array(args).reshape(1,-1))))
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))

            def f_pRx(*args):
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))\
                    * self.cregression.predict(np.array(args))
            a = integrate.nquad(f_pRx, bounds, opts=opts)[0]
            b = integrate.nquad(f_p, bounds, opts=opts)[0]

            if b:
                result = a / b
            else:
                result = None
        end = datetime.now()
        time_cost = (end - start).total_seconds()
        if self.b_print_time_cost and result != None:
            self.logger.info("Approximate AVG: %.4f." % result)
            self.logger.info(
                "Time spent for approximate AVG: %.4fs." % time_cost)
        return result, time_cost

    def approximate_sum_from_to(self, x_min, x_max, x_columnID,epsabs=epsabs, epsrel=epsrel,limit=limit):
        start = datetime.now()
        if self.dimension is 1:
            # average = self.approximate_ave_from_to(x_min,x_max,x_columnID)
            def f_pRx(*args):
                # print("integral is "+str(self.cregression.predict(np.array(args))))
                # print("points is" + str(self.num_training_points))
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))\
                    * self.cregression.predict(np.array(args))
            # print(integrate.quad(f_pRx, x_min, x_max, epsabs=epsabs, epsrel=epsrel)[0])
            result = integrate.quad(f_pRx, x_min, x_max, epsabs=epsabs, epsrel=epsrel)[0] * float(self.num_training_points)
            # return result

        if self.dimension > 1:
            data_range_length_half = [(max(self.training_data.features[:, i]) - min(
                self.training_data.features[:, i])) * 0.5 for i in range(self.dimension)]
            data_range = [[min(self.training_data.features[:, i]) - data_range_length_half[i], max(
                self.training_data.features[:, i]) + data_range_length_half[i]] for i in range(self.dimension)]

            # generate the integral bounds
            bounds = []
            for i in range(x_columnID):
                bounds.append(data_range[i])
            bounds.append([x_min, x_max])
            # print(bounds)
            for i in range(x_columnID + 1, self.dimension):
                bounds.append(data_range[i])

            def f_pRx(*args):
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))\
                    * self.cregression.predict(np.array(args))
            result = integrate.nquad(f_pRx, bounds, opts=opts)[
                0] * self.num_training_points
        end = datetime.now()
        time_cost = (end - start).total_seconds()
        if self.b_print_time_cost and result != None:
            self.logger.info("Approximate SUM: %.4f." % result)
            self.logger.info(
                "Time spent for approximate SUM: %.4fs." % time_cost)
        return result, time_cost

    # def approximate_avgx_from_to(self, x_min, x_max, x_columnID):
    #     """ calculate the approximate average value between x_min and x_max

    #     Args:
    #         x_min (TYPE): lower bound
    #         x_max (TYPE): upper bound
    #         x_columnID (TYPE): the index of the x to be interated

    #     Returns:
    #         TYPE: the integeral value
    #     """
    #     start = datetime.now()
    #     if self.dimension is 1:
    #         def f_pRx(x):
    #             # print(self.cregression.predict(x))
    #             return np.exp(self.kde.score_samples(x)) * x

    #         def f_p(x):
    #             return np.exp(self.kde.score_samples(x))
    #         a = integrate.quad(f_pRx, x_min, x_max,
    #                            epsabs=epsabs, epsrel=epsrel)[0]
    #         b = integrate.quad(f_p, x_min, x_max,
    #                            epsabs=epsabs, epsrel=epsrel)[0]

    #         if b:
    #             result = a / b
    #         else:
    #             result = None
    #     end = datetime.now()
    #     time_cost = (end - start).total_seconds()
    #     if self.b_print_time_cost:
    #         self.logger.info("Approximate AVG: %.4f." % result)
    #         self.logger.info(
    #             "Time spent for approximate AVG: %.4fs." % time_cost)
    #     return result, time_cost

    # def approximate_sumx_from_to(self, x_min, x_max, x_columnID):
    #     start = datetime.now()
    #     if self.dimension is 1:
    #         # average = self.approximate_ave_from_to(x_min,x_max,x_columnID)
    #         def f_pRx(*args):
    #             return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))\
    #                 * np.array(args)[0]
    #         result = integrate.quad(f_pRx, x_min, x_max, epsabs=epsabs, epsrel=epsrel)[
    #             0] * self.num_training_points
    #         # return result

        
    #     time_cost = (end - start).total_seconds()
    #     if self.b_print_time_cost:
    #         self.logger.info("Approximate SUM: %.4f." % result)
    #         self.logger.info(
    #             "Time spent for approximate SUM: %.4fs." % time_cost)
    #     return result, time_cost

    def approximate_count_from_to(self, x_min, x_max, x_columnID,epsabs=epsabs, epsrel=epsrel,limit=limit):
        start = datetime.now()
        if self.dimension is 1:
            # average = self.approximate_ave_from_to(x_min,x_max,x_columnID)
            def f_p(*args):
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))
            result = integrate.quad(f_p, x_min, x_max, epsabs=epsabs, epsrel=epsrel)[0]
            # print("result is "+ str(result))
            result = result * float(self.num_training_points)
            # return result

        if self.dimension > 1:
            data_range_length_half = [(max(self.training_data.features[:, i]) - min(
                self.training_data.features[:, i])) * 0.5 for i in range(self.dimension)]
            data_range = [[min(self.training_data.features[:, i]) - data_range_length_half[i], max(
                self.training_data.features[:, i]) + data_range_length_half[i]] for i in range(self.dimension)]

            # generate the integral bounds
            bounds = []
            for i in range(x_columnID):
                bounds.append(data_range[i])
            bounds.append([x_min, x_max])
            # print(bounds)
            for i in range(x_columnID + 1, self.dimension):
                bounds.append(data_range[i])

            def f_p(*args):
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))
            result = integrate.nquad(f_p, bounds, opts=opts)[
                0] * self.num_training_points
        end = datetime.now()
        time_cost = (end - start).total_seconds()
        if self.b_print_time_cost and result != None:
            self.logger.info("Approximate count: %.4f." % result)
            self.logger.info(
                "Time spent for approximate COUNT: %.4fs." % time_cost)
        return int(result), time_cost

    def approximate_variance_x_from_to(self, x_min=-np.inf, x_max=np.inf, x_columnID=0,epsabs=epsabs, epsrel=epsrel,limit=limit):
        start = datetime.now()
        if self.dimension is 1:
            # average = self.approximate_ave_from_to(x_min,x_max,x_columnID)
            def f_p(*args):
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))

            def f_x2Px(*args):
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1))) * np.array(args)[0]**2

            def f_xPx(*args):
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1))) * np.array(args)[0]

            def f_xRxPx(*args):
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))\
                    * self.cregression.predict(np.array(args)) * np.array(args)[0]
            # result = integrate.quad(f_p,x_min,x_max,epsabs=epsabs,epsrel=epsrel)[0]*self.num_training_points
            Ep = integrate.quad(f_p, x_min, x_max,
                                epsabs=epsabs, epsrel=epsrel)[0]
            Ex2 = integrate.quad(f_x2Px, x_min, x_max,
                                 epsabs=epsabs, epsrel=epsrel)[0]
            Ex_2 = integrate.quad(f_xPx, x_min, x_max,
                                  epsabs=epsabs, epsrel=epsrel)[0] ** 2
            result = Ex2 / Ep - Ex_2 / Ep / Ep

        # if self.dimension > 1:
        #     data_range_length_half = [(max(self.training_data.features[:, i])-min(
        #         self.training_data.features[:, i]))*0.5 for i in range(self.dimension)]
        #     data_range = [[min(self.training_data.features[:, i])-data_range_length_half[i], max(
        # self.training_data.features[:, i])+data_range_length_half[i]] for i
        # in range(self.dimension)]

        #     # generate the integral bounds
        #     bounds = []
        #     for i in range(x_columnID):
        #         bounds.append(data_range[i])
        #     bounds.append([x_min, x_max])
        #     # print(bounds)
        #     for i in range(x_columnID+1, self.dimension):
        #         bounds.append(data_range[i])

        #     def f_p(*args):
        #         return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))
        #     result = integrate.nquad(f_p, bounds, opts=opts)[0]*self.num_training_points
        end = datetime.now()
        time_cost = (end - start).total_seconds()
        if self.b_print_time_cost and result != None:
            self.logger.info("Approximate variance x: %.4f." % result)
            self.logger.info(
                "Time spent for approximate variance x: %.4fs." % time_cost)
        return result, time_cost

    def approximate_variance_y_from_to(self, x_min=-np.inf, x_max=np.inf, x_columnID=0,epsabs=epsabs, epsrel=epsrel,limit=limit):
        start = datetime.now()
        if self.dimension is 1:
            # average = self.approximate_ave_from_to(x_min,x_max,x_columnID)
            def f_p(*args):
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))

            def f_R2Px(*args):
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1))) * self.cregression.predict(np.array(args))**2

            def f_RxPx(*args):
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1))) * self.cregression.predict(np.array(args))
            # result = integrate.quad(f_p,x_min,x_max,epsabs=epsabs,epsrel=epsrel)[0]*self.num_training_points
            Ep = integrate.quad(f_p, x_min, x_max,
                                epsabs=epsabs, epsrel=epsrel)[0]
            Ey2 = integrate.quad(f_R2Px, x_min, x_max,
                                 epsabs=epsabs, epsrel=epsrel)[0]
            Ey_2 = integrate.quad(f_RxPx, x_min, x_max,
                                  epsabs=epsabs, epsrel=epsrel)[0] ** 2
            result = Ey2 / Ep - Ey_2 / Ep / Ep

        # if self.dimension > 1:
        #     data_range_length_half = [(max(self.training_data.features[:, i])-min(
        #         self.training_data.features[:, i]))*0.5 for i in range(self.dimension)]
        #     data_range = [[min(self.training_data.features[:, i])-data_range_length_half[i], max(
        # self.training_data.features[:, i])+data_range_length_half[i]] for i
        # in range(self.dimension)]

        #     # generate the integral bounds
        #     bounds = []
        #     for i in range(x_columnID):
        #         bounds.append(data_range[i])
        #     bounds.append([x_min, x_max])
        #     # print(bounds)
        #     for i in range(x_columnID+1, self.dimension):
        #         bounds.append(data_range[i])

        #     def f_p(*args):
        #         return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))
        #     result = integrate.nquad(f_p, bounds, opts=opts)[0]*self.num_training_points
        end = datetime.now()
        time_cost = (end - start).total_seconds()
        if self.b_print_time_cost and result != None:
            self.logger.info("Approximate variance y: %.4f." % result)
            self.logger.info(
                "Time spent for approximate variance y: %.4fs." % time_cost)
            if result < 0:
                self.logger.warning(
                    "Negtive approximate variance y: %.4f. is predicted..." % result)
        return result, time_cost

    def approximate_covar_from_to(self, x_min=-np.inf, x_max=np.inf, x_columnID=0,epsabs=epsabs, epsrel=epsrel,limit=limit):
        start = datetime.now()
        if self.dimension is 1:
            # average = self.approximate_ave_from_to(x_min,x_max,x_columnID)
            def f_p(*args):
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))

            def f_px(*args):
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1))) * np.array(args)[0]

            def f_pRx(*args):
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))\
                    * self.cregression.predict(np.array(args))

            def f_xRxPx(*args):
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))\
                    * self.cregression.predict(np.array(args)) * np.array(args)[0]
            # result = integrate.quad(f_p,x_min,x_max,epsabs=epsabs,epsrel=epsrel)[0]*self.num_training_points
            Ep = integrate.quad(f_p, x_min, x_max,
                                epsabs=epsabs, epsrel=epsrel)[0]
            ExPx = integrate.quad(f_px, x_min, x_max,
                                  epsabs=epsabs, epsrel=epsrel)[0]
            ERxPx = integrate.quad(f_pRx, x_min, x_max,
                                   epsabs=epsabs, epsrel=epsrel)[0]
            ExRxPx = integrate.quad(
                f_xRxPx, x_min, x_max, epsabs=epsabs, epsrel=epsrel)[0]
            result = ExRxPx / Ep - ExPx * ERxPx / Ep / Ep

        # if self.dimension > 1:
        #     data_range_length_half = [(max(self.training_data.features[:, i])-min(
        #         self.training_data.features[:, i]))*0.5 for i in range(self.dimension)]
        #     data_range = [[min(self.training_data.features[:, i])-data_range_length_half[i], max(
        # self.training_data.features[:, i])+data_range_length_half[i]] for i
        # in range(self.dimension)]

        #     # generate the integral bounds
        #     bounds = []
        #     for i in range(x_columnID):
        #         bounds.append(data_range[i])
        #     bounds.append([x_min, x_max])
        #     # print(bounds)
        #     for i in range(x_columnID+1, self.dimension):
        #         bounds.append(data_range[i])

        #     def f_p(*args):
        #         return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))
        #     result = integrate.nquad(f_p, bounds, opts=opts)[0]*self.num_training_points
        end = datetime.now()
        time_cost = (end - start).total_seconds()
        if self.b_print_time_cost and result != None:
            self.logger.info("Approximate COVAR: %.4f." % result)
            self.logger.info(
                "Time spent for approximate COVAR: %.4fs." % time_cost)
        return result, time_cost

    def approximate_corr_from_to(self, x_min=-np.inf, x_max=np.inf, x_columnID=0):
        start = datetime.now()
        tmp_b = self.b_print_time_cost
        self.b_print_time_cost = False
        if self.dimension is 1:
            var_x, _ = self.approximate_variance_x_from_to(
                x_min=x_min, x_max=x_max, x_columnID=1)
            var_y, _ = self.approximate_variance_y_from_to(
                x_min=x_min, x_max=x_max, x_columnID=1)
            if (var_x >= 0) and (var_y >= 0):
                var_x = var_x**0.5
                var_y = var_y**0.5
                result = self.approximate_covar_from_to(
                    x_min=x_min, x_max=x_max, x_columnID=1)[0] / var_x / var_y
                self.logger.info("Approximate CORR: %.4f." % result)
            else:
                result = None
                self.logger.warning(
                    "Cant be divided by zero! see Function approximate_corr_from_to()")

        end = datetime.now()
        time_cost = (end - start).total_seconds()
        self.b_print_time_cost = tmp_b
        if self.b_print_time_cost:
            self.logger.info("Approximate CORR: %.4f." % result)
            self.logger.info(
                "Time spent for approximate CORR: %.4fs." % time_cost)
        return result, time_cost

    def approximate_percentile_from_to(self, p, x_columnID=0,q_min_boundary=None, q_max_boundary=None ):
        start = datetime.now()
        if q_min_boundary is None:
            q_min_boundary=self.x_min
        if q_max_boundary is None:
            q_max_boundary=self.x_max
        # self.logger.info("min: "+str(q_min_boundary))
        # self.logger.info("max: "+str(q_max_boundary))
        extra = 0.05*(q_max_boundary-q_min_boundary)
        q_min_boundary=q_min_boundary-extra
        q_max_boundary=q_max_boundary+extra
        result = generate_random.percentile(
            p, self.kde, q_min_boundary, q_max_boundary, steps=50, n_bisect=30)
        end = datetime.now()
        time_cost = (end - start).total_seconds()
        if self.b_print_time_cost:
            self.logger.info("Approximate PERCENTILE: %.4f." % result[0])
            self.logger.info(
                "Time spent for approximate PERCENTILE: %.4fs." % time_cost)
        return result[0], time_cost

    def approximate_min_from_to(self,  q_min_boundary, q_max_boundary, x_columnID=0,steps=100,
        ci=True,confidence=0.95):
        """Summary
        
        Args:
            q_min_boundary (TYPE): Description
            q_max_boundary (TYPE): Description
            x_columnID (int, optional): Description
            steps (int, optional): The number of divisions in the mesh
        
        Returns:
            TYPE: Description
        """
        start = datetime.now()
        step = (q_max_boundary - q_min_boundary)/steps
        predictions = []
        for i in range(steps + 1):
            if ci:
                var= self.cregression.CI( np.array(q_min_boundary + i * step),confidence=confidence)
                # print(var)
                predictions.append(self.cregression.predict(np.array(q_min_boundary + i * step))
                    - var)
            else:
                predictions.append(self.cregression.predict(np.array(q_min_boundary + i * step)))
        # print(predictions)
        end = datetime.now()
        time_cost = (end - start).total_seconds()
        if self.b_print_time_cost:
            self.logger.info("Approximate MIN: %.4f." % min(predictions))
            self.logger.info(
                "Time spent for approximate MIN: %.4fs." % time_cost)
        return min(predictions), time_cost

    def approximate_max_from_to(self,  q_min_boundary, q_max_boundary, x_columnID=0,steps=100,
        ci=True,confidence=0.95):
        """Summary
        
        Args:
            q_min_boundary (TYPE): Description
            q_max_boundary (TYPE): Description
            x_columnID (int, optional): Description
            steps (int, optional): The number of divisions in the mesh
        
        Returns:
            TYPE: Description
        """
        start = datetime.now()
        step = (q_max_boundary - q_min_boundary)/steps
        predictions = []
        for i in range(steps + 1):
            if ci:
                var= self.cregression.CI( np.array(q_min_boundary + i * step),confidence=confidence)
                # print(var)
                predictions.append(self.cregression.predict(np.array(q_min_boundary + i * step))
                    + var )
            else:
                predictions.append(self.cregression.predict(np.array(q_min_boundary + i * step)))

        # print(predictions)
        end = datetime.now()
        time_cost = (end - start).total_seconds()
        if self.b_print_time_cost:
            self.logger.info("Approximate MAX: %.4f." % max(predictions))
            self.logger.info(
                "Time spent for approximate MAX: %.4fs." % time_cost)
        return max(predictions), time_cost

    def get_size(self):
        # str_size=dill.dumps(self)
        # self.__sizeof__ = sys.getsizeof(str_size)
        # gc.collect()
        # return self.__sizeof__
        str_size=pickle.dumps(self)
        self.__sizeof__ = sys.getsizeof(str_size)
        gc.collect()
        return self.__sizeof__

    def set_name(name):
        self.name = name

    def get_name(self):
        if hasattr(self, 'name'):
            return self.name
        else:
            return None


if __name__ == "__main__":
    import generate_random
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)

    logger = logs.QueryLogs()
    logger.set_no_output()
    data = dl.load2d(5)
    cRegression = CRegression(logger_object=logger)
    cRegression.fit(data)

    # cRegression.plot_training_data_2d()
    logger.set_logging()
    qe = QueryEngine(cRegression, logger_object=logger)
    qe.density_estimation()
    # qe.desngity_estimation_plt2d()

    # r = generate_random.percentile(
    #     0.9, qe.kde, 30, 100, steps=200, n_bisect=100)
    # print(r)
    # plt.plot(r)
    # plt.show()

    # qe.desngity_estimation_plt3d()
    # qe.approximate_avg_from_to(70, 80, 0)[0]
    # qe.approximate_sum_from_to(70, 80, 0)[0]
    # qe.approximate_count_from_to(70, 80, 0)[0]
    # qe.approximate_variance_x_from_to(70, 80, 0)[0]
    # qe.approximate_variance_y_from_to(70, 80, 0)[0]
    # qe.approximate_covar_from_to(70, 80, 0)[0]
    #
    # qe.approximate_avg_from_to(0.6, 0.8, 0)[0]
    # qe.approximate_sum_from_to(0.6, 0.8, 0)[0]
    # qe.approximate_count_from_to(0.6, 0.8, 0)[0]
    # qe.approximate_variance_x_from_to(0.6, 0.8, 0)[0]
    # qe.approximate_variance_y_from_to(0.6, 0.8, 0)[0]
    # qe.approximate_covar_from_to(0.6, 0.8, 0)[0]
    print(qe.approximate_max_from_to(50, 70, 0)[0])

    # qe.desngity_estimation_plt2d()
