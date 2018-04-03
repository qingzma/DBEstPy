#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function, division
from core import CRegression
import tools
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import evaluation
import logs
import data_loader as dl
from scipy import integrate
from datetime import datetime

epsabs = 1E-01
epsrel = 1E-03
mesh_grid_num = 30
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

    def __init__(self, cregression, logger_object=None, b_print_time_cost=True):
        self.num_training_points = cregression.num_total_training_points
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

        X1d = X.reshape(mesh_grid_num*mesh_grid_num)
        Y1d = Y.reshape(mesh_grid_num*mesh_grid_num)
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

    def approximate_avg_from_to(self, x_min, x_max, x_columnID):
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
                return np.exp(self.kde.score_samples(x))*self.cregression.predict(x)
            def f_p(x):
                return np.exp(self.kde.score_samples(x))
            a = integrate.quad(f_pRx,x_min,x_max,epsabs=epsabs,epsrel=epsrel)[0]
            b = integrate.quad(f_p,x_min,x_max,epsabs=epsabs,epsrel=epsrel)[0]

            if  b:
                result = a/b
            else:
                result =  None

        if self.dimension > 1:
            data_range_length_half = [(max(self.training_data.features[:, i])-min(
                self.training_data.features[:, i]))*0.5 for i in range(self.dimension)]
            data_range = [[min(self.training_data.features[:, i])-data_range_length_half[i], max(
                self.training_data.features[:, i])+data_range_length_half[i]] for i in range(self.dimension)]

            # generate the integral bounds
            bounds = []
            for i in range(x_columnID):
                bounds.append(data_range[i])
            bounds.append([x_min, x_max])
            # print(bounds)
            for i in range(x_columnID+1, self.dimension):
                bounds.append(data_range[i])

            def f_p(*args):
                # print(np.exp(self.kde.score_samples(np.array(args).reshape(1,-1))))
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))

            def f_pRx(*args):
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))\
                    * self.cregression.predict(np.array(args))
            a = integrate.nquad(f_pRx, bounds, opts=opts)[0]
            b = integrate.nquad(f_p, bounds, opts=opts)[0]

            if  b:
                result = a/b
            else:
                result =  None
        end = datetime.now()
        if self.b_print_time_cost:
            self.logger.info("Time spent for AVG: %.4fs." % (end - start).total_seconds())
        return result

    def approximate_sum_from_to(self, x_min, x_max, x_columnID):
        start = datetime.now()
        if self.dimension is 1:
            # average = self.approximate_ave_from_to(x_min,x_max,x_columnID)
            def f_pRx(*args):
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))\
                    * self.cregression.predict(np.array(args))
            result = integrate.quad(f_pRx,x_min,x_max,epsabs=epsabs,epsrel=epsrel)[0]*self.num_training_points
            # return result

        if self.dimension > 1:
            data_range_length_half = [(max(self.training_data.features[:, i])-min(
                self.training_data.features[:, i]))*0.5 for i in range(self.dimension)]
            data_range = [[min(self.training_data.features[:, i])-data_range_length_half[i], max(
                self.training_data.features[:, i])+data_range_length_half[i]] for i in range(self.dimension)]

            # generate the integral bounds
            bounds = []
            for i in range(x_columnID):
                bounds.append(data_range[i])
            bounds.append([x_min, x_max])
            # print(bounds)
            for i in range(x_columnID+1, self.dimension):
                bounds.append(data_range[i])

            def f_pRx(*args):
                return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))\
                    * self.cregression.predict(np.array(args))
            result = integrate.nquad(f_pRx, bounds, opts=opts)[0]*self.num_training_points
        end = datetime.now()
        if self.b_print_time_cost:
            self.logger.info("Time spent for SUM: %.4fs." % (end - start).total_seconds())
        return(result)




if __name__ == "__main__":
    logger = logs.QueryLogs()
    logger.set_no_output()
    data = dl.load2d(5)
    cRegression = CRegression(logger_object=logger)
    cRegression.fit(data)
    # cRegression.plot_training_data_3d()
    # exit(1)
    cRegression.plot_training_data_2d()
    logger.set_logging()
    qe = QueryEngine(cRegression, logger_object=logger)
    qe.density_estimation()
    qe.desngity_estimation_plt2d()
    # qe.desngity_estimation_plt3d()
    print(qe.approximate_avg_from_to(70, 80, 0))
    print(qe.approximate_sum_from_to(70, 80, 0))
    # qe.desngity_estimation_plt2d()
