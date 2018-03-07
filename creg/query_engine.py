#!/usr/bin/env python
# coding=utf-8
from core import CRegression
import tools
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import evaluation
import logs
import data_loader as dl


class QueryEngine:
	"""Built upon core, aimed to accomplish approximate query processing via regression"""
	def __init__(self,cregression):
		self.num_training_points = cregression.num_total_training_points
		self.log_dens=None
		self.training_data = cregression.training_data
		self.kde = None # kernel density object


	def density_estimation(self,kernel=None):
		"""Estimate the density of points.

		Args:
		    kernel (None, optional): Should be one of [‘gaussian’,’tophat’,’epanechnikov’,’exponential’,’linear’,’cosine’]
		    Default is ‘gaussian’.

		Returns:
		    TYPE: Description
		"""
		if  kernel is  None:
			kernel = 'gaussian'
		self.kde = KernelDensity(kernel=kernel).fit(self.training_data.features)
		return self.kde
	def desngity_estimation_plt2d(self):
		""" plot the density distribution

		Returns:
		    TYPE: Ture if the plot is shown
		"""
		X_plot = np.linspace(min(self.training_data.features),max(self.training_data.features),1000)[:,np.newaxis]
		self.log_dens = self.kde.score_samples(X_plot)

		ax = plt.subplot(111)

		ax.plot(X_plot[:, 0], np.exp(self.log_dens), '-')
		ax.plot(self.training_data.features,-0.001- 0.001 * np.random.random(self.training_data.features.shape[0]), '+k')
		ax.set_xlabel(self.training_data.headers[0])
		ax.set_ylabel(self.training_data.headers[1])
		ax.set_title("Density Estimation")
		plt.show()
		return True
	def desngity_estimation_plt3d(self):
		X_plot = np.linspace(min(self.training_data.features),max(self.training_data.features),1000)[:,np.newaxis]
		self.log_dens = self.kde.score_samples(X_plot)

		ax = plt.subplot(111)

		ax.plot(X_plot[:, 0], np.exp(self.log_dens), '-')
		ax.plot(self.training_data.features,-0.001- 0.001 * np.random.random(self.training_data.features.shape[0]), '+k')
		plt.show()
		return True






if __name__=="__main__":
	logger =  logs.QueryLogs()
	data = dl.load3d(5)
	cRegression = CRegression()
	cRegression.fit(data)
	qe=QueryEngine(cRegression)
	qe.density_estimation()
	# qe.desngity_estimation_plt2d()
	# Plot a 1D density example
	# N = 1000
	# np.random.seed(1)
	# X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
	#                     np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]

	# X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

	# true_dens = (0.3 * norm(0, 1).pdf(X_plot[:, 0])
	#              + 0.7 * norm(5, 1).pdf(X_plot[:, 0]))

	# fig, ax = plt.subplots()
	# ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2,
	#         label='input distribution')

	# for kernel in ['gaussian', 'tophat', 'epanechnikov']:
	#     kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
	#     log_dens = kde.score_samples(X_plot)
	#     ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
	#             label="kernel = '{0}'".format(kernel))

	# ax.text(6, 0.38, "N={0} points".format(N))

	# ax.legend(loc='upper left')
	# ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')

	# ax.set_xlim(-4, 9)
	# ax.set_ylim(-0.02, 0.4)
	# plt.show()
