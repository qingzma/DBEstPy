import unittest
from context import creg

class AdvancedTestSuite(unittest.TestCase):
	@unittest.SkipTest  #("included by test_density_estimation_plt_2d")
	def test_density_estimation_2d(self):
		logger = creg.logs.QueryLogs()
		logger.set_no_output();
		data = creg.data_loader.load2d(5)
		cRegression = creg.core.CRegression(logger_object=logger)
		cRegression.fit(data)
		qe=creg.query_engine.QueryEngine(cRegression)
		self.assertTrue(qe.density_estimation())
	def test_density_estimation_plt_2d(self):
		logger = creg.logs.QueryLogs()
		logger.set_no_output();
		data = creg.data_loader.load2d(5)
		cRegression = creg.core.CRegression(logger_object=logger)
		cRegression.fit(data)
		qe=creg.query_engine.QueryEngine(cRegression)
		qe.density_estimation()
		self.assertTrue(qe.desngity_estimation_plt2d)

if __name__=="__main__":
	unittest.main()
