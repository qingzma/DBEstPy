import unittest
from context import creg


class AdvancedTestSuite(unittest.TestCase):
	""" Unit test for ClientClass.py"""
	def test_get_predictions(self):
		cc=creg.core.CRegression(logger_object=None, base_models=None, ensemble_models=None, classifier_type=creg.tools.classifier_xgboost_name, b_show_plot=False, b_disorder=False, b_select_classifier=False)
		self.assertIsNotNone(cc.app_names_deployed)


if __name__=="__main__":
	unittest.main()