from unittest import TestCase

import dbest

class TestDBEst(TestCase):
	def test_is_string(self):
		s="123"
		self.assertEqual(s,"123")