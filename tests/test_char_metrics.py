import json
import pandas as pd
import numpy as np
import os
import codecs
from pandas.util.testing import assert_frame_equal
import pickle

import unittest

from dramavis import LinaCorpus, Lina

class CharMetricsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(CharMetricsTestCase):
        CharMetricsTestCase.LC = LinaCorpus(inputfolder="tests/testdata",
                                            outputfolder="testoutput_temp")

    @classmethod
    def tearDownClass(CharMetricsTestCase):
        os.rmdir("testoutput_temp")

    def test_read_dramas(self):
        number_dramas = len(list(self.LC.read_dramas(metrics=False)))
        self.assertEqual(number_dramas, 11)

    
