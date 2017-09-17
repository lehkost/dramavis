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

    def test_central_characters(self):
        true_df = pd.read_csv("testoutput_chars/central_characters.csv", sep=";")
        test_df = self.LC.get_central_characters()
        true_df.reset_index(inplace=True)
        test_df.reset_index(inplace=True)
        true_df.fillna("", inplace=True)
        test_df.fillna("", inplace=True)
        true_df.drop('index', axis=1, inplace=True)
        test_df.drop('index', axis=1, inplace=True)
        assert_frame_equal(test_df, true_df)

    
