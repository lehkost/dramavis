#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# dramavis by frank fischer (@umblaetterer) & christopher kittel (@chris_kittel)

__author__ = "Christopher Kittel <web at christopherkittel.eu>, Frank Fischer <ffischer at hse.ru>"
__copyright__ = "Copyright 2017"
__license__ = "MIT"
__version__ = "0.4 (beta)"
__maintainer__ = "Frank Fischer <ffischer at hse.ru>"
__status__ = "Development" # 'Development', 'Production' or 'Prototype'

from lxml import etree
import os
import glob
import pandas as pd
import networkx as nx
import csv
from itertools import chain, zip_longest
from collections import Counter
import argparse
from superposter import plotGraph, plot_superposter
import logging
import numpy

from linacorpus import LinaCorpus


class DramaPlotter(LinaCorpus):
    """Takes corpus object, add plotting functions."""

    def get_plots(self, randomization=True):
        """
        Main function executing the pipeline from
        reading and parsing lina-xmls,
        creating and plotting drama-networks,
        computing graph-metrics and random-graph-metrics,
        exporting SVGs, CSVs and edgelists.
        Can take a while.
        """
        dramas = self.read_dramas()
        # for ID, drama in dramas.items():
        for drama in dramas:
        # yields parsed dramas dicts
            # if args.debug:
            #     print("TITLE:", drama.title)
            if os.path.isfile(os.path.join(self.outputfolder, "_".join([str(drama.ID),drama.title])+".svg")):
                continue
            self.capture_fringe_cases(drama)
            if randomization:
                for i in range(0, 5):
                    R = nx.gnm_random_graph(drama.graph_metrics.get("charcount"), drama.graph_metrics.get("edgecount"))
                    plotGraph(R, filename=os.path.join(self.outputfolder, str(drama.ID)+"random"+str(i)+".svg"))

    def capture_fringe_cases(self, drama):
        if drama.graph_metrics.get("all_in_index") is None:
            print(drama.title)
