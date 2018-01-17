#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import os
import csv
from itertools import chain, zip_longest
from collections import Counter
import logging
import numpy as np
from numpy import ma
import pandas as pd
import networkx as nx
from scipy import stats
from scipy.optimize import curve_fit
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from linacorpus import LinaCorpus, Lina
from dramaplotter import plotGraph
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

__author__ = """Christopher Kittel <web at christopherkittel.eu>,
                Frank Fischer <ffischer at hse.ru>"""
__copyright__ = "Copyright 2017"
__license__ = "MIT"
__version__ = "0.4 (beta)"
__maintainer__ = "Frank Fischer <ffischer at hse.ru>"
__status__ = "Development"  # 'Development', 'Production' or 'Prototype'


class CorpusAnalyzer(LinaCorpus):

    def __init__(self, inputfolder, outputfolder, logpath, major_only=False,
                 randomization=1000):
        super(CorpusAnalyzer, self).__init__(inputfolder, outputfolder)
        self.logger = logging.getLogger("corpusAnalyzer")
        formatter = logging.Formatter('%(asctime)-15s %(name)s [%(levelname)s]'
                                      '%(message)s')
        fh = logging.FileHandler(logpath)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logpath = logpath
        self.major_only = major_only
        self.randomization = randomization

    def analyze_dramas(self, action):
        """
        Reads all XMLs in the inputfolder,
        returns an iterator of lxml.etree-objects created
        with lxml.etree.parse("dramafile.xml").
        """
        for dramafile in tqdm(self.dramafiles, desc="Dramas", mininterval=1):
            drama = DramaAnalyzer(dramafile, self.outputfolder, self.logpath,
                                  action, self.major_only, self.randomization)
            yield drama

    def get_char_metrics(self):
        self.logger.info("Exporting character metrics.")
        dramas = self.analyze_dramas(action="char_metrics")
        header = [
                    'ID', 'author', 'title', 'year',
                    'frequency', 'degree', 'betweenness', 'closeness'
                 ]
        dfs = []
        quot_quot_dfs = []
        for drama in dramas:
            temp_df = pd.DataFrame(index=[drama.ID])
            for m in header[1:3]:
                temp_df[m] = drama.metadata.get(m)
            temp_df['year'] = drama.metadata.get('date_definite')
            for m in header[4:]:
                temp_df[m] = drama.get_top_ranked_chars()[m]
            temp_df['ID'] = drama.ID
            dfs.append(temp_df)
            quot_quot_dfs.append(drama.quartile_quot)
        df = pd.concat(dfs)
        df = df[header]
        df.index = df['ID']
        df.index.name = 'index'
        df.to_csv(os.path.join(self.outputfolder,
                               "central_characters.csv"), sep=";")
        self.logger.info("Exporting corpus quartile metrics.")
        df = pd.concat(quot_quot_dfs, axis=1).T
        df.index.name = "index"
        df.to_csv(os.path.join(self.outputfolder,
                               "corpus_quartile_metrics.csv"),
                  sep=";")

    def get_graph_metrics(self):
        self.logger.info("Exporting corpus metrics.")
        dramas = self.analyze_dramas(action="corpus_metrics")
        df = pd.concat([d.graph_metrics for d in dramas])
        header = [
                'ID', 'author', 'title', 'subtitle', 'year', 'genretitle',
                'filename',
                'charcount', 'edgecount', 'maxdegree', 'avgdegree', 'diameter',
                'clustering_coefficient', 'clustering_coefficient_random',
                'avgpathlength', 'average_path_length_random', 'density',
                'segment_count', 'count_type', 'all_in_index',
                'change_rate_mean', 'change_rate_std',
                'final_scene_size_index', 'characters_last_in',
                'connected_components', 'spearman_rho_avg', 'spearman_rho_std',
                'spearman_rho_content_vs_network',
                'spearman_rho_content_vs_network_top',
                'spearman_rho_content_vs_network_bottom',
                'component_sizes'
                ]
        df.index = df["ID"]
        df.index.name = "index"
        df[header].to_csv(os.path.join(self.outputfolder,
                                       "corpus_metrics.csv"), sep=";")

    def get_both_metrics(self):
        self.logger.info("Exporting character metrics.")
        dramas = self.analyze_dramas(action="both")
        header = [
                    'ID', 'author', 'title', 'year',
                    'frequency', 'degree', 'betweenness', 'closeness'
                 ]
        char_dfs = []
        graph_dfs = []
        quot_quot_dfs = []
        for drama in dramas:
            temp_df = pd.DataFrame(index=[drama.ID])
            for m in header[1:3]:
                temp_df[m] = drama.metadata.get(m)
            temp_df['year'] = drama.metadata.get('date_definite')
            for m in header[4:]:
                temp_df[m] = drama.get_top_ranked_chars()[m]
            temp_df['ID'] = drama.ID
            char_dfs.append(temp_df)
            graph_dfs.append(drama.graph_metrics)
            quot_quot_dfs.append(drama.quartile_quot)
        df = pd.concat(char_dfs)
        df = df[header]
        df.index = df['ID']
        df.index.name = 'index'
        df.to_csv(os.path.join(self.outputfolder,
                               "central_characters.csv"), sep=";")
        self.logger.info("Exporting corpus metrics.")
        df = pd.concat(graph_dfs)
        header = [
                'ID', 'author', 'title', 'subtitle', 'year', 'genretitle',
                'filename',
                'charcount', 'edgecount', 'maxdegree', 'avgdegree', 'diameter',
                'clustering_coefficient', 'clustering_coefficient_random',
                'avgpathlength', 'average_path_length_random', 'density',
                'segment_count', 'count_type', 'all_in_index',
                'change_rate_mean', 'change_rate_std',
                'final_scene_size_index', 'characters_last_in',
                'connected_components', 'spearman_rho_avg', 'spearman_rho_std',
                'spearman_rho_content_vs_network',
                'spearman_rho_content_vs_network_top',
                'spearman_rho_content_vs_network_bottom',
                'component_sizes'
                ]
        df.index = df["ID"]
        df.index.name = "index"
        df.to_csv(os.path.join(self.outputfolder, "corpus_metrics.csv"),
                  sep=";")
        self.logger.info("Exporting corpus quartile metrics.")
        df_cq = pd.concat(quot_quot_dfs, axis=1).T
        df_cq.index = df["ID"]
        df_cq.index.name = "index"
        df_cq.to_csv(os.path.join(self.outputfolder,
                                  "corpus_quartile_metrics.csv"), sep=";")


class DramaAnalyzer(Lina):

    def __init__(self, dramafile, outputfolder, logpath,
                 action, major_only, randomization=1000):
        super(DramaAnalyzer, self).__init__(dramafile, outputfolder)
        self.logger = logging.getLogger("dramaAnalyzer")
        formatter = logging.Formatter('%(asctime)-15s %(name)s [%(levelname)s]'
                                      '%(message)s')
        fh = logging.FileHandler(logpath)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.major_only = major_only
        self.n_personae = len(self.personae)
        self.centralities = pd.DataFrame(index=[p for p in self.personae])
        self.centralities.index.name = "name"
        self.randomization = randomization
        self.metrics = pd.DataFrame()
        self.G = self.create_graph()
        self.action = action
        if action == "char_metrics":
            self.analyze_characters()
            self.get_character_frequencies()
            self.get_character_speech_amounts()
            self.get_character_ranks()
            self.get_centrality_ranks()
            self.get_rank_stability_measures()
            self.add_rank_stability_metrics()
            self.get_structural_ranking_measures()
            self.get_quartiles()
            self.export_char_metrics()
        if action == "corpus_metrics":
            self.graph_metrics = self.get_graph_metrics()
            self.export_graph_metrics()
        if action == "both":
            self.graph_metrics = self.get_graph_metrics()
            self.analyze_characters()
            self.get_character_frequencies()
            self.get_character_speech_amounts()
            self.get_character_ranks()
            self.get_centrality_ranks()
            self.get_rank_stability_measures()
            self.add_rank_stability_metrics()
            self.get_structural_ranking_measures()
            self.get_quartiles()
            self.get_regression_metrics()
            self.export_char_metrics()
            self.export_graph_metrics()

    def add_rank_stability_metrics(self):
        self.graph_metrics["spearman_rho_avg"] = (self.rank_stability
                                                      .stack()
                                                      .mean())
        self.graph_metrics["spearman_rho_std"] = (self.rank_stability
                                                      .stack()
                                                      .std())
        (self.graph_metrics["top_rank_char_count"],
         self.graph_metrics["top_rank_char_avg"],
         self.graph_metrics["top_rank_char_std"]) = (
                self.get_top_ranked_char_count())

    def get_final_scene_size(self):
        last_scene_size = len(self.segments[-1])
        return last_scene_size / self.n_personae

    def get_drama_change_rate_metrics(self):
        change_rates = self.get_drama_change_rate()
        cr_mean = np.mean(change_rates)
        cr_std = np.std(change_rates)
        return cr_mean, cr_std

    def get_drama_change_rate(self):
        change_rates = []
        for x, y in zip_longest(self.segments[:-1], self.segments[1:]):
            s = set(x)
            t = set(y)
            u = s.intersection(t)
            cr = abs(len(s)-len(u)) + abs(len(u)-len(t))
            cr_sum = len(s.union(t))
            change_rates.append(cr/cr_sum)
        return change_rates

    def get_central_character_entry(self):
        central_character = self.get_central_character()
        for i, segment in enumerate(self.segments):
            if central_character in segment:
                i += 1
                central_character_entry_index = float(i/len(self.segments))
                return central_character_entry_index

    def get_central_character(self):
        # either that or hardcode
        ranks = [c for c in self.centralities.columns if c.endswith("rank")]
        # sum up all rank values per character, divide by nr. of rank metrics
        avg_ranks = self.centralities[ranks].sum(axis=1)/len(ranks)
        min_rank = min(avg_ranks)
        central_chars = avg_ranks[avg_ranks == min_rank].index.tolist()
        if len(central_chars) == 1:
            return central_chars[0]
        else:
            return "SEVERAL"

    def get_character_frequencies(self):
        self.centralities['frequency'] = 0
        frequencies = Counter(list(chain.from_iterable(self.segments)))
        for char, freq in frequencies.items():
            self.centralities.loc[char, 'frequency'] = freq

    def get_character_speech_amounts(self):
        for amount in ["speech_acts", "words", "lines", "chars"]:
            self.centralities[amount] = 0
            for name, person in self.personae.items():
                self.centralities.loc[person.name, amount] = (person.amounts
                                                              .get(amount))

    def get_top_ranked_chars(self):
        top_ranked = {}
        # check whether metric should be sorted asc(min) or desc(max)
        for metric in ['degree', 'closeness', 'betweenness', 'frequency']:
            cent_max = self.centralities[metric].max()
            top_char = self.centralities[self.centralities[metric]
                                         == cent_max].index.tolist()
            if len(top_char) != 1:
                top_ranked[metric] = "SEVERAL"
            else:
                top_ranked[metric] = top_char[0]
        # top_ranked['central'] = self.get_central_character()
        return top_ranked

    def get_top_ranked_char_count(self):
        avg_min = self.centralities['centrality_rank_avg'].min()
        top_chars = self.centralities[self.centralities['centrality_rank_avg']
                                      == avg_min].index.tolist()
        top_std = self.centralities[self.centralities['centrality_rank_avg']
                                    == avg_min]['centrality_rank_std']
        return len(top_chars), avg_min, top_std

    def get_character_ranks(self):
        for metric in ['degree', 'closeness', 'betweenness',
                       'strength', 'eigenvector_centrality',
                       'frequency', 'speech_acts', 'words']:
            # ascending: False for ranks by high (1) to low (N)
            # check ascending value for each metric
            self.centralities[metric+"_rank"] = (self.centralities[metric]
                                                     .rank(method='min',
                                                           ascending=False))

    def get_quartiles(self):
        metrics = ['degree', 'closeness', 'betweenness',
                   'strength', 'eigenvector_centrality',
                   'frequency', 'speech_acts', 'words']
        index = ["q4", "q3", "q2", "q1"]
        df = pd.DataFrame(columns=metrics, index=index)
        for metric in metrics:
            df[metric] = ((pd.cut(self.centralities[metric], 4)
                             .value_counts()
                             .sort_index(ascending=False) /
                           len(self.centralities))
                          .tolist())
        self.quartile_quot = df.loc["q4"]/df.loc["q1"]
        self.quartile_quot.name = self.ID
        self.quartile_quot = self.quartile_quot.append(df.T.stack())

    def get_centrality_ranks(self):
        ranks = [c for c in self.centralities.columns if c.endswith("rank")]
        self.centralities['centrality_rank_avg'] = (self.centralities[ranks]
                                                        .sum(axis=1) /
                                                    len(ranks))
        self.centralities['centrality_rank_std'] = (self.centralities[ranks]
                                                        .std(axis=1) /
                                                    len(ranks))
        for metric in ['centrality_rank_avg', 'centrality_rank_std']:
            self.centralities[metric+"_rank"] = (self.centralities[metric]
                                                     .rank(method='min',
                                                           ascending=True))

    def get_rank_stability_measures(self):
        ranks = [c
                 for c in self.centralities.columns
                 if c.endswith("rank")][:8]
        self.rank_stability = (self.centralities[ranks]
                                   .corr(method='spearman'))
        np.fill_diagonal(self.rank_stability.values, np.nan)
        self.rank_stability.index.name = "rank_name"

    def get_structural_ranking_measures(self):
        graph_ranks = ['degree_rank', 'closeness_rank', 'betweenness_rank',
                       'strength_rank', 'eigenvector_centrality_rank']
        content_ranks = ['frequency_rank', 'speech_acts_rank', 'words_rank']
        avg_graph_rank = (self.centralities[graph_ranks]
                              .mean(axis=1)
                              .rank(method='min'))
        avg_content_rank = (self.centralities[content_ranks]
                                .mean(axis=1)
                                .rank(method='min'))
        self.centralities["avg_graph_rank"] = avg_graph_rank
        self.centralities["avg_content_rank"] = avg_content_rank
        self.centralities["overall_avg"] = (self.centralities[
                                                 ["avg_graph_rank",
                                                  "avg_content_rank"]]
                                                .mean(axis=1))
        self.centralities["overall_avg_rank"] = (self.centralities[
                                                        "overall_avg"]
                                                     .rank(method='min'))
        struct_corr = stats.spearmanr(avg_content_rank, avg_graph_rank)[0]
        self.graph_metrics["spearman_rho_content_vs_network"] = struct_corr
        top, bottom = np.split(self.centralities,
                               [int(.5*len(self.centralities))])
        struct_corr_top = stats.spearmanr(top["avg_content_rank"],
                                          top["avg_graph_rank"])[0]
        self.graph_metrics["spearman_rho_content_vs_network_top"] = (
                                                            struct_corr_top)
        struct_corr_bottom = stats.spearmanr(bottom["avg_content_rank"],
                                             bottom["avg_graph_rank"])[0]
        self.graph_metrics["spearman_rho_content_vs_network_bottom"] = (
                                                        struct_corr_bottom)

    def get_characters_all_in_index(self):
        appeared = set()
        for i, speakers in enumerate(self.segments):
            for sp in speakers:
                appeared.add(sp)
            if len(appeared) >= self.num_chars_total:
                i += 1
                all_in_index = float(i/len(self.segments))
                return all_in_index

    def export_char_metrics(self):
        self.centralities.index.name = "name"
        self.centralities.to_csv(
                os.path.join(
                            self.outputfolder,
                            "%s_%s_chars.csv" % (self.ID, self.title)
                            ))
        self.rank_stability.to_csv(
                os.path.join(
                            self.outputfolder,
                            "%s_%s_spearmanrho.csv" % (self.ID, self.title)
                            ))

    def export_graph_metrics(self):
        self.graph_metrics.index.name = "ID"
        self.graph_metrics.index = self.graph_metrics["ID"]
        self.graph_metrics.to_csv(os.path.join(
                                    self.outputfolder,
                                    "%s_%s_graph.csv" % (self.ID, self.title)),
                                  sep=";")
        self.export_table(
                self.get_drama_change_rate(),
                "_".join([self.filepath, self.title, "change_rates"])+".csv")
        nx.write_edgelist(
                self.G,
                os.path.join(self.outputfolder,
                             "_".join([str(self.ID),
                                      self.title, "edgelist"])+".csv"),
                delimiter=";",
                data=["weight"])
        plotGraph(
                self.G,
                filename=os.path.join(self.outputfolder,
                                      "_".join([str(self.ID),
                                               self.title])+".svg"))

    def export_table(self, t, filepath):
        with open(filepath, 'w') as f:  # Just use 'w' mode in 3.x
            csvwriter = csv.writer(f, delimiter=';')
            csvwriter.writerow(["segment", "change_rate"])
            for i, t in enumerate(t):
                csvwriter.writerow([i+1, t])

    def get_graph_metrics(self):
        graph_metrics = self.analyze_graph()
        graph_metrics["ID"] = self.ID
        (graph_metrics["average_path_length_random"],
         graph_metrics["clustering_coefficient_random"]) = (
                self.randomize_graph(graph_metrics.get("charcount"),
                                     graph_metrics.get("edgecount")))
        graph_metrics["year"] = self.metadata.get("date_definite")
        graph_metrics["author"] = self.metadata.get("author")
        graph_metrics["title"] = self.title
        graph_metrics["filename"] = self.metadata.get("filename")
        graph_metrics["genretitle"] = self.metadata.get("genretitle")
        graph_metrics["subtitle"] = self.metadata.get("subtitle")
        graph_metrics["segment_count"] = self.metadata.get("segment_count")
        graph_metrics["count_type"] = self.metadata.get("count_type")
        graph_metrics["all_in_index"] = self.get_characters_all_in_index()
        (graph_metrics["change_rate_mean"],
         graph_metrics["change_rate_std"]) = (
            self.get_drama_change_rate_metrics())
        graph_metrics["final_scene_size_index"] = self.get_final_scene_size()
        graph_metrics["characters_last_in"] = self.get_characters_last_in()
        return pd.DataFrame.from_dict(graph_metrics, orient='index').T

    def get_characters_last_in(self):
        last_chars = self.segments[-1]
        return ",".join(last_chars)

    def create_graph(self):
        """
        First creates a bipartite graph with scenes on the one hand,
        and speakers in one scene on the other.
        The graph is then projected into a unipartite graph of speakers,
        which are linked if they appear in one scene together.

        Returns a networkx weighted projected graph.
        """
        speakerset = self.segments

        B = nx.Graph()
        labels = {}
        for i, speakers in enumerate(speakerset):
            # speakers are Character objects
            source = str(i)
            targets = speakers
            # if args.debug:
            #     print("SOURCE, TARGET:", source, targets)

            if source not in B.nodes():
                B.add_node(source, bipartite=0)
                labels[source] = source

            for target in targets:
                if target not in B.nodes():
                    B.add_node(target, bipartite=1)
                B.add_edge(source, target)

        scene_nodes = set(n
                          for n, d in B.nodes(data=True)
                          if d['bipartite'] == 0)
        person_nodes = set(B) - scene_nodes
        nx.is_bipartite(B)
        G = nx.bipartite.weighted_projected_graph(B, person_nodes)
        if self.major_only:
            G = max(nx.connected_component_subgraphs(G), key=len)
        return G

    def analyze_graph(self):
        """
        Computes various network metrics for a graph G,
        returns a dictionary:
        values =
        {
            "charcount" = len(G.nodes()),
            "edgecount" = len(G.edges()),
            "maxdegree" = max(G.degree().values()) or "NaN"
                          if ValueError: max() arg is an empty sequence,
            "avgdegree" = sum(G.degree().values())/len(G.nodes()) or "NaN"
                          if ZeroDivisionError: division by zero,
            "density" = nx.density(G) or "NaN",
            "avgpathlength" = nx.average_shortest_path_length(G) or "NaN"
                              if NetworkXError: Graph is not connected,
                              then it tries to get the average shortest path
                              length from the giant component,
            "avgpathlength" = nx.average_shortest_path_length(
                             max(nx.connected_component_subgraphs(G),
                                 key=len))
                             except NetworkXPointlessConcept:
                             ('Connectivity is undefined for the null graph.'),
            "clustering_coefficient" = nx.average_clustering(G) or "NaN"
                            if ZeroDivisionError: float division by zero
        }
        """
        G = self.G
        values = {}
        values["charcount"] = len(G.nodes())
        values["edgecount"] = len(G.edges())
        try:
            values["maxdegree"] = max(G.degree().values())
        except:
            self.logger.error(
                "ID %s ValueError: max() arg is an empty sequence" % self.ID)
            values["maxdegree"] = "NaN"

        try:
            values["avgdegree"] = sum(G.degree().values())/len(G.nodes())
        except:
            self.logger.error(
                "ID %s ZeroDivisionError: division by zero" % self.ID)
            values["avgdegree"] = "NaN"

        try:
            values["density"] = nx.density(G)
        except:
            values["density"] = "NaN"

        try:
            values["avgpathlength"] = nx.average_shortest_path_length(G)
        except nx.NetworkXError:
            self.logger.error(
                "ID %s NetworkXError: Graph is not connected." % self.ID)
            try:
                self.randomization = 50
                values["avgpathlength"] = nx.average_shortest_path_length(
                            max(nx.connected_component_subgraphs(G), key=len))
            except:
                values["avgpathlength"] = "NaN"
        except:
            self.logger.error("ID %s NetworkXPointlessConcept: ('Connectivity"
                              "is undefined for the null graph.')" % self.ID)
            values["avgpathlength"] = "NaN"

        try:
            values["clustering_coefficient"] = nx.average_clustering(G)
        except:
            self.logger.error(
                "ID %s ZeroDivisionError: float division by zero" % self.ID)
            values["clustering_coefficient"] = "NaN"
        values["connected_components"] = nx.number_connected_components(G)
        components = nx.connected_component_subgraphs(G)
        values["component_sizes"] = [len(c.nodes()) for c in components]
        try:
            values["diameter"] = nx.diameter(G)
        except nx.NetworkXError:
            self.logger.error(
                "ID %s NetworkXError: Graph is not connected." % self.ID)
            values["diameter"] = nx.diameter(
                        max(nx.connected_component_subgraphs(G), key=len))
        return values

    def analyze_characters(self):
        """
        Computes per-character metrics of a graph G,
        returns dictionary of dictionaries:
        character_values =
        {
            "betweenness" = nx.betweenness_centrality(G),
            "degree" = nx.degree(G),
            "closeness" = nx.closeness_centrality(G)
        }
        """
        # initialize columns with 0
        for metric in ['betweenness', 'degree',
                       'closeness', 'closeness_corrected',
                       'strength',
                       'eigenvector_centrality']:
            self.centralities[metric] = 0
        for char, metric in nx.betweenness_centrality(self.G).items():
            self.centralities.loc[char, 'betweenness'] = metric
        for char, metric in nx.degree(self.G).items():
            self.centralities.loc[char, 'degree'] = metric
        for char, metric in nx.degree(self.G, weight="weight").items():
            self.centralities.loc[char, 'strength'] = metric
        for char, metric in nx.closeness_centrality(self.G).items():
            self.centralities.loc[char, 'closeness'] = metric
        for g in nx.connected_component_subgraphs(self.G):
            for char, metric in nx.closeness_centrality(g).items():
                self.centralities.loc[char, 'closeness_corrected'] = metric
        try:
            for char, metric in nx.eigenvector_centrality(
                                    self.G, max_iter=500).items():
                self.centralities.loc[char, 'eigenvector_centrality'] = metric
        except Exception as e:
            self.logger.error(
                "%s networkx.exception.NetworkXError:"
                " eigenvector_centrality(): power iteration failed to converge"
                " in 500 iterations." % self.ID)
        self.centralities['avg_distance'] = 1/self.centralities['closeness']
        self.centralities['avg_distance_corrected'] = (
                                1 / self.centralities['closeness_corrected'])

    def transpose_dict(self, d):
        """
        Transpose dict of character-network metrics to an exportable dict,
        essentially transposes rows and columns of the character.csv.
        """
        td = {}
        try:
            for cent, chars in d.items():
                for char in chars:
                    td[char] = {}
        except:
            pass
        try:
            for cent, chars in d.items():
                for char, value in chars.items():
                    td[char][cent] = value
        except:
            pass
        return td

    def randomize_graph(self, n, e):
        """
        Creates 1000 random graphs with
        networkx.gnm_random_graph(nodecount, edgecount),
        and computes average_clustering_coefficient and
        average_shortest_path_length, to compare with drama-graph.
        Returns a tuple:
        randavgpathl, randcluster = (float or "NaN", float or "NaN")
        """
        randcluster = 0
        randavgpathl = 0
        # what is c, what is a, what is n, what is e?
        # e=edges?, c=clustering_coefficient?, a=average_shortest_path_length?
        c = 0
        a = 0
        if not self.randomization:  # hack so that quartett poster works
            self.randomization = 50
        for i in tqdm(range(self.randomization), desc="Randomization",
                      mininterval=1):
            R = nx.gnm_random_graph(n, e)
            try:
                randcluster += nx.average_clustering(R)
                c += 1
            except ZeroDivisionError:
                pass
            j = 0
            while True:
                j += 1
                try:
                    R = nx.gnm_random_graph(n, e)
                    randavgpathl += nx.average_shortest_path_length(R)
                    a += 1
                except:
                    pass
                else:
                    break
                if j > 50:
                    randavgpathl = "NaN"
                    break
        try:
            randcluster = randcluster / c
        except:
            randcluster = "NaN"
        try:
            randavgpathl = randavgpathl / a
        except:
            randavgpathl = "NaN"
        return randavgpathl, randcluster

    def get_regression_metrics(self):
        metrics = ['degree', 'closeness', 'betweenness',
                   'strength', 'eigenvector_centrality',
                   'frequency', 'speech_acts', 'words']
        metrics_dfs = []
        for metric in metrics:
            temp_df = pd.DataFrame(columns=[metric])
            temp_df[metric+"_interval"] = [
                        i.mid for i in pd.cut(self.centralities[metric], 10)
                                         .value_counts()
                                         .index.tolist()]
            temp_df[metric] = (pd.cut(self.centralities[metric], 10)
                                 .value_counts()
                                 .tolist())
            temp_df.sort_values(metric+"_interval", inplace=True)
            temp_df.reset_index(drop=True, inplace=True)
            metrics_dfs.append(temp_df)
        index = ["linear", "exponential", "powerlaw", "quadratic"]
        reg_metrics = pd.DataFrame(columns=metrics, index=index)
        # fit linear models
        fig = plt.figure(figsize=(len(metrics)*4, len(index)*4))
        gs = gridspec.GridSpec(len(index), len(metrics))
        i = 0  # subplot enumerator
        for metric, temp_df in zip(metrics, metrics_dfs):
            X = np.array(temp_df[metric+"_interval"]).reshape(-1, 1)
            y = np.array(temp_df[metric]).reshape(-1, 1)
            model = linear_model.LinearRegression()
            model.fit(X, y)
            score = model.score(X, y)
            reg_metrics.loc["linear", metric] = score
            ax = plt.subplot(gs[i])
            plt.scatter(X, y)
            plt.plot(X, model.predict(X), 'r--',
                     label='coeff: %.3f, intercept: %.3f' % (model.coef_[0][0],
                                                             model.intercept_[0]))
            # plt.legend(fontsize='x-small')
            ax.set_title(metric + " linear R2 %.3f" % score, size='medium')
            ax.set_xlabel(metric)
            ax.set_ylabel("value counts")
            i += 1
        # fit quadratic models
        for metric, temp_df in zip(metrics, metrics_dfs):
            X = np.array(temp_df[metric+"_interval"]).reshape(-1, 1)
            y = np.array(temp_df[metric]).reshape(-1, 1)
            regr = linear_model.LinearRegression()
            model = Pipeline(steps=[('polyfeatures', PolynomialFeatures(2)),
                                    ('reg', regr)])
            model.fit(X, y)
            score = model.score(X, y)
            reg_metrics.loc["quadratic", metric] = score
            ax = plt.subplot(gs[i])
            plt.scatter(X, y)
            plt.plot(X, model.predict(X), 'r--',
                     label='coeff: %s, intercept: %s' % (
                                    str(model.named_steps['reg'].coef_),
                                    str(model.named_steps['reg'].intercept_)))
            # plt.legend(fontsize='x-small')
            ax.set_title(metric + " quadratic R2 %.3f" % score, size='medium')
            ax.set_xlabel(metric)
            ax.set_ylabel("value counts")
            i += 1
        # fit exp models
        for metric, temp_df in zip(metrics, metrics_dfs):
            X = np.array(temp_df[metric+"_interval"]).reshape(-1, 1)
            y = np.array(temp_df[metric]).reshape(-1, 1)
            logy = ma.log(y).reshape(-1, 1)
            model = linear_model.LinearRegression()
            model.fit(X, logy)
            score = model.score(X, logy)
            reg_metrics.loc["exponential", metric] = score
            ax = plt.subplot(gs[i])
            plt.scatter(X, logy)
            plt.plot(X, model.predict(X), 'r--')
            # plt.legend(fontsize='x-small')
            ax.set_title(metric + " exp. R2 %.3f" % score, size='medium')
            ax.set_xlabel(metric)
            ax.set_ylabel("value counts (log)")
            i += 1
        # fit power law models
        for metric, temp_df in zip(metrics, metrics_dfs):
            X = np.array(temp_df[metric+"_interval"])
            y = np.array(temp_df[metric])
            logx = ma.log(X).reshape(-1, 1)
            logy = ma.log(y).reshape(-1, 1)
            model = linear_model.LinearRegression()
            model.fit(logx, logy)
            score = model.score(logx, logy)
            reg_metrics.loc["powerlaw", metric] = score
            ax = plt.subplot(gs[i])
            plt.scatter(logx, logy)
            plt.plot(logx, model.predict(logx), 'r--',
                     label='coeff: %s, intercept: %s' % (str(model.coef_),
                                                         str(model.intercept_)))
            # plt.legend(fontsize='x-small')
            ax.set_title(metric + " power law R2 %.3f" % score, size='medium')
            ax.set_xlabel(metric + " (log)")
            ax.set_ylabel("value counts (log)")
            i += 1
        plt.tight_layout()
        self.reg_metrics = reg_metrics.T
        self.reg_metrics.index.name = "metrics"
        self.reg_metrics["max_val"] = self.reg_metrics.apply(
                                                lambda x: np.max(x), axis=1)
        self.reg_metrics["max_type"] = self.reg_metrics.apply(
                                                lambda x: np.argmax(x), axis=1)
        for metric in metrics:
            self.graph_metrics[metric+"_reg_type"] = (self.reg_metrics
                                                          .loc[metric,
                                                               'max_type'])
            self.graph_metrics[metric+"_reg_val"] = (self.reg_metrics
                                                         .loc[metric,
                                                              'max_val'])
        self.reg_metrics.to_csv(os.path.join(self.outputfolder,
                                             "%s_%s_regression_table.csv"
                                             % (self.ID, self.title)))
        for temp_df in metrics_dfs:
            temp_df.to_csv(os.path.join(os.path.join(self.outputfolder),
                                        "%s_%s_regression_table.csv"
                                        % (self.ID, self.title)),
                           mode='a', header=True)
        fig.savefig(os.path.join(self.outputfolder,
                                 '%s_%s_regression_plots.png'
                                 % (self.ID, self.title)))


def exponential_func(t, a, b):
    return a + t*np.log(t)
