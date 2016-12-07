#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# dramavis by frank fischer (@umblaetterer) & christopher kittel (@chris_kittel)

__author__ = "Christopher Kittel <web at christopherkittel.eu>, Frank Fischer <ffischer at hse.ru>"
__copyright__ = "Copyright 2016"
__license__ = "MIT"
__version__ = "0.3 (beta)"
__maintainer__ = "Frank Fischer <ffischer at hse.ru>"
__status__ = "Development" # 'Development', 'Production' or 'Prototype'

from lxml import etree
import os
import glob
import networkx as nx
import csv
from itertools import chain, zip_longest
from collections import Counter
import argparse
from superposter import plotGraph, plot_superposter
import logging
import numpy


class LinaCorpus(object):
    """docstring for LinaCorpus"""
    def __init__(self, inputfolder, outputfolder):
        self.inputfolder = inputfolder
        self.outputfolder = outputfolder
        self.dramafiles = glob.glob(os.path.join(self.inputfolder, '*.xml'))
        self.size = sum(1 for d in self.read_dramas(metrics=False))
        if not os.path.isdir(outputfolder):
            os.mkdir(outputfolder)

    def read_dramas(self, metrics=True):
        """
        Reads all XMLs in the inputfolder,
        returns an iterator of lxml.etree-objects created with lxml.etree.parse("dramafile.xml").
        """
        # dramas = {}
        for dramafile in self.dramafiles:
            # ID, ps = parse_drama(tree, filename)
            # dramas[ID] = ps
            drama = Lina(dramafile, self.outputfolder, metrics)
            yield drama

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
            if args.debug:
                print("TITLE:", drama.title)
            if os.path.isfile(os.path.join(self.outputfolder, "_".join([str(drama.ID),drama.title])+".svg")):
                continue
            self.capture_fringe_cases(drama)
            if randomization:
                for i in range(0, 5):
                    R = nx.gnm_random_graph(drama.graph_metrics.get("charcount"), drama.graph_metrics.get("edgecount"))
                    plotGraph(R, filename=os.path.join(self.outputfolder, str(drama.ID)+"random"+str(i)+".svg"))

    def get_central_characters(self):
        dramas = self.read_dramas(metrics=False)
        header = [
                    'author', 'title', 'year',
                    'frequency', 'degree', 'betweenness', 'closeness',
                    'central'
                 ]
        with open(os.path.join(self.outputfolder, "central_characters.csv"), "w") as outfile:
            csvwriter = csv.writer(outfile, delimiter=";", quotechar='"')
            csvwriter.writerow(header)
        with open(os.path.join(self.outputfolder, "central_characters.csv"), "a") as outfile:
            csvwriter = csv.writer(outfile, delimiter=";", quotechar='"')
            for drama in dramas:
                metadata = [drama.metadata.get('author'), drama.metadata.get('title'), drama.metadata.get('date_definite')]
                chars = [drama.get_top_characters()[m] for m in header[3:]]
                csvwriter.writerow(metadata+chars)


    def get_metrics(self):
        dramas = self.read_dramas()
        header =    [
                    'ID', 'author', 'title', 'subtitle', 'year', 'genretitle', 'filename',
                    'charcount', 'edgecount', 'maxdegree', 'avgdegree',
                    'clustering_coefficient', 'clustering_coefficient_random', 'avgpathlength', 'average_path_length_random', 'density',
                    'segment_count', 'count_type', 'all_in_index', 'central_character_entry_index', 'change_rate_mean', 'change_rate_std', 'final_scene_size_index',
                    'central_character', 'characters_last_in'
                    ]
        with open(os.path.join(self.outputfolder, "corpus_metrics.csv"), "w") as outfile:
            csvwriter = csv.writer(outfile, delimiter=";", quotechar='"')
            csvwriter.writerow(header)
        with open(os.path.join(self.outputfolder, "corpus_metrics.csv"), "a") as outfile:
            csvwriter = csv.writer(outfile, delimiter=";", quotechar='"')
            for drama in dramas:
                metrics = [drama.graph_metrics[m] for m in header]
                csvwriter.writerow(metrics)
                drama.write_output()

    def capture_fringe_cases(self, drama):
        if drama.graph_metrics.get("all_in_index") is None:
            print(drama.title)


class Lina(object):
    """docstring for Lina"""
    def __init__(self, dramafile, outputfolder, metrics=True):
        self.outputfolder = outputfolder
        self.tree = etree.parse(dramafile)
        self.filename = os.path.splitext(os.path.basename((dramafile)))[0]
        ID, metadata, personae, segments = self.parse_drama()
        self.ID = ID
        self.metadata = metadata
        self.personae = personae
        self.num_chars_total = len(personae)
        self.segments = segments
        self.filepath = os.path.join(self.outputfolder, str(self.ID))
        self.title = self.metadata.get("title", self.ID)
        self.G = self.create_graph()
        self.character_metrics = self.get_character_metrics()
        self.character_centralities = self.get_central_characters()
        if metrics:
            self.graph_metrics = self.get_graph_metrics()

    def get_final_scene_size(self):
        personae = set(list(chain.from_iterable(self.segments)))
        last_scene_size = len(self.segments[-1])
        return last_scene_size / len(personae)

    def get_drama_change_rate_metrics(self):
        change_rates = self.get_drama_change_rate()
        cr_mean = numpy.mean(change_rates)
        cr_std = numpy.std(change_rates)
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
        cc = sorted(self.character_centralities, key=self.character_centralities.__getitem__)
        cr = [self.character_centralities[c] for c in cc]
        minrank = min(cr)
        central_chars = [i for i, j in enumerate(cr) if j == minrank]
        if len(central_chars) == 1:
            return cc[central_chars[0]]
        else:
            return None

    def get_character_frequencies(self):
        frequencies = Counter(list(chain.from_iterable(self.segments)))
        return frequencies

    def get_top_characters(self):
        top_chars = {}
        top_chars['frequency'] = self.get_character_frequencies().most_common(1)[0][0]
        ranked_metrics = self.get_ranked_characters()
        top_chars['degree'] = ranked_metrics['degree'][0]
        top_chars['closeness'] = ranked_metrics['closeness'][0]
        top_chars['betweenness'] = ranked_metrics['betweenness'][0]
        top_chars['central'] = self.get_central_character()
        return top_chars

    def get_ranked_characters(self):
        ranked_metrics = {}
        ranked_metrics['degree'] = sorted(self.character_metrics['degree'], key=self.character_metrics['degree'].__getitem__, reverse=True)
        ranked_metrics['closeness'] = sorted(self.character_metrics['closeness'], key=self.character_metrics['degree'].__getitem__, reverse=True)
        ranked_metrics['betweenness'] = sorted(self.character_metrics['betweenness'], key=self.character_metrics['degree'].__getitem__, reverse=True)
        return ranked_metrics

    def get_character_ranks(self):
        ranks = {}
        personae = set(list(chain.from_iterable(self.segments)))
        for person in personae:
            ranks[person] = {}
        ranked_metrics = self.get_ranked_characters()
        for person in personae:
            ranks[person]['degree'] = ranked_metrics['degree'].index(person)+1
            ranks[person]['closeness'] = ranked_metrics['closeness'].index(person)+1
            ranks[person]['betweenness'] = ranked_metrics['betweenness'].index(person)+1
        return ranks

    def get_centrality_ranks(self):
        ranks = self.get_character_ranks()
        centrality_ranks = {}
        personae = set(list(chain.from_iterable(self.segments)))
        for person in personae:
            centrality_ranks[person] = float(sum([ranks[person]['degree'],ranks[person]['closeness'],ranks[person]['betweenness']]) / 3.)
        return centrality_ranks

    def get_central_characters(self):
        frequencies = self.get_character_frequencies()
        frequency_ranks = sorted(frequencies, key=frequencies.__getitem__, reverse=True)
        centrality_ranks = self.get_centrality_ranks()
        central_characters = {}
        personae = set(list(chain.from_iterable(self.segments)))
        for person in personae:
            central_characters[person] = float(sum([frequency_ranks.index(person)+1, centrality_ranks[person]]) / 2.)
        return central_characters


    def get_characters_all_in_index(self):
        appeared = set()
        for i, speakers in enumerate(self.segments):
            for sp in speakers:
                appeared.add(sp)
            if len(appeared) >= self.num_chars_total:
                i += 1
                all_in_index = float(i/len(self.segments))
                # print(all_in_index, len(self.segments), len(appeared), self.num_chars_total)
                return all_in_index

    def write_output(self):
        self.export_dict(self.graph_metrics, "_".join([self.filepath,self.title,"graph"])+".csv")
        self.export_table(self.get_drama_change_rate(), "_".join([self.filepath, self.title,"change_rates"])+".csv")
        chars = self.character_metrics
        chars['weighted_centralities_rank'] = self.get_character_ranks()
        chars['central_character_rank'] = self.character_centralities
        self.export_dicts(chars, "_".join([self.filepath,self.title,"chars"])+".csv")
        nx.write_edgelist(self.G, os.path.join(self.outputfolder, "_".join([str(self.ID),self.title,"edgelist"])+".csv"), delimiter=";", data=["weight"])
        plotGraph(self.G, filename=os.path.join(self.outputfolder, "_".join([str(self.ID),self.title])+".svg"))

    def export_table(self, t, filepath):
        with open(filepath, 'w') as f:  # Just use 'w' mode in 3.x
            csvwriter = csv.writer(f, delimiter=';')
            csvwriter.writerow(["segment", "change_rate"])
            for i, t in enumerate(t):
                csvwriter.writerow([i+1, t])

    def get_graph_metrics(self):
        graph_metrics = self.analyze_graph()
        graph_metrics["ID"] = self.ID
        graph_metrics["average_path_length_random"], graph_metrics["clustering_coefficient_random"] = self.randomize_graph(graph_metrics.get("charcount"), graph_metrics.get("edgecount"))
        graph_metrics["year"] = self.metadata.get("date_definite")
        graph_metrics["author"] = self.metadata.get("author")
        graph_metrics["title"] = self.title
        graph_metrics["filename"] = self.metadata.get("filename")
        graph_metrics["genretitle"] = self.metadata.get("genretitle")
        graph_metrics["subtitle"] = self.metadata.get("subtitle")
        graph_metrics["segment_count"] = self.metadata.get("segment_count")
        graph_metrics["count_type"] = self.metadata.get("count_type")
        graph_metrics["all_in_index"] = self.get_characters_all_in_index()
        graph_metrics["central_character_entry_index"] = self.get_central_character_entry()
        graph_metrics["change_rate_mean"], graph_metrics["change_rate_std"] = self.get_drama_change_rate_metrics()
        graph_metrics["final_scene_size_index"] = self.get_final_scene_size()
        graph_metrics["central_character"] = self.get_central_character()
        graph_metrics["characters_last_in"] = self.get_characters_last_in()
        return graph_metrics

    def get_characters_last_in(self):
        last_chars = self.segments[-1]
        return ",".join(last_chars)

    def get_character_metrics(self):
        character_metrics = self.analyze_characters(self.G)
        return character_metrics

    def parse_drama(self):
        """
        Parses a single drama,
        runs extractors for metadata, personae, speakers and scenes,
        adds filename and scene count to metadata.
        returns dictionary:
        {ID:
            {
            "metadata":metadata,
            "personae":personae,
            "structure":structure
            }
        }
        """
        root = self.tree.getroot()
        ID = root.attrib.get("id")
        header = root.find("{*}header")
        persons = root.find("{*}personae")
        text = root.find("{*}text")
        metadata = self.extract_metadata(header)
        metadata["filename"] = self.filename
        personae = self.extract_personae(persons)
        charmap = self.create_charmap(personae)
        segments = self.extract_speakers(charmap)
        metadata["segment_count"] = len(segments)
        metadata["count_type"] = self.get_count_type()
        # parsed_drama = (ID, {"metadata": metadata, "personae":personae, "speakers":speakers})
        # return parsed_drama

        if args.debug:
            print("SEGMENTS:", segments)
        return ID, metadata, personae, segments

    def extract_metadata(self, header):
        """

        Extracts metadata from the header-tag of a lina-xml,
        returns dictionary:

        metadata = {
            "title":title,
            "subtitle":subtitle,
            "genretitle":genretitle,
            "author":author,
            "pnd":pnd,
            "date_print":date_print,
            "date_written":date_written,
            "date_premiere":date_premiere,
            "date_definite":date_definite,
            "source_textgrid":source_textgrid
        }
        """
        title = header.find("{*}title").text
        try:
            subtitle = header.find("{*}subtitle").text
        except AttributeError:
            subtitle = ""
        try:
            genretitle = header.find("{*}genretitle").text
        except AttributeError:
            genretitle = ""
        author = header.find("{*}author").text
        pnd = header.find("{*}title").text
        try:
            date_print = int(header.find("{*}date[@type='print']").attrib.get("when"))
        except:
            date_print = None
        try:
            date_written = int(header.find("{*}date[@type='written']").attrib.get("when"))
        except:
            date_written = None
        try:
            date_premiere = int(header.find("{*}date[@type='premiere']").attrib.get("when"))
        except:
            date_premiere = None

        if date_print and date_premiere:
            date_definite = min(date_print, date_premiere)
        elif date_premiere:
            date_definite = date_premiere
        else:
            date_definite = date_print

        ## date is a string hotfix
        # if type(date_print) != int:
        #     date_print = 9999
        # if type(date_written) != int:
        #     date_print = 9999
        # if type(date_premiere) != int:
        #     date_print = 9999

        if date_written and date_definite:
            if date_definite - date_written > 10:
                date_definite = date_written
        elif date_written and not date_definite:
            date_definite = date_written

        source_textgrid = header.find("{*}source").text

        metadata = {
            "title":title,
            "subtitle":subtitle,
            "genretitle":genretitle,
            "author":author,
            "pnd":pnd,
            "date_print":date_print,
            "date_written":date_written,
            "date_premiere":date_premiere,
            "date_definite":date_definite,
            "source_textgrid":source_textgrid
        }
        return metadata

    def extract_personae(self, persons):
        """
        Extracts persons and aliases from the personae-tag of a lina-xml,
        returns list of dictionaries:
        personae = [
            {"charactername":["list", "of", "aliases"]},
            {"charactername2":["list", "of", "aliases"]}
        ]
        """
        personae = []
        for char in persons.getchildren():
            name = char.find("{*}name").text
            aliases = [alias.attrib.get('{http://www.w3.org/XML/1998/namespace}id') for alias in char.findall("{*}alias")]
            if args.debug:
                print("ALIASES:", aliases)
            if name:
                personae.append({name:aliases})
            else:
                personae.append({aliases[0]:aliases})
        if args.debug:
            print("PERSONAE:", personae)
        return personae

    def extract_structure(self):
        text = self.tree.getroot().find("{*}text")
        sps = text.findall(".//{*}sp")
        parentsegments = list()
        for sp in sps:
            parent = sp.getparent()
            head = parent.getchildren()[0]
            if parent not in parentsegments:
                parentsegments.append(parent)
            # check if scene (ends with "Szene/Szene./Auftritt/Auftritt.")
        return parentsegments

    def extract_speakers(self, charmap):
        parentsegments = self.extract_structure()
        segments = list()
        for segment in parentsegments:
            speakers = [speaker.attrib.get("who").replace("#","").split() for speaker in segment.findall(".//{*}sp")]
            speakers = list(chain.from_iterable(speakers))
            speakers = [charmap[speaker] for speaker in speakers]
            segments.append(list(set(speakers)))
        return segments

    def get_count_type(self):
        text = self.tree.getroot().find("{*}text")
        sps = text.findall(".//{*}sp")
        count_type = "acts"
        for sp in sps:
            parent = sp.getparent()
            head = parent.getchildren()[0]
            if head.text.endswith("ne") or head.text.endswith("ne.") or head.text.endswith("tt") or head.text.endswith("tt."):
                count_type = "scenes"
                # print(head.text)
        return count_type

    def create_charmap(self, personae):
        """
        Maps aliases back to the definite personname,
        returns a dictionary:
        charmap =
        {"alias1_1":"PERSON1",
         "alias1_2":"PERSON1",
         "alias2_1":"PERSON2",
         "alias2_2":"PERSON2"
        }
        """
        charmap = {}
        for person in personae:
            for charname, aliases in person.items():
                for alias in aliases:
                    charmap[alias] = charname
        return charmap

    def create_graph(self):
        """
        First creates a bipartite graph with scenes on the one hand,
        and speakers in one scene on the other.
        The graph is then projected into a unipartite graph of speakers,
        which are linked if they appear in one scene together.

        Returns a networkx weighted projected graph.
        """
        speakerset = self.segments
        personae = self.personae

        B = nx.Graph()
        labels = {}
        for i, speakers in enumerate(speakerset):

            source = str(i)
            targets = speakers
            if args.debug:
                print("SOURCE, TARGET:", source, targets)

            if not source in B.nodes():
                B.add_node(source, bipartite=0)
                labels[source] = source

            for target in targets:
                if not target in B.nodes():
                    B.add_node(target, bipartite=1)
                B.add_edge(source, target)

        if args.debug:
            print("EDGES:", B.edges())
        scene_nodes = set(n for n,d in B.nodes(data=True) if d['bipartite']==0)
        person_nodes = set(B) - scene_nodes
        nx.is_bipartite(B)
        G = nx.bipartite.weighted_projected_graph(B, person_nodes)
        return G

    def analyze_graph(self):
        """
        Computes various network metrics for a graph G,
        returns a dictionary:
        values =
        {
            "charcount" = len(G.nodes()),
            "edgecount" = len(G.edges()),
            "maxdegree" = max(G.degree().values()) or "NaN" if ValueError: max() arg is an empty sequence,
            "avgdegree" = sum(G.degree().values())/len(G.nodes()) or "NaN" if ZeroDivisionError: division by zero,
            "density" = nx.density(G) or "NaN",
            "avgpathlength" = nx.average_shortest_path_length(G) or "NaN" if NetworkXError: Graph is not connected,
                                then it tries to get the average_shortest_path_length from the giant component,
            "avgpathlength" = nx.average_shortest_path_length(max(nx.connected_component_subgraphs(G), key=len))
                                    except NetworkXPointlessConcept: ('Connectivity is undefined ', 'for the null graph.'),
            "clustering_coefficient" = nx.average_clustering(G) or "NaN" if ZeroDivisionError: float division by zero
        }
        """
        G = self.G
        values = {}
        values["charcount"] = len(G.nodes())
        values["edgecount"] = len(G.edges())
        try:
            values["maxdegree"] = max(G.degree().values())
        except:
            print("ValueError: max() arg is an empty sequence")
            values["maxdegree"] = "NaN"

        try:
            values["avgdegree"] = sum(G.degree().values())/len(G.nodes())
        except:
            print("ZeroDivisionError: division by zero")
            values["avgdegree"] = "NaN"

        try:
            values["density"] = nx.density(G)
        except:
            values["density"] = "NaN"

        try:
            values["avgpathlength"] = nx.average_shortest_path_length(G)
        except nx.NetworkXError:
            print("NetworkXError: Graph is not connected.")
            try:
                values["avgpathlength"] = nx.average_shortest_path_length(max(nx.connected_component_subgraphs(G), key=len))
            except:
                values["avgpathlength"] = "NaN"
        except:
            print("NetworkXPointlessConcept: ('Connectivity is undefined ', 'for the null graph.')")
            values["avgpathlength"] = "NaN"

        try:
            values["clustering_coefficient"] = nx.average_clustering(G)
        except:
            print("ZeroDivisionError: float division by zero")
            values["clustering_coefficient"] = "NaN"
        return values

    def analyze_characters(self, G):
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
        character_values = {}
        character_values["betweenness"] = nx.betweenness_centrality(G)
        character_values["degree"] = nx.degree(G)
        character_values["closeness"] = nx.closeness_centrality(G)
        return character_values

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

    def export_dict(self, d, filepath):
        with open(filepath, 'w') as f:  # Just use 'w' mode in 3.x
            w = csv.DictWriter(f, d.keys())
            w.writeheader()
            w.writerow(d)

    def export_dicts(self, d, filepath):
        with open(filepath, 'w') as f:  # Just use 'w' mode in 3.x
            w = csv.writer(f, delimiter=";")
            d = self.transpose_dict(d)
            try:
                subkeys = list(list(d.values())[0].keys())
                w.writerow([""] + subkeys)
                for k, v in d.items():
                    w.writerow([k] + list(v.values()))
            except:
                print("Empty values.")

    def randomize_graph(self, n, e):
        """
        Creates 1000 random graphs with networkx.gnm_random_graph(nodecount, edgecount),
        and computes average_clustering_coefficient and average_shortest_path_length,
        to compare with drama-graph.
        Returns a tuple:
        randavgpathl, randcluster = (float or "NaN", float or "NaN")
        """
        randcluster = 0
        randavgpathl = 0
        c = 0
        a = 0

        for i in range(0, 1000):
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


def main(args):
    corpus = LinaCorpus(args.inputfolder, args.outputfolder)
    if args.action == "plotsuperposter":
        plot_superposter(corpus, args.outputfolder, args.debug)
    if args.action == "metrics":
        corpus.get_metrics()
    if args.action == "char_ranks":
        corpus.get_central_characters()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='analyze and plot from lina-xml to networks')
    parser.add_argument('--input', dest='inputfolder', help='relative or absolute path of the input-xmls folder')
    parser.add_argument('--output', dest='outputfolder', help='relative or absolute path of the output folder')
    parser.add_argument('--action', dest='action', help='what to do, either plotsuperposter, metrics, char_ranks')
    parser.add_argument('--debug', dest='debug', help='print debug message or not', action="store_true")
    parser.add_argument('--randomization', dest='random', help='plot randomized graphs', action="store_true")
    args = parser.parse_args()
    main(args)
