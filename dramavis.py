#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# dramavis by frank fischer (@umblaetterer) & christopher kittel (@chris_kittel)

__author__ = "Christopher Kittel <web at christopherkittel.eu>, Frank Fischer <frank.fischer at sub.uni-goettingen.de>"
__copyright__ = "Copyright 2016"
__license__ = "MIT"
__version__ = "0.2"
__maintainer__ = "Frank Fischer <frank.fischer at sub.uni-goettingen.de>"
__status__ = "Development" # 'Development', 'Production' or 'Prototype'

from lxml import etree
import os
import glob
import networkx as nx
import csv
from itertools import chain
import argparse
from superposter import plotGraph, plot_superposter


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

    def get_metrics(self):
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
            #     print(drama.title)
            if os.path.isfile(os.path.join(self.outputfolder, str(drama.ID)+drama.title+".svg")):
                continue
            speakers = drama.speakers
            personae = drama.personae
            # if args.debug:
            #     print(personae)
            #     print(speakers)

            drama.write_output()

            for i in range(0, 5):
                R = nx.gnm_random_graph(drama.graph_metrics.get("charcount"), drama.graph_metrics.get("edgecount"))
                plotGraph(R, filename=os.path.join(self.outputfolder, str(drama.ID)+"random"+str(i)+".svg"))


class Lina(object):
    """docstring for Lina"""
    def __init__(self, dramafile, outputfolder, metrics=True):
        self.outputfolder = outputfolder
        self.tree = etree.parse(dramafile)
        self.filename = os.path.splitext(os.path.basename((dramafile)))[0]
        ID, metadata, personae, speakers = self.parse_drama()
        self.ID = ID
        self.metadata = metadata
        self.personae = personae
        self.speakers = speakers
        self.filepath = os.path.join(self.outputfolder, str(self.ID))
        self.title = self.metadata.get("title", self.ID)
        self.G = self.create_graph()
        if metrics:
            self.graph_metrics = self.get_graph_metrics()
            self.character_metrics = self.get_character_metrics()


    def write_output(self):
        self.export_dict(self.graph_metrics, self.filepath+self.title+"graph.csv")
        self.export_dicts(self.character_metrics, self.filepath+self.title+"chars.csv")
        nx.write_edgelist(self.G, os.path.join(self.outputfolder, str(self.ID)+self.title+"edgelist.csv"), delimiter=";", data=["weight"])
        plotGraph(self.G, filename=os.path.join(self.outputfolder, str(self.ID)+self.title+".svg"))

    def get_graph_metrics(self):
        graph_metrics = self.analyze_graph()
        graph_metrics["ID"] = self.ID
        graph_metrics["average_path_length_random"], graph_metrics["clustering_coefficient_random"] = self.randomize_graph(graph_metrics.get("charcount"), graph_metrics.get("edgecount"))
        graph_metrics["year"] = self.metadata.get("date_print")
        graph_metrics["author"] = self.metadata.get("author")
        graph_metrics["title"] = self.title
        graph_metrics["filename"] = self.metadata.get("filename")
        graph_metrics["genretitle"] = self.metadata.get("genretitle")
        graph_metrics["scenecount"] = self.metadata.get("scenecount")
        return graph_metrics

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
            "speakers":speakers
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
        speakers, scene_count = self.extract_speakers(text)
        metadata["scenecount"] = scene_count
        # parsed_drama = (ID, {"metadata": metadata, "personae":personae, "speakers":speakers})
        # return parsed_drama
        return ID, metadata, personae, speakers

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
        if type(date_print) != int:
            date_print = 9999
        if type(date_written) != int:
            date_print = 9999
        if type(date_premiere) != int:
            date_print = 9999

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
            # if args.debug:
            #     print(aliases)
            if name:
                personae.append({name:aliases})
            else:
                personae.append({aliases[0]:aliases})
        # if args.debug:
        #     print(personae)
        return personae

    def extract_structure(self):
        text = self.tree.getroot().find("{*}text")
        acts = dict()
        scenes = dict()
        sps = text.findall(".//{*}sp")
        scene_count = 1
        act_count = 1
        parents = set()
        for sp in sps:
            parent = sp.getparent()
            head = parent.getchildren()[0]
            # check if act (ends with (Akt/Akt./Aufzug/Aufzug.))
            if head.text.endswith("kt") or head.text.endswith("kt.") or head.text.endswith("ug") or head.text.endswith("ug."):
                # print(head.text)
                if not head.text in acts:
                    acts[act_count] = list()
                    speaker = sp.attrib.get("who").replace("#","").split()
                    acts[act_count].extend(speaker)
                if parent not in parents:
                    act_count += 1
            # check if scene (ends with "Szene/Szene./Auftritt/Auftritt.")
            if head.text.endswith("ne") or head.text.endswith("ne.") or head.text.endswith("tt") or head.text.endswith("tt."):
                # print(head.text)
                if not head.text in scenes:
                    scenes[scene_count] = list()
                    speaker = sp.attrib.get("who").replace("#","").split()
                    scenes[scene_count].extend(speaker)
                if parent not in parents:
                    scene_count += 1
            parents.add(parent)
        return acts, scenes

    def extract_speakers(self, text):
        """
        Extracts speakers that appear in the same scene,
        returns a dict of dict of lists, and the overall scene count:
        acts =
        {"act1":
            {"scene1":["speaker1", "speaker2"],
            "scene2":["speaker2", "speaker3"]},
         "act2":
            {"scene1":["speaker3", "speaker2"],
            "scene2":["speaker2", "speaker1"]}
            }
        }
        scene_count = 4

        """
        acts = {}
        scene_count = 0
        for act in text.getchildren():
            try:
                actname = act.find("{*}head").text
            except:
                actname = str(scene_count)
            if not actname:
                actname = str(scene_count)
                scene_count += 1
            acts[actname] = {}

            for scene in act.getchildren():
                try:
                    scenename = scene.find("{*}head").text
                except:
                    scenename = str(scene_count)
                if not scenename:
                    scenename = str(scene_count)
                speakers = [speaker.attrib.get("who").replace("#","").split() for speaker in scene.findall(".//{*}sp")]
                speakers = list(chain.from_iterable(speakers))
                if speakers:
                    acts[actname][scenename] = speakers
                scene_count += 1
        return acts, scene_count

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
        speakerset = self.speakers
        personae = self.personae
        charmap = self.create_charmap(personae)
        # if args.debug:
        #     print(charmap)

        B = nx.Graph()
        labels = {}
        for act, scenes in speakerset.items():
            for scene, speakers in scenes.items():
                try:
                    source = " ".join([act, scene])
                except TypeError:
                    source = " ".join([scene, scene])
                targets = speakers

                if not source in B.nodes():
                    B.add_node(source, bipartite=0)
                    labels[source] = source

                for target in targets:
                    target = charmap.get(target)
                    if not target in B.nodes():
                        B.add_node(target, bipartite=1)
                    B.add_edge(source, target)

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
            values["avgdegree"] = "NaN"

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

    def randomize_graph(self, n,e):
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
            randavgpathl = randavgpathl / 1000
        except:
            randavgpathl = "NaN"
        return randavgpathl, randcluster


def main(args):
    corpus = LinaCorpus(args.inputfolder, args.outputfolder)
    if args.action == "plotsuperposter":
        plot_superposter(corpus, args.outputfolder, args.debug)
    if args.action == "metrics":
        corpus.get_metrics()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='analyze and plot from lina-xml to networks')
    parser.add_argument('--input', dest='inputfolder', help='relative or absolute path of the input-xmls folder')
    parser.add_argument('--output', dest='outputfolder', help='relative or absolute path of the output folder')
    parser.add_argument('--action', dest='action', help='what to do, either plotsuperposter or metrics')
    parser.add_argument('--debug', dest='debug', help='print debug message or not', action="store_true")
    args = parser.parse_args()
    main(args)
