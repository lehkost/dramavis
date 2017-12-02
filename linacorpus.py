#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# dramavis by frank fischer (@umblaetterer) & christopher kittel (@chris_kittel)

import os
import glob
from lxml import etree
from itertools import chain
import pandas as pd


__author__ = "Christopher Kittel <web at christopherkittel.eu>, Frank Fischer <ffischer at hse.ru>"
__copyright__ = "Copyright 2017"
__license__ = "MIT"
__version__ = "0.4 (beta)"
__maintainer__ = "Frank Fischer <ffischer at hse.ru>"
__status__ = "Development" # 'Development', 'Production' or 'Prototype'


class LinaCorpus(object):
    """docstring for LinaCorpus"""
    def __init__(self, inputfolder, outputfolder):
        self.inputfolder = inputfolder
        self.outputfolder = outputfolder
        self.dramafiles = glob.glob(os.path.join(self.inputfolder, '*.xml'))
        self.size = len(self.dramafiles)
        if not os.path.isdir(outputfolder):
            os.mkdir(outputfolder)


class Lina(object):
    """docstring for Lina"""
    def __init__(self, dramafile, outputfolder):
        self.outputfolder = outputfolder
        self.tree = etree.parse(dramafile)
        self.filename = os.path.splitext(os.path.basename((dramafile)))[0]
        self.parse_drama()
        self.num_chars_total = len(self.personae)
        self.filepath = os.path.join(self.outputfolder, str(self.ID))
        self.title = self.metadata.get("title", self.ID)


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
        self.ID = root.attrib.get("id")
        header = root.find("{*}header")
        persons = root.find("{*}personae")
        text = root.find("{*}text")
        self.metadata = self.extract_metadata(header)
        self.metadata["filename"] = self.filename
        self.personae = self.extract_personae(persons)
        self.charmap = self.create_charmap()
        self.segments = self.extract_speakers()
        self.metadata["segment_count"] = len(self.segments)
        self.metadata["count_type"] = self.get_count_type()
        # parsed_drama = (ID, {"metadata": metadata, "personae":personae, "speakers":speakers})
        # return parsed_drama

        # if args.debug:
        #     print("SEGMENTS:", segments)

    def extract_metadata(self, header):
        """
        Extracts metadata from the header-tag of a lina-xml,
        returns dictionary:

        metadata = {
             'author': 'Benedix, Julius Roderich',
             'count_type': 'scenes',
             'date_definite': 1862,
             'date_premiere': None,
             'date_print': 1862,
             'date_written': None,
             'filename': '1862-Benedix_Julius_Roderich-Die_Lügnerin-lina',
             'genretitle': 'Lustspiel',
             'pnd': 'Die Lügnerin',
             'segment_count': 8,
             'source_textgrid': 'https://textgridlab.org/1.0/tgcrud-public/rest/textgrid:kjfz.0/data',
             'subtitle': 'Lustspiel in einem Aufzuge',
             'title': 'Die Lügnerin'}
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
        returns dict of Character objects, {name:Character}:
        """
        personae = {}
        for char in persons.getchildren():
            name = char.find("{*}name").text
            aliases = [alias.attrib
                            .get('{http://www.w3.org/XML/1998/namespace}id')
                       for alias in char.findall("{*}alias")]
            # if args.debug:
            #     print("ALIASES:", aliases)
            if name:
                personae[name] = Character(name, aliases)
            else:
                name = aliases[0]
                personae[name] = Character(name, aliases)
        # if args.debug:
        #     print("PERSONAE:", personae)
        return personae

    def extract_structure(self):
        """Returns list of etree-elements"""
        text = self.tree.getroot().find("{*}text")
        speakers = text.findall(".//{*}sp")
        parentsegments = list()
        for speaker in speakers:
            parent = speaker.getparent()
            head = parent.getchildren()[0]
            if parent not in parentsegments:
                parentsegments.append(parent)
            # check if scene (ends with "Szene/Szene./Auftritt/Auftritt.")
        return parentsegments

    def extract_speakers(self):
        """ e.g.
        [['CONSTANZE', 'LANGENBERG'],
         ['GUSTCHEN', 'CONSTANZE', 'LANGENBERG'],
         ['BACKES', 'HAHNENBEIN', 'CONSTANZE', 'GUSTCHEN', 'MORITZ'],
         ['CONSTANZE', 'GUSTCHEN', 'MORITZ'],
         ['FR. GREINER', 'CONSTANZE', 'MORITZ'],
         ['BACKES', 'HAHNENBEIN', 'GUSTCHEN', 'CONSTANZE', 'MORITZ'],
         ['HAHNENBEIN', 'CONSTANZE', 'BACKES', 'GUSTCHEN', 'MORITZ', 'HAUPTMANN'],
         ['HAHNENBEIN', 'CONSTANZE', 'BACKES', 'GUSTCHEN', 'MORITZ', 'HAUPTMANN', 'LANGENBERG']]
        """
        parentsegments = self.extract_structure()
        segments = list()
        for i, segment in enumerate(parentsegments):
            speakers = [speaker.attrib.get("who").replace("#", "").split()
                        for speaker in segment.findall(".//{*}sp")]
            speakers = list(chain.from_iterable(speakers))
            for speaker in speakers:
                for amount in ["speech_acts", "words", "lines", "chars"]:
                    try:
                        spk = self.charmap[speaker]
                    except:
                        pass
                    try:
                        n = segment.findall(".//{*}sp[@who='*#%s']/{*}amount[@unit='%s']" %(speaker, amount))[0].attrib.get('n')
                        n = int(n)
                        self.personae[spk].amounts[amount] += n
                    except Exception as e:
                        continue
            speakers = [self.charmap[speaker]
                        for speaker in speakers
                        if speaker in self.charmap]
            speakers = list(set(speakers))
            for name in speakers:
                self.personae[name].appears_in.add(i)
            segments.append(speakers)
        return segments

    def get_count_type(self):
        """
        type of segment counting, default: acts, else scenes
        """
        text = self.tree.getroot().find("{*}text")
        speakers = text.findall(".//{*}sp")
        count_type = "acts"
        for speaker in speakers:
            parent = speaker.getparent()
            head = parent.getchildren()[0]
            if (head.text.endswith("ne") or
                head.text.endswith("ne.") or
                head.text.endswith("tt") or
                head.text.endswith("tt.")):
                count_type = "scenes"
                # print(head.text)
        return count_type

    def create_charmap(self):
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
        for person in self.personae.values():
            for alias in person.aliases:
                charmap[alias] = person.name
        return charmap


class Character(object):
    """docstring for Character"""
    def __init__(self, name, aliases):
        super(Character, self).__init__()
        self.name = name
        self.aliases = aliases
        self.appears_in = set() # which segments
        self.amounts = {"speech_acts":0,
                        "words":0,
                        "lines":0,
                        "chars":0}
        self.data = pd.DataFrame()
