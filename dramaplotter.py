#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# dramavis by frank fischer (@umblaetterer) & christopher kittel (@chris_kittel)

import os
import networkx as nx

from linacorpus import LinaCorpus

__author__ = "Christopher Kittel <web at christopherkittel.eu>, Frank Fischer <ffischer at hse.ru>"
__copyright__ = "Copyright 2017"
__license__ = "MIT"
__version__ = "0.4 (beta)"
__maintainer__ = "Frank Fischer <ffischer at hse.ru>"
__status__ = "Development" # 'Development', 'Production' or 'Prototype'

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
            if os.path.isfile(os.path.join(self.outputfolder,
                                           "_".join([str(drama.ID),
                                                     drama.title])+".svg")):
                continue
            self.capture_fringe_cases(drama)
            if randomization:
                for i in range(0, 5):
                    R = nx.gnm_random_graph(drama.graph_metrics.get("charcount"),
                                            drama.graph_metrics.get("edgecount"))
                    plotGraph(R, filename=os.path.join(self.outputfolder,
                                                       str(drama.ID)+"random"+str(i)+".svg"))

    def capture_fringe_cases(self, drama):
        if drama.graph_metrics.get("all_in_index") is None:
            print(drama.title)

class QuartettPlotter(LinaCorpus):
    """
    """


def plotGraph(G, figsize=(8, 8), filename=None):
    """
    Plots an individual graph, node size by degree centrality,
    edge size by edge weight.
    """
    labels = {n:n for n in G.nodes()}

    try:
        # for networks with only one node
        d = nx.degree_centrality(G)
        nodesize = [v * 250 for v in d.values()]
    except:
        nodesize = [1 * 250 for n in G.nodes()]

    layout=nx.spring_layout
    pos=layout(G)

    plt.figure(figsize=figsize)
    plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0.01,hspace=0.01)

    # nodes
    nx.draw_networkx_nodes(G,pos,
                            nodelist=G.nodes(),
                            node_color="steelblue",
                            node_size=nodesize,
                            alpha=0.8)
    try:
        weights = [G[u][v]['weight'] for u,v in G.edges()]
    except:
        weights = [1 for u,v in G.edges()]
    nx.draw_networkx_edges(G,pos,
                           with_labels=False,
                           edge_color="grey",
                           width=weights
                        )

    if G.order() < 1000:
        nx.draw_networkx_labels(G,pos, labels)
    plt.savefig(filename)
    plt.close("all")
