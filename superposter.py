#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# dramavis by frank fischer (@umblaetterer) & christopher kittel (@chris_kittel)

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import math
import numpy as np
from tqdm import tqdm


def plot_superposter(corpus, outputdir, debug=False):
    """
    Plot harmonically layoutted drama network subplots in 16:9 format.
    Node size by degree centrality,
    edge size by log(weight+1).
    """
    size = corpus.size
    y = int(math.sqrt(size/2)*(16/9))
    x = int(size/y)+1

    fig = plt.figure(figsize=(160, 90))
    gs = gridspec.GridSpec(x, y)
    gs.update(wspace=0.0, hspace=0.00)  # set the spacing between axes.
    i = 0

    # build rectangle in axis coords for text plotting
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    dramas = {drama.ID: drama
              for drama in corpus.analyze_dramas(action=None)}
    id2date = {drama.ID: drama.metadata.get("date_definite")
               for drama in corpus.analyze_dramas(action=None)}
    if debug:
        print(id2date)

    # http://pythoncentral.io/how-to-sort-python-dictionaries-by-key-or-value/
    sorted_by_date = sorted(id2date, key=id2date.__getitem__)

    for ID in sorted_by_date:
        drama = dramas.get(ID)
        if debug:
            print(drama.metadata)

        G = drama.G

        try:
            # for networks with only one node
            d = nx.degree_centrality(G)
            nodesize = [v * 110 for v in d.values()]
        except:
            nodesize = [1 * 110 for n in G.nodes()]
        layout = nx.spring_layout
        pos = layout(G)

        ax = plt.subplot(gs[i])
        ax.tick_params(color='white', labelcolor='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')

        if "Goethe" in drama.metadata.get("author"):
            ax.patch.set_facecolor('firebrick')
            ax.patch.set_alpha(0.2)
        if "Hebbel" in drama.metadata.get("author"):
            ax.patch.set_facecolor('purple')
            ax.patch.set_alpha(0.2)
        if "Weißenthurn" in drama.metadata.get("author"):
            ax.patch.set_facecolor('darkgreen')
            ax.patch.set_alpha(0.2)
        if "Schiller" in drama.metadata.get("author"):
            ax.patch.set_facecolor('darkslategrey')
            ax.patch.set_alpha(0.2)
        if "Wedekind" in drama.metadata.get("author"):
            ax.patch.set_facecolor('darkslateblue')
            ax.patch.set_alpha(0.2)
        if "Schnitzler" in drama.metadata.get("author"):
            ax.patch.set_facecolor('tomato')
            ax.patch.set_alpha(0.2)

        node_color = "steelblue"
        nx.draw_networkx_nodes(G, pos,
                               nodelist=G.nodes(),
                               node_color=node_color,
                               node_size=nodesize,
                               alpha=0.8)

        weights = [math.log(G[u][v]['weight']+1)
                   for u, v in G.edges()]

        edge_color = "grey"
        nx.draw_networkx_edges(G, pos,
                               with_labels=False,
                               edge_color=edge_color,
                               width=weights)

        title_bark = "".join([w[0] for w in drama.title.split()])
        caption = ", ".join([drama.metadata.get("author").split(",")[0],
                             title_bark,
                             str(drama.metadata.get("date_definite"))])

        ax.text(0.5*(left+right), 0*bottom, caption,
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=20, color='black',
                transform=ax.transAxes)

        ax.set_frame_on(True)
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)

        i += 1

    fig.savefig(os.path.join(outputdir, "superposter.svg"))
    plt.close(fig)


def plot_quartett_poster(corpus, outputdir):
    dramas = {drama.ID: drama
              for drama in corpus.analyze_dramas(action="both")}
    id2date = {drama.ID: drama.metadata.get("date_definite")
               for drama in corpus.analyze_dramas(action=None)}
    sorted_by_date = sorted(id2date, key=id2date.__getitem__)

    x = 4
    y = 8
    fig = plt.figure(figsize=(80, 80))
    outer = gridspec.GridSpec(x, y)
    outer.update(wspace=0.06, hspace=0.06)  # set the spacing between axes.
    i = 0
    for ID in tqdm(sorted_by_date, desc="Plotting"):
        drama = dramas.get(ID)

        inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                 subplot_spec=outer[i],
                                                 wspace=0.0, hspace=0.0)
        # inner = outer.new_subplotspec(i, rowspan=2)
        # inner.update(wspace=0.125, hspace=0.125)
        G = drama.G

        # PLOT NETWORK
        ax = plt.subplot(inner[0])
        try:
            # for networks with only one node
            d = nx.degree_centrality(G)
            nodesize = [v * 110 for v in d.values()]
        except:
            nodesize = [1 * 110 for n in G.nodes()]
        layout = nx.spring_layout
        pos = layout(G)
        ax.tick_params(color='white', labelcolor='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')

        node_color = "steelblue"
        nx.draw_networkx_nodes(G, pos,
                               nodelist=G.nodes(),
                               node_color=node_color,
                               node_size=nodesize,
                               alpha=0.8)

        weights = [math.log(G[u][v]['weight']+1)
                   for u, v in G.edges()]

        edge_color = "grey"
        nx.draw_networkx_edges(G, pos,
                               with_labels=False,
                               edge_color=edge_color,
                               width=weights)

        title_bark = "".join([w[0] for w in drama.title.split()])
        caption = ", ".join([drama.metadata.get("author").split(",")[0],
                             title_bark,
                             str(drama.metadata.get("date_definite"))])

        ax.text(0.5, 0.0, caption,
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=30, color='black',
                transform=ax.transAxes)

        ax.set_frame_on(True)
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        fig.add_subplot(ax)

        # PLOT TEXTBOX

        text_ax = plt.subplot(inner[0])
        # Autor*in – Titel – Untertitel – Jahr
        metadata = ['author', 'title', 'subtitle', 'date_definite']
        metadata = [": ".join([md, str(drama.metadata.get(md))])
                    for md in metadata]
        metadata = "\n".join(metadata)
        # Anzahl von Subgraphen – Netzwerkgröße – Netzwerkdichte –
        # Clustering-Koeffizient – Durchschnittliche Pfadlänge –
        # Höchster Degreewert und Name der entsprechenden Figur, all-in index
        metrics = ['charcount', 'density', 'connected_components',
                   'clustering_coefficient', 'avgpathlength',
                   'all_in_index']
        metric_strings = []
        for metric in metrics:
            v = drama.graph_metrics.loc[drama.ID][metric]
            if type(v) == int:
                value = "%.d" % v
            if type(v) == float:
                value = "%.2f" % v
            if metric == "all_in_index":
                v = int(v*100)
                value = "%.d" % v +"%"
            else:
                value = str(value)
            metric_strings.append(": ".join([metric, value]))
        max_degree = drama.graph_metrics.loc[drama.ID]['maxdegree']
        cent_max = drama.centralities['degree'].max()
        top_char = drama.centralities[drama.centralities['degree'] == cent_max].index.tolist()
        if len(top_char) != 1:
            max_degree_char = "SEVERAL"
        else:
            max_degree_char = top_char[0]
        metric_strings.append('max_degree: %.d, (%s)' % (max_degree, max_degree_char))
        metric_strings = "\n".join(metric_strings)
        text_ax.text(0, -0.15, metadata+"\n"+"\n"+metric_strings,
                     ha='left', va="top",
                     wrap=True, transform=text_ax.transAxes,
                     fontsize=20)

        text_ax.set_frame_on(True)
        text_ax.axes.get_yaxis().set_visible(False)
        text_ax.axes.get_xaxis().set_visible(False)
        fig.add_subplot(text_ax)

        i += 1
    # plt.tight_layout()

    fig.savefig(os.path.join(outputdir, "quartettposter.svg"))
    plt.close(fig)
