#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# dramavis by frank fischer (@umblaetterer) & christopher kittel (@chris_kittel)

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import math

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

def plot_superposter(corpus, outputdir, debug=False):
    """
    Plot harmonically layoutted drama network subplots in 16:9 format.
    Node size by degree centrality,
    edge size by log(weight+1).
    """
    size = corpus.size
    y = int(math.sqrt(size/2)*(16/9))
    x = int(size/y)+1

    fig = plt.figure(figsize = (160,90))
    gs = gridspec.GridSpec(x, y)
    gs.update(wspace=0.0, hspace=0.00) # set the spacing between axes.
    i = 0

    # build rectangle in axis coords for text plotting
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    dramas = {drama.ID:drama for drama in corpus.read_dramas(metrics=False)}
    id2date = {drama.ID:drama.metadata.get("date_definite") for drama in corpus.read_dramas(metrics=False)}
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
        layout=nx.spring_layout
        pos=layout(G)

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
        nx.draw_networkx_nodes(G,pos,
                            nodelist=G.nodes(),
                            node_color=node_color,
                            node_size=nodesize,
                            alpha=0.8)

        weights = [math.log(G[u][v]['weight']+1)  for u,v in G.edges()]

        edge_color = "grey"
        nx.draw_networkx_edges(G,pos,
                               with_labels=False,
                               edge_color=edge_color,
                               width=weights
                            )



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

    fig.savefig(os.path.join(outputdir,"superposter.svg"))
    plt.close(fig)
