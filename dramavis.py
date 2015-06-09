#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# dramavis by frank fischer (@umblaetterer) & christopher kittel (@chris_kittel)

import os
import csv
import igraph as ig
from igraph.drawing.text import TextDrawer
import cairo
from django.template import Context, Template
from django.conf import settings

# Where are the CSV files stored?
inputfolder = "input/"
outputfolder = "output/"

settings.configure()


def main():
    all_drama_csvs = (read_csv_files(inputfolder))
    for single_drama_csv in all_drama_csvs:
        parse_single_csv(single_drama_csv)
    parsed_dramas = [parse_single_csv(single_drama_csv) for single_drama_csv in all_drama_csvs]
    
    print("Building graph objects for each drama.")
    for drama in parsed_dramas:
        drama["graph"] = create_graph(drama)

    print("Analyzing drama-related network metrics.")
    for drama in parsed_dramas:
        drama["values"] = analyze_graph(drama.get("graph"))

    print("Analyzing character-related network metrics.")
    for drama in parsed_dramas:
        drama["charvalues"] = analyze_characters(drama.get("graph"))

    print("Exporting network metrics to CSV file.")
    export2table(parsed_dramas)
    print("Exporting character metrics to HTML file.")
    export2html(parsed_dramas)

    progress = 0
    maxi = len(parsed_dramas)
    print("Writing PNG graphs of each drama network.")
    for drama in parsed_dramas:
        plot(drama)
        update_progress(progress/maxi*100)
        progress += 1
    print("\nFinished.")


def update_progress(progress):
    print("\r[{0}] {1}%".format("#"*(int(progress/10)), int(progress)), end='')


def read_csv_files(inputfolder):
    """
    Reads all files in the inputfolder,
    returns a list of dramas.
    """
    filelist = [f for f in os.listdir(inputfolder)]
    return filelist


def get_filename(filepath):
    root, ext = os.path.splitext(filepath)
    head, tail = os.path.split(root)
    return tail


def parse_single_csv(single_drama_csv):
    """
    Reads a drama CSV and returns a dict,
    containing as keys 'name' and 'relations',
    where value of relations is a list of dicts
    {source:"A", target:"B", weight:3}.
    """
    parsed_drama = {}
    with open(inputfolder+single_drama_csv, "r") as infile:
        dramareader = csv.reader(infile, delimiter=";")
        parsed_drama["title"] = get_filename(single_drama_csv)

        relations =  []
        for rel in dramareader:
            try:
                source, target, weight = rel
            except:
                parsed_drama["relations"] = None

            source = source.strip()
            target = target.strip()
            weight = int(weight)
            
            relations.append({"source":source,
                    "target":target,
                    "weight":weight})

        parsed_drama["relations"] = relations

    return parsed_drama


def create_graph(parsed_drama):
    """
    Takes a drama dict and returns an igraph graph object.
    """
    G = ig.Graph() # generate empty graph
    if not parsed_drama.get("relations"):
        G = ig.Graph.Formula()
    for rel in parsed_drama.get("relations"):
        # check whether source or target already exist as node
        # if not, add
        try:
            G.vs.find(rel.get("source"))
        except:
            G.add_vertex(rel.get("source"), text=rel.get("source"))
        try:
            G.vs.find(rel.get("target"))
        except:
            G.add_vertex(rel.get("target"), text=rel.get("target"))
        # add edge
        G.add_edge(rel.get("source"), rel.get("target"), weight=rel.get("weight"))

    G.simplify(multiple=True, combine_edges={"weight":"min"})

    return G


def analyze_graph(G):
    values = {}
    values["charcount"] = get_nr_of_characters(G)
    values["maxdegree"] = get_maxdegree(G)
    values["avgdegree"] = get_avgdegree(G)
    values["density"] = get_density(G)
    values["avgpathlength"] = get_avgpathlength(G)
    return values


def get_nr_of_characters(G):
    return len(G.vs)

def get_maxdegree(G):
    return round(G.maxdegree(), 2)

def get_avgdegree(G):
    return round(ig.mean(G.degree()), 2)

def get_density(G):
    return round(G.density(), 2)

def get_avgpathlength(G):
    return round(G.average_path_length(), 2)


def analyze_characters(G):
    character_values = []
    for char in G.vs:
        charvalue = {}
        charvalue["name"] = char["name"]
        charvalue["betweenness"] = round(G.betweenness(char), 2)
        charvalue["degree"] = round(G.degree(char), 2)
        charvalue["avgdistance"] = round(1/G.closeness(char), 2)
        charvalue["closeness"] = round(G.closeness(char), 2)
        character_values.append(charvalue)
    return character_values


def export2html(parsed_dramas):
    dj_template ="""
                <html>
                    <head>
                        <title>Drama Character Values</title>
                        <meta http-equiv="content-type" content="text/html; charset=utf-8">
                    </head>
                        <body>
                            {# dramadata #}
                            {% for drama, rows in dramadata %}
                                <h1>{{drama}}</h1>
                                <table border="1" cellspacing="0" cellpadding="4">
                                {# headings #}
                                    <tr>
                                    {% for heading in headings %}
                                        <th>{{ heading }}</th>
                                    {% endfor %}
                                    </tr>
                                {# rows #}
                                {% for row in rows %}
                                    <tr align="center">
                                        {% for val in row %}
                                        <td>{{ val|default:'' }}</td>
                                        {% endfor %}
                                    </tr>
                                {% endfor %}
                                </table>
                            {% endfor %}
                        </body>
                </html>
                """

    headings = ["Character",
                "Degree",
                "Betweenness Centrality",
                "Average Distance",
                "Closeness Centrality"]

    dramanames = [drama.get("title") for drama in parsed_dramas]
    dramadata = []
    for drama in parsed_dramas:
        chardata = []
        for char in drama.get("charvalues"):
            row = [char.get("name"),
                    char.get("degree"),
                    char.get("betweenness"),
                    char.get("avgdistance"),
                    char.get("closeness")]
            chardata.append(row)
            # sort by degree
            chardata.sort(key=lambda x: x[1], reverse=True)
        dramadata.append((drama.get("title"), chardata))
    #sort by date
    dramadata.sort(key=lambda x: x[0])
    

    tmpl = Template(dj_template)
    with open(outputfolder+"drama_character_values.html", "w") as outfile:
        outfile.write(tmpl.render(Context(dict(dramadata=dramadata, headings=headings))))


def export2table(parsed_dramas):
    with open(outputfolder+"drama_network_values.csv", "w") as outfile:
        csvwriter = csv.writer(outfile, delimiter=";")
        csvwriter.writerow(["Title",
                            "Number of characters",
                            "Max Degree",
                            "Average Degree",
                            "Density",
                            "Average Path Length"])
        for drama in parsed_dramas:
            values = drama.get("values")
            csvwriter.writerow([drama.get("title"),
                                values.get("charcount"),
                                values.get("maxdegree"),
                                values.get("avgdegree"),
                                values.get("density"),
                                values.get("avgpathlength")])


def plot(drama, caption=True):

    plot = ig.Plot(outputfolder+drama.get("title")+".png",
                                    bbox=(600, 600), background="white")    

    try:
        graph = ig.VertexClustering(drama.get("graph")).giant()
        visual_style = {}
        visual_style["layout"] = graph.layout_fruchterman_reingold()
        visual_style["vertex_color"] = "#0000ff"
        visual_style["vertex_shape"] = "rectangle"
        visual_style["vertex_size"] = 8
        visual_style["vertex_label"] = graph.vs["name"]
        visual_style["vertex_label_size"] = 15
        visual_style["vertex_label_dist"] = 1.5
        visual_style["edge_color"] = "#6495ed"
        visual_style["edge_width"] = graph.es["weight"]
        visual_style["bbox"] = (600, 600)
        visual_style["margin"] = 50
        plot.add(graph, **visual_style)
    except:
        pass


    if caption:
       # Make the plot draw itself on the Cairo surface.
        plot.redraw()

        # Grab the surface, construct a drawing context and a TextDrawer.
        ctx = cairo.Context(plot.surface)
        ctx.set_font_size(15)
        drawer = TextDrawer(ctx, drama.get("title"),
                            halign=TextDrawer.CENTER)
        drawer.draw_at(0, 597, width=600)

    plot.save()



if __name__ == "__main__":
    main()
