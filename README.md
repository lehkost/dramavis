# dramavis

**by Frank Fischer ([@umblaetterer](https://twitter.com/umblaetterer)) & Christopher Kittel ([@chris_kittel](https://twitter.com/chris_kittel))**

## Purposes of this Python script:

* reading character networks of dramatic pieces from lina-xml files;
* plotting these networks into SVG graphs (and generating a superposter containing all graphs in chronological order, [see our showcase poster](https://dx.doi.org/10.6084/m9.figshare.3101203.v1));
* writing drama network and character metrics values to CSV files.

## Version history:

* v0.0: Spaghetti-code version written in August, 2014 (never published);
* v0.1: rewritten in June, 2015 (archived [here](https://github.com/lehkost/dramavis/tree/master/archive/v0.1));
* v0.2: major rewrite in February, 2016;
* v0.2.1: minor bugfixes and usability improvements;
* v0.3: object-oriented restructurations of code base, introduced measures for dynamic network changes; December, 2016.

## New in v0.2.1:

* added virtualenv file

## New in v0.3:

* introduced values to describe dynamic changes in character networks for plot analysis, cf. [our paper for DHd2017 in Bern/CH](https://dlina.github.io/presentations/2017-bern/) (in German)

## Installation

Depends on [Anaconda](https://www.continuum.io/downloads) for Python 3

Prepare:
```
conda env create -f dramavis.yml
source activate dramavis
```

Run:

```
python3 dramavis.py --input LINAXMLFOLDER --output OUTPUTFOLDER --action metrics
```
or

```
python3 dramavis.py --input LINAXMLFOLDER --output OUTPUTFOLDER --action plotsuperposter
```

* additional flags
  * `--debug` prints alot of internal variables when running
  * `--randomization` prints randomized graphs, takes longer to run

Running dramavis can take up to 4 seconds per play with an average of 2.5 seconds, this is mainly due to the network randomization for statistics. plotsuperposter takes around 1 second per play.

## Input data

You can download the dlina XMLs from [here](https://github.com/dlina/project/tree/master/data/zwischenformat)
(should be 465 XML files); it might be easier to get hold of them if you ``git clone``
our [DLINA project repository](https://github.com/dlina/project) and then extract the
XMLs from the /zwischenformat/ folder.

## Plans for v0.3 (December 2016)

* main goal: introduce measures/formats for dynamic network analysis
* planned minor enhancements:
  * option to put labels on all nodes or a specific number of nodes per graph (based on node values like degree, average distance or betweenness centrality)
  * reintroduce option for input of CSV files (like in v0.1)
  * introduce option for output as CSV files (to further use them with Gephi, Cytoscape or the likes)
  * option for choosing type of spring-embedder layout
  * collect centrality valus over time and check how they change (is the character with the highest betweenness centrality in the first act still the one with the highest BC at the end?)
  * …
