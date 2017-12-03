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
* v0.4: rewritten in September 2017, reworked datamodel, added new metrics

## New in v0.4:

* reworked composite ranking index now based on 5 network-metrics and 3 content metrics (character-level)
* introduced Kendall-Tau measure for ranking stability (drama-level)
* reworked data model, now based on pandas (functions and workflow now cleaner and simpler)
* reworked package structure, separated into workflow, I/O, plotting, and analysis

## Installation

Depends on [Anaconda](https://www.continuum.io/downloads) for Python 3

Prepare:
```
conda env create -f dramavis.yml
source activate dramavis
```

Run:

```
python3 workflow.py --input /home/chris/data/dlina/data/zwischenformat --output charmetrics --action char_metrics --logpath all.log
```
* alternative actions: corpus_metrics, both

* additional flags
  * `--debug` prints alot of internal variables when running
  * `--randomization` prints randomized graphs, takes longer to run

Running dramavis can take up to 4 seconds per play with an average of 2.5 seconds, this is mainly due to the network randomization for statistics. plotsuperposter takes around 1 second per play.

## Input data

An easy way to download the dlina 'zwischenformat' XMLs without additional repo information is this:

```svn export https://github.com/dlina/project/trunk/data/zwischenformat```

Then just point the input directory to the cloned folder (which should include 465 XML files in the main directory).
