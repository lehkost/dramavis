# dramavis

**by Frank Fischer ([@umblaetterer](https://twitter.com/umblaetterer)) & Christopher Kittel ([@chris_kittel](https://twitter.com/chris_kittel))**

## Purposes of this Python script:

* reading character networks of dramatic pieces from lina-xml files;
* plotting these networks into SVG graphs (and generating a superposter containing all graphs in chronological order, [see our showcase poster](https://dx.doi.org/10.6084/m9.figshare.3101203.v1));
* writing drama network and character metrics values to CSV files.

## Version history:

* v0.0: Spaghetti-code version written in August, 2014 (never published);
* v0.1: rewritten in June, 2015 (archived [here](https://github.com/lehkost/dramavis/tree/master/archive/v0.1));
* v0.2: major rewrite in February, 2016.

## New in v0.2:

* added superposter-plotter
* added random graph metrics and plots
* edgelist-export to csv
* enabled command-line executability

Run:

```
python3 dramavis.py --input LINAXMLFOLDER --output OUTPUTFOLDER --action dramavis
```
or

```
python3 dramavis.py --input LINAXMLFOLDER --output OUTPUTFOLDER --action plotsuperposter
```

Running dramavis can take up to 30min, plotsuperposter up to 5.

## Input data

You can download the dlina XMLs from [here](https://github.com/dlina/project/tree/master/data/zwischenformat)
(should be 465 XML files); it might be easier to get hold of them if you ``git clone``
our [DLINA project repository](https://github.com/dlina/project) and then extract the
XMLs from the /zwischenformat/ folder.
