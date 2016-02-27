# dramavis

**by Frank Fischer & Christopher Kittel**

Spaghetti-code version written in August, 2014;
rewritten in June, 2015;
major rewrite in February, 2016.

* added superposter-plotter
* added random graph metrics and plots
* edgelist-export to csv
* enabled command-line executability

Purposes of this Python script:

* reading character networks of dramatic pieces from lina-xml files and
* plotting these networks into SVG graphs,
* writing drama network and character metrics values to CSV files

Run:

```
python3 dramavis.py --input LINAXMLFOLDER --output OUTPUTFOLDER --action dramavis
```
or

```
python3 dramavis.py --input LINAXMLFOLDER --output OUTPUTFOLDER --action plotsuperposter
```

Running dramavis can take up to 30min, plotsuperposter up to 5.

# Input Data

You can download the dlina XMLs from [here](https://github.com/dlina/project/tree/master/data/zwischenformat)
(should be 465 XML files); it might be easier to get hold of them if you ``git clone``
our [DLINA project repository](https://github.com/dlina/project) and then extract the
XMLs from the /zwischenformat/ folder.
