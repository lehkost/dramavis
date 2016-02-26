# dramavis, by Frank Fischer & Christopher Kittel
Spaghetti-code version written in August, 2014; rewritten in June, 2015.
Major rewrite in February, 2016:

* added superposter-plotter
* added random graph metrics and plots
* edgelist-export to csv
* made commandline-executable

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