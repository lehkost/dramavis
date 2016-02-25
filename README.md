# dramavis, by Frank Fischer & Christopher Kittel
Spaghetti-code version written in August, 2014; rewritten in June, 2015.

Purposes of this Python script:
* reading character networks of dramatic pieces from CSV files and
* plotting these networks into PNG graphs,
* writing drama network values to a CSV file,
* writing drama character values to an HTML file.

### Data schema

for each drama we create one dictionary:

{ID:
	{
	"metadata":{
		"title":"",
		"subtitle":"",
		"genretitle":"", 
		"author":"", 
		"pnd":"",
		"date_print":"",
		"date_written":"", 
		"date_premiere":"",
		"source_textgrid":""
		},
	"personae":
		{"character_name":["character_aliases"]
		},
	"network":{
		""
		}
	}
}