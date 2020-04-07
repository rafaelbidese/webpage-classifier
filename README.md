# Webpage Classification Project

This project uses [The 4 Universities Data Set](http://www.cs.cmu.edu/afs/cs/project/theo-20/www/data/) to classify webpages in student, faculty, staff, department, course, project or other. 

## Page-Classifier API

The best model from this repository is currently being served on Heroku at: https://pageclassifier-api.herokuapp.com/

<p align="center">
  <img src="./../img/page-classifier.png">
</p>

## Dataset

The data was collected within the domains of four universities: Cornell, Texas, Washington and Wisconsin mainly within the engineering deparments.

More information on the dataset can be obtained at the [link](http://www.cs.cmu.edu/afs/cs/project/theo-20/www/data/) and direct access to the data can be obtained [here](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/webkb-data.gtar.gz).

## Pre-processing

* **fetch_dataset.py**: manipulate the files to keep the file tree consistent and immitates the interface used in the other native datasets provided by the scikit-learn API
* **tokenizer.py**: extracts the text features from the provided ".html" files


## Modelling

* **modelling.py**: evaluation over multiple classifiers, modified to use the CMU dataset


## Query

* **query.py**: persisted best model for later prediction given an URL



