# Web Application for Disaster Response

![Intro Pic](images/banner.jpg)

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting_started)
	1. [Overview](#overview)
	2. [Dependencies](#dependencies)
	3. [Files Descriptions](#files)
	4. [Installing](#installing)
	5. [Executing Program](#executing)
	6. [Additional Material](#material)
3. [Authors](#authors)
4. [License](#license)
5. [Acknowledgement](#acknowledgement)
6. [Screenshots](#screenshots)

<a name="descripton"></a>
## Introduction

In 2021, the Emergency Event Database (EM-DAT) recorded 432 disastrous events
related to natural hazards worldwide Overall, these accounted for 10,492 deaths,
affected 101.8 million people and caused approximately 252.1 billion US$ of
economic losses. Globally, whilst the number of deaths and the number of people 
affected were below their 20-year averages, 2021 was marked by an increase 
in the number of disaster events and extensive economic losses. 
Five of the top ten most economically costly disasters in 2021 occurred in 
the United States of America and resulted in a total economic cost of 112.5 billion US$.

![Top10](images/Top10.png)

As above statistics, disaster damage is very very high. So, to reduce damage, in this project, 
we will be building a disaster response web application that will 
classify the message into different categories like medical supplies, food, or block road 
and direct them to the right organization to provide speedy recovery as soon as possible!

<a name="getting_started"></a>
## Getting Started

<a name="overview"></a>
### Overview
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Appen.
The initial dataset contains pre-labelled tweet and messages from real-life disaster. 
The aim of the project is to build a Natural Language Processing tool that categorize messages.

The Project is divided in the following Sections:

1. Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure
2. Machine Learning Pipeline to train a model able to classify text message in categories
3. Web App to show model results in real time. 

### Files Descriptions <a name="files"></a>

The files structure is arranged as below:

	- README.md: read me file
	- ETL Pipeline Preparation.ipynb: contains ETL pipeline preparation code
	- ML Pipeline Preparation.ipynb: contains ML pipeline preparation code
	- app
		- templates
			- master.html: main page of the web application 
			- go.html: result web page
		- run.py: flask file to run the app
	- data
		- disaster_categories.csv: categories dataset
		- disaster_messages.csv: messages dataset
		- DisasterResponse.db: disaster response database
		- process_data.py: ETL process
	- models
		- train_classifier.py: classification code
		- classifier.tar.gz # saved model pkl file format zip

<a name="dependencies"></a>
### Dependencies
* Python 3.5+ (I used Python 3.7)
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly

<a name="installing"></a>
### Installing
Clone this GIT repository:
```
https://github.com/dinhquants/udacity_dsnano_disaster_response_pipeline.git
```


<a name="executing"></a>
### Executing Program:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3000/

<a name="material"></a>
### Additional Material

You can find two jupyter notebook that will help you understand how the model works step by step:
1. **ETL Preparation Notebook**: learn everything about the implemented ETL pipeline
2. **ML Pipeline Preparation Notebook**: look at the Machine Learning Pipeline developed with NLTK and Scikit-Learn

You can use **ML Pipeline Preparation Notebook** to re-train the model or tune it through a dedicated Grid Search section.
In this case, it is warmly recommended to use a Linux machine to run Grid Search, especially if you are going to try a large combination of parameters.
Using a standard desktop/laptop (4 CPUs, RAM 8Gb or above) it may take several hours to complete. 

<a name="authors"></a>
## Authors

* [Nguyen Duc An Dinh (dinhnda@fsoft.com.vn)](https://github.com/dinhquants)

<a name="license"></a>
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a name="acknowledgement"></a>
## Acknowledgements

* [Udacity](https://www.udacity.com/) for providing such a complete Data Science Nanodegree Program
* [Appen](https://www.figure-eight.com/) for providing messages dataset to train my model

<a name="screenshots"></a>
## Screenshots

1. This is an example of a message you can type to test Machine Learning model performance  
`We have a lot of problem at Tokyo, those people need water and food.`

2. After clicking **Classify Message**, you can see the categories which the message belongs to highlighted in green

![Sample Output](images/image2.png)

3. The main page shows some graphs about training dataset, provided by Appen

![Main Page](images/image1.png)
