# DS4400 Final Project
_"Next came Dionysus, the son of the virgin, bringing the counterpart to bread: 
wine and the blessings of life's flowing juices. His blood, the blood of the grape, 
lightens the burden of our mortal misery."_  — Euripides

## Who?
This project was completed by Jay Sherman and Michael Wheeler for Northeastern's 
_DS4400: Introduction to Machine Learning 1_, taught by Prof. Ehsan Elhamifar in the Spring of 2020.

## What?
We use basic machine learning methods to determine if the color and quality of a wine can be 
predicted from its physical properties. We examine a dataset including red/white classification and 
a rating of quality on a 10-point scale. We use linear regression models and basis function expansions 
to predict the rating of each wine. We also use logistic regression and support vector 
machines to classify a wine by color.

## How?
To run the code:
  - Clone the repository from GitHub: `git clone https://github.com/mikewheel/ds4400-final-project`
  - Create and activate a Python 3.8 virtual environment: `python3.8 -m venv virtualenv && source virtualenv/bin
  /activate`
  - Install the project requirements: `pip install -r requirements.txt`
  - Execute training: `python3 -m main`
