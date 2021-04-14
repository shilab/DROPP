# DROPP: Dimensionality Reduction OrientedPhenotype Prediction Using Genetic Algorithms

This repository holds the source code for "Mowlaei, M. E., Shi, X.: DROPP: Dimensionality Reduction OrientedPhenotype Prediction Using Genetic Algorithms" manuscript, submitted to ICIBM2021. DROPP is a bi-objective framwork, designed to perform phenotype prediction and QTL detection. The main code resides in the Main.py file. The results reported in the paper are stored in experimentResults.zip as comma seperated (CSV) files. Overlap_Cheker.ipynb contains codes used for detecting overlap of QTLs between our results and previous research, as well as LD concordance calculations.

# Getting Started:

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

## Prerequisites
'''
Python3.8 
virtualenv >= 16.4.3
Jupyter notebook 6.1.5
'''
## Setup

Create virtual environment

'''
git clone https://github.com/shilab/DROPP.git
cd DROPP
mkdir venv
python3 -m venv venv/
source venv/bin/activate
'''
Install requirment dependents
'''
pip3 install scipy sklearn pandas jupyter matplotlib scikit-allel
'''
