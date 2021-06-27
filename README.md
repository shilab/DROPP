# DROPP: Dimensionality Reduction OrientedPhenotype Prediction Using Genetic Algorithms

This repository holds the source code for "Mowlaei, M. E., Shi, X.: DROPP: Dimensionality Reduction Operated Phenotype Prediction Framework Using Genetic Algorithms" manuscript, accepted at ICIBM2021. DROPP is a bi-objective framwork, designed to perform phenotype prediction and QTL detection. The main code resides in the Main.py file. The results reported in the paper are stored in experimentResults.zip as comma seperated (CSV) files. Overlap_Cheker.ipynb contains codes used for detecting overlap of QTLs between our results and previous research, as well as LD concordance calculations.

## DROPP framework:
Dimensionality ReductionOperated Phenotype Prediction (DROPP)Phenotype prediction involves solving two problems, namely epistatic interactionsamong loci and curse of dimensionality. To address the latter, we propose DROPP in order to reduce the search space for effective SNPs in phenotype prediction. While our method does not directly address the epistasis detection problem, it can be used as a prior step in order to solve the aforementioned problem. Our motivation andmethod will be described in the following. The overall pipeline used in this study isillustrated below:

<p align="center">
  <img width="460" height="300" src="https://github.com/shilab/DROPP/blob/3fdca0252e373ffe5194a2aa9bb0a07762a3f80e/assets/Figure%203.png">
</p>


As a side note, henceforth, we will be using loci and featuresinterchangeably.The  dataset  is  first  partitioned  into  training  and  test  sets.  Afterwards,  pre-processing  is  applied  to  both  sets,  restricting  features  that  will  be  used  for  theGA in the next step. In the GA, we aim to find the optimal set of features thatmaximize our criteria, namelyR2Adj, on the training set.Aregressionmodelisthenfitonthetestsetusingtheselectedsubsetoffeaturesandweevaluatetheperfor-manceusingthismodel.SelectedfeaturescandependontheregressionmodelusedinthefitnessfunctionoftheGA;however,thefinaloutputoftheGAisthesetofthefeatures. Since GA is a stochastic algorithm, we run it more than once andconsider the overlap of produced sets as the final output.
## Getting Started:

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites
```
Python3.8 
virtualenv >= 16.4.3
Jupyter notebook 6.1.5
```
### Setup

Create virtual environment

```
git clone https://github.com/shilab/DROPP.git
cd DROPP
mkdir venv
python3 -m venv venv/
source venv/bin/activate
```
Install requirment dependents
```
pip3 install scipy sklearn pandas jupyter matplotlib seaborn scikit-allel
```
