# DROPP: Dimensionality Reduction OrientedPhenotype Prediction Using Genetic Algorithms

This repository holds the source code for "Mowlaei, M. E., Shi, X.: DROPP: Dimensionality Reduction Operated Phenotype Prediction Framework Using Genetic Algorithms" manuscript, accepted at ICIBM2021. DROPP is a bi-objective framwork, designed to perform phenotype prediction and QTL detection. The main code resides in the Main.py file. The results reported in the paper are stored in experimentResults.zip as comma seperated (CSV) files. Overlap_Cheker.ipynb contains codes used for detecting overlap of QTLs between our results and previous research, as well as LD concordance calculations.

## DROPP framework:
Dimensionality Reduction Operated Phenotype Prediction (DROPP) Phenotype prediction involves solving two problems, namely epistatic interactionsamong loci and curse of dimensionality. To address the latter, we propose DROPP in order to reduce the search space for effective SNPs in phenotype prediction. While our method does not directly address the epistasis detection problem, it can be used as a prior step in order to solve the aforementioned problem. Our motivation and method will be described in the following. The overall pipeline used in this study is illustrated below:

<p align="center">
  <img width="460" height="auto" src="https://github.com/shilab/DROPP/blob/3fdca0252e373ffe5194a2aa9bb0a07762a3f80e/assets/Figure%203.png">
</p>


The dataset is first partitioned into training and test sets. Afterwards, pre-processing is applied to both sets, restricting features that will be  used for the GA in the next step. In the GA, we aim to find the optimal set of features that maximize our criteria, namely <img src="https://render.githubusercontent.com/render/math?math=R^{2}_adj">, on the training set. A regression model is then fit on the test set using the selected subset of features and we evaluate the performance using this model. Selected features can depend on the regression model used in the fitness function of the GA; however, the final output of the GA is the set of the features. Since GA is a stochastic algorithm, we run it more than once and consider the overlap of produced sets as the final output.

### The Optimization Problem

GA is a nature-inspired method and a major constitution of Computational Intelligence (CI), designed to solve real world problems. Our objective in the GA is to find the optimal solution for an optimization problem.
	
In this study, given a certain regression model *M* and a dataset *D*, we look for the minimal subset of SNPs in *D* that provides us the best phenotype prediction results, as our goal. The normal procedure for stopping a GA, in case the optimal goal is not met, includes (but is not limited to) setting time limit on the runtime of the algorithm or the number of iterations. Here, we employ the latter and set maximum number of iterations to 5000.

There are a limited number of metrics used for regression problems. Among them, MSE is commonly used as a measure to compare different methods. However, through empirical study we found out that $\adjr$, which has been used in literature for regression problems \cite{leach2007use}, serves as a better objective for the task at hand. This metric can be calculated as follows:
<img src="https://render.githubusercontent.com/render/math?math=R^{2}_adj = 1 - \dfrac{(1-R^2) (n - 1)}{n-p-1};">
	where \textit{p} is number of independent features selected for training the model, \textit{n} is the number of samples, and $R^2$ is calculated as below:
	\begin{equation} \label{eq:r2}	
		R^2 = 1 - \dfrac{RSS}{TSS};
	\end{equation}
	where \textit{RSS} is the sum of squares of residuals and \textit{TSS} is total sum of squares for a given trait. $\adjr$  and $R^2$ range from 0 to 1, with 1/0 being the best/worst value. We set maximizing $\adjr$ and minimizing the number of features as primary and secondary objectives in our GA, respectively. In other terms, our optimization algorithm maximizes $\adjr$ of phenotype prediction using as few features as possible.

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
