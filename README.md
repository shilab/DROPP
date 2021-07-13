# FSF-GA: A Feature Selection Framework for Phenotype Prediction Using Genetic Algorithm

This repository holds the source code for "Mowlaei, M. E., Shi, X.: FSF-GA: A Feature Selection Framework for Phenotype Prediction Using Genetic Algorithm" manuscript, accepted at ICIBM2021. FSF-GA is a bi-objective framwork, designed to perform phenotype prediction and QTL detection. The main code resides in the Main.py file. The results reported in the paper are stored in experimentResults.zip as comma seperated (CSV) files. Overlap_Cheker.ipynb contains codes used for detecting overlap of QTLs between our results and previous research, as well as LD concordance calculations.

## FSF-GA framework:
Feature Selection Framework for Phenotype Prediction Using Genetic Algorithm (FSF-GA) Phenotype prediction involves solving two problems, namely epistatic interactionsamong loci and curse of dimensionality. To address the latter, we propose FSF-GA in order to reduce the search space for effective SNPs in phenotype prediction. While our method does not directly address the epistasis detection problem, it can be used as a prior step in order to solve the aforementioned problem. Our motivation and method will be described in the following. The overall pipeline used in this study is illustrated below:

<p align="center">
  <img width="460" height="auto" src="https://github.com/shilab/DROPP/blob/3fdca0252e373ffe5194a2aa9bb0a07762a3f80e/assets/Figure%203.png">
</p>


The dataset is first partitioned into training and test sets. Afterwards, pre-processing is applied to both sets, restricting features that will be  used for the GA in the next step. In the GA, we aim to find the optimal set of features that maximize our criteria, namely <img src="https://render.githubusercontent.com/render/math?math=R_{\text{Adj}}^2">, on the training set. A regression model is then fit on the test set using the selected subset of features and we evaluate the performance using this model. Selected features can depend on the regression model used in the fitness function of the GA; however, the final output of the GA is the set of the features. Since GA is a stochastic algorithm, we run it more than once and consider the overlap of produced sets as the final output.

### The Optimization Problem

GA is a nature-inspired method and a major constitution of Computational Intelligence (CI), designed to solve real world problems. Our objective in the GA is to find the optimal solution for an optimization problem.
	
In this study, given a certain regression model *M* and a dataset *D*, we look for the minimal subset of SNPs in *D* that provides us the best phenotype prediction results, as our goal. The normal procedure for stopping a GA, in case the optimal goal is not met, includes (but is not limited to) setting time limit on the runtime of the algorithm or the number of iterations. Here, we employ the latter and set maximum number of iterations to 5000.

There are a limited number of metrics used for regression problems. Among them, MSE is commonly used as a measure to compare different methods. However, through empirical study we found out that <img src="https://render.githubusercontent.com/render/math?math=R_{\text{Adj}}^2">, serves as a better objective for the task at hand. This metric can be calculated as follows:

<img src="https://render.githubusercontent.com/render/math?math=R_{\text{Adj}}^2 = 1 - \dfrac{(1-R^2) (n - 1)}{n-p-1};">

where *p* is number of independent features selected for training the model, *n* is the number of samples, and <img src="https://render.githubusercontent.com/render/math?math=R^{2}"> is calculated as below:

<img src="https://render.githubusercontent.com/render/math?math=R^{2} = 1 - \dfrac{RSS}{TSS}">


where *RSS* is the sum of squares of residuals and *TSS* is total sum of squares for a given trait. <img src="https://render.githubusercontent.com/render/math?math=R_{\text{Adj}}^2">  and <img src="https://render.githubusercontent.com/render/math?math=R^{2}"> range from 0 to 1, with 1/0 being the best/worst value. We set maximizing <img src="https://render.githubusercontent.com/render/math?math=R_{\text{Adj}}^2"> and minimizing the number of features as primary and secondary objectives in our GA, respectively. In other terms, our optimization algorithm maximizes <img src="https://render.githubusercontent.com/render/math?math=R_{\text{Adj}}^2"> of phenotype prediction using as few features as possible.

### Pre-processing

The purpose of GA in our approach is to find the minimum set of the features, for each trait, that delivers the best prediction power. However, evolutionary algorithms alone cannot prioritize suitable features, resulting in extremely long run-times until convergence. In order to guide our GA, we first mark valid SNPs for each trait and our GA is only allowed to use them in order to form the output set of the features. To do so, we make use of LD between SNP pairs and Pearson Correlation Coefficients among each SNP and the target trait. LD between SNP pairs is calculated using Python *scikit-allel* package. The code for pre-processing can be found at <cite><a href="https://github.com/shilab/DROPP/blob/096614014fe9a002be121980e21d31d5ad4bb0fd/Main.py#L82-L99">Main.py</a></cite>:


### The Genetic Algorithm

In this section, we present the details of our GA. The inputs to our GA are the training set and valid feature indices acquired by the pre-processing step. The output is the set of selected SNPs that give the optimal result for predicting the phenotype of interest. Since randomness in evolutionary algorithms is inevitable, specially in this problem, we run the GA three times and use the intersection of outputs as the final set, for each setting. 
	
The building blocks of each GA are chromosomes and three functions named *fitness* , *mutate* , and *crossover*. The overall process of the proposed GA is illustrated below:

<p align="center">
  <img width="460" height="auto" src="https://github.com/shilab/DROPP/blob/a194c2d303e7da1b6b8c247eb194b7bd689543d3/assets/Figure%204.png">
</p>

In our algorithm, each chromosome contains a vector of binary values (1/0) called *genes*. In other terms, genes is referred to the parameters of the solutions in our problem, this should not be confused with the concept of genes in genetics, and through this paper, we use gene(s) only in context of GA. The length of each array is equal to the number of loci in genotypes. Setting each element in these arrays to 1/0 indicates that the corresponding feature should be used/discarded in the respective data subset. In other terms, these arrays mask the presence of features in the dataset, as demonstrated below:

<p align="center">
  <img width="510" height="auto" src="https://github.com/shilab/DROPP/blob/515c9f7bcfef96becb3d905ffc8609877c96f507/assets/Figure%205.png">
</p>

The *fitness* function in the proposed GA simply calculates *fitness score* on the training set, that is, <img src="https://render.githubusercontent.com/render/math?math=R_{\text{Adj}}^2"> in this study, using Bayesian Ridge regressor implementation from *Scikit-learn* package. The key to selecting the model for the *fitness* function is that it should not have inherent *L1* penalty (e.g. Lasso), so that redundant features affect model performance are removed in the process. Tabu Search (TS) is incorporated into our GA in order to improve local search and prohibiting it from re-checking previously-visited solutions. Furthermore, TS can save time by preventing redundant calculations in the *fitness* function.

The *mutate* function takes a chromosome and modifies its genes, exploring the search space for the global optimum. <cite><a href="https://github.com/shilab/DROPP/blob/a194c2d303e7da1b6b8c247eb194b7bd689543d3/GeneSelector.py#L63-L91">Mutate function</a></cite> contains the code for the *mutate* function. The inputs of *crossover* function are two chromosomes, named parents <img src="https://render.githubusercontent.com/render/math?math=(G_P, G_D)">, their respective fitness scores, and *fitness* function. Generally speaking, in GA, crossover operation combines two sets of genes, resulting in a new chromosome, named child <img src="https://render.githubusercontent.com/render/math?math=G_C">, in which genes are inherited from either of parents --performing exploitation and leading to convergence in search subspace. The same is applied in our *crossover* function. The source code of *crossover* function is can be found at <cite><a href="https://github.com/shilab/DROPP/blob/a194c2d303e7da1b6b8c247eb194b7bd689543d3/GeneSelector.py#L94-L117">Crossover function</a></cite>.

The base code of our GA is adopted from <cite><a href="https://github.com/handcraftsman/GeneticAlgorithmsWithPython">Genetic Algorithms with Python</a></cite>. However, as mentioned above, the code was heavily modified. The size of parent pool in our GA is set to 10. The rate of mutation and crossover is set dynamically according to the last 3 improvement. However, we have designed the algorithm so that mutation/crossover rate cannot fall under 20\%, and in each turn, only one of these operations is performed on each chromosome. For instance, if 2 out of 3 last improvements are resulted from crossover, then the next function to apply on the next chromosomes, until the next improvement is found, are crossover/mutation with a probability of 60/40\%.
	
### Computational Complexity
The bottleneck in proposed GA is the *fitness* function, the most costly operation in this function is training	Bayesian Ridge model on the data. Considering we have *n* samples and *p* features, the training takes <img src="https://render.githubusercontent.com/render/math?math=O(np^2+p^3)"> operations. After pre-processing step <img src="https://render.githubusercontent.com/render/math?math=n \gg p">; hence, we can safely assume the cost of Bayesian Ridge to be <img src="https://render.githubusercontent.com/render/math?math=O(p^2n)">. Mutation and crossover take <img src="https://render.githubusercontent.com/render/math?math=O(Lnp^2)"> since their inner loops involve calls to *fitness* function up to *L* times. The GA runs for *K* iterations and parent pool holds *S* chromosomes in total, meaning that the total computational complexity of our GA, in this study, equals to <img src="https://render.githubusercontent.com/render/math?math=O(LKSnp^2)">.

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

### Running the code

You can simply run the code by opening a terminal in working directory and entering the following command:

```
.../<DROPP-main>$python3 Main.py -pi [Phenotype Index] -ldt [LD cutoff Threshold]
```

The dataset should be partitioned into genotype and phenotype parts, and the path of these files should be set at <cite><a href="https://github.com/shilab/DROPP/blob/75df9384794f0c56c0b84ea3c03cbf4eb6866fb1/Main.py#L397-L398">Main.py</a></cite>.
