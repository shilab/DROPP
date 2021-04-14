import datetime
import random

import copy

from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge
import GeneticsForSelection as gns
import sys
import numpy as np
from mUtils import regression_accuracy_scorer, LimitedSizeDict


class GeneSelector:

    def __init__(self, n_process=8, feature_bound=5):
        self.Memo = LimitedSizeDict(size_limit=1000)
        self.FitnessCites = 1
        self.SavedCalculationTimes = 1
        self.Genetics = None
        self.n_process = n_process
        self.featureBound = feature_bound

    def reset_mini_timer(self, recalculateFitnesses=True):
        if not self.Genetics is None:
            self.Genetics.reset_mini_timer(recalculateFitnesses)

    def __get_fitness_instance(self, metric, bic, std, nog):
        return Fitness(metric, bic, std, nog)

    def __get_fitness(self, genes, x, y, initialFitness=None, runs=1):
        nog = np.count_nonzero(genes)
        if nog == 0:
            return self.__get_fitness_instance(0, 1000, 1, len(genes))

        hashedGenes = hash(genes.data.tobytes())
        self.FitnessCites += 1
        if initialFitness is not None and hashedGenes in self.Memo: #Tabu mem search
            self.SavedCalculationTimes += 1
            return self.__get_fitness_instance(initialFitness.Metric - 10, 1000, initialFitness.Std + 0.1,
                                               initialFitness.NOG)

        clf = BayesianRidge(n_iter=100)
        clf.fit(x[:, np.where(genes == 1)[0]], y)
        scores2, bic = regression_accuracy_scorer(clf, x[:, np.where(genes == 1)[0]], y)
        fitness = self.__get_fitness_instance(scores2, bic, 0, nog)
        self.Memo[hashedGenes] = fitness
        return fitness

    def __removeRedundantGenes(self, fnGetFitness, genes, fitness):
        lastFitness = copy.deepcopy(fitness)
        onedLoci = np.where(genes == 1)[0]
        genesCopy = np.copy(genes)
        for locus in onedLoci:
            gensCopyCopy = np.copy(genesCopy)
            gensCopyCopy[locus] = 0
            newFitness = fnGetFitness(gensCopyCopy, initialFitness=fitness)
            if newFitness > lastFitness:
                lastFitness = newFitness
                genesCopy[locus] = 0

        return genesCopy, lastFitness

    def __mutate(self, genes, initialFitness, fnGetFitness, x, votes):
        bound = 2
        maxTries = random.randint(1, 5)
        genesCopy = np.copy(genes)
        fitness = copy.deepcopy(initialFitness)
        for _ in range(maxTries):
            toAdd = random.randint(1, 2) == 1

            if toAdd:
                zeroedLoci = np.where((genesCopy == 0) & (votes != 0))[0]
                if len(zeroedLoci) > 0:
                    p_index = np.random.choice(zeroedLoci, random.randint(1, min(bound, len(zeroedLoci)))
                        if len(zeroedLoci) > 1 else 1,
                        replace=False, p= None)
                    genesCopy[p_index] = 1

            else:
                onedLoci = np.where((genesCopy == 1) & (votes != 0))[0]
                zeroIndices = np.random.choice(onedLoci, random.randint(1, min(bound, len(onedLoci - 1))),
                                               replace=False, p=None)
                genesCopy[zeroIndices] = 0
                if np.count_nonzero(genesCopy) == 0:
                    genesCopy[zeroIndices] = 1
                    continue

            fitness = fnGetFitness(genesCopy, initialFitness=initialFitness)
            if fitness > initialFitness:
                break
        return genesCopy, fitness


    def __crossover(self, parentGenes, donorGenes, initialFitness, donorFitness, fnGetFitness, votes):
        # global limit
        if np.array_equal(parentGenes, donorGenes):
            return None, None
        bound = 2
        maxTries = random.randint(1, 5)
        childGenes = np.copy(parentGenes)
        fitness = copy.deepcopy(initialFitness)
        for _ in range(maxTries):
            candidateGeneIndices = np.argwhere((childGenes != donorGenes) & (votes!=0)).flatten()
            indicesToChange = np.random.choice(candidateGeneIndices,
                                               random.randint(1, min(bound, len(candidateGeneIndices)))
                                               if len(candidateGeneIndices) > 1 else 1,
                                               replace=False, p=None)

            childGenes[indicesToChange] = donorGenes[indicesToChange]
            if np.array_equal(childGenes, donorGenes) or np.count_nonzero(childGenes) == 0:
                childGenes = np.copy(parentGenes)
                continue

            fitness = fnGetFitness(childGenes, initialFitness=initialFitness)
            if initialFitness < fitness:
                break
        return childGenes, fitness




    def __create(self, gene_len, probs, poolSize, index, votes):

        valid_indices = np.where(votes != 0)[0]
        count = self.featureBound
        target_indices = np.random.choice(valid_indices, count, replace=False)
        genes = np.zeros((gene_len,), dtype='int8')
        genes[target_indices] = 1

        return genes


    def __display(self, candidate, pindex, startTime):
        timeDiff = str(datetime.datetime.now() - startTime)
        fitness = candidate.Fitness
        sys.stdout.write(
            # print(
            #     "\r" +
            "Fitness(Strategy: %s[%d]), %s\t%s\n" % (

                candidate.Strategy.name,
                pindex,
                fitness,
                timeDiff
            )
            # , end='\r'
        )
        sys.stdout.flush()





    def just_do_it(self, x, y, votes, maxAge=200, poolSize=200, maxSeconds= None, maxIdleRounds=None,
                   maxSecondsMiniStroke=None):
        random.seed(2020)
        startTime = datetime.datetime.now()
        # self.Memo = np.array([hash(str(np.ones((x.shape[1])).data.tobytes())), ])

        votesSum = np.sum(votes)
        oneProbs = np.apply_along_axis(lambda x: x / votesSum, 0, votes)


        def fnDisplay(candidate, pindex):
            self.__display(candidate, pindex, startTime)

        def fnGetFitness(genes, initialFitness=None):
            return self.__get_fitness(genes, x, y, initialFitness)

        def fnMutate(genes, fitness):
            return self.__mutate(genes, fitness, fnGetFitness, x, votes)

        def fnCreate(index):
            return self.__create(x.shape[1], oneProbs, poolSize, index, votes)

        def fnCrossover(parentGenes, donorGenes, parentFitness, donorFitness):
            return self.__crossover(parentGenes, donorGenes, parentFitness, donorFitness, fnGetFitness, votes)

        optimalFitness = self.__get_fitness_instance(1, -1000000, 0, 1)
        self.Genetics = gns.Genetics(fnGetFitness)

        for timedOut, best, percentage, NOG in self.Genetics.get_best(0, optimalFitness,
                                                  fnDisplay, custom_mutate=fnMutate, custom_create=fnCreate, maxAge=maxAge,
                                                  poolSize=poolSize, crossover=fnCrossover,
                                                  maxSeconds=maxSeconds, maxIdleRounds=maxIdleRounds,
                                                    maxSecondsMiniStroke=maxSecondsMiniStroke
                                                  # triggerLimit=limit
                                                  ):
            # bestGenes, fitness = self.__removeRedundantGenes(fnGetFitness, best.Genes, best.Fitness)
            yield timedOut, best.Genes, percentage, NOG, datetime.datetime.now() - startTime, \
                  self.SavedCalculationTimes / self.FitnessCites  # if percentage >= limit else SavedAssTimes1 / fitnessCites1


class Fitness:
    def __init__(self, metric, bic, std, nog):
        self.Metric = metric
        self.Std = std
        self.NOG = nog
        self.BIC = bic
        # self.CompoundScore = self.Metric * (1 + self.NOG * beta) + self.Std * alpha
        self.CompoundScore = self.Metric - 2*self.Std
    def __int__(self):
        return int(self.Metric)

    def __add__(self, other):
        return self.Metric + other.Metric

    def __float__(self):
        return self.Metric

    def __gt__(self, other):
        return self.CompoundScore > other.CompoundScore if self.CompoundScore != other.CompoundScore \
            else self.NOG < other.NOG
            # else  self.BIC < other.BIC if self.BIC != other.BIC\

            # else self.Std < other.Std if self.Std != other.Std\


    def __lt__(self, other):
        return self.CompoundScore < other.CompoundScore if self.CompoundScore != other.CompoundScore \
            else self.NOG > other.NOG
            # else self.BIC > other.BIC if self.BIC != other.BIC \
            # else self.Std > other.Std if self.Std != other.Std \


    def __eq__(self, other):
        return self.Metric == other.Metric and self.Std == other.Std and self.NOG == other.NOG

    def __str__(self):
        return "AdjR^2: {:e}\tBIC: {}\tSt.D.: {:e}\tNOG: {}".format(float(self.Metric), float(self.BIC), float(self.Std), self.NOG)
