import warnings

from scipy.stats import pearsonr, norm, rankdata

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

np.random.seed(seed=28213)

import csv
from pathlib2 import Path
import argparse
import time

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge
from allel import rogers_huff_r_between

import GeneSelector as gs
import pandas as pd
from mUtils import Queue, regression_accuracy_scorer, pearson_correlation_coef,bcolors

def get_pearson_cc(x, y):
    return abs(pearsonr(x, y)[0])

def get_feature_to_target_correlation(x, y):
    scores = np.apply_along_axis(get_pearson_cc, 0, x, y)
    assert(len(scores) == x.shape[1])
    return scores


def feature_ranking(score):
    """
    Rank features in descending order according to their score, the larger the score, the more important the
    feature is
    """
    idx = np.argsort(score, 0)
    return idx[::-1]

def removeRedundantGenes(x, y, selectedIndices, n_folds=9):
    clf = BayesianRidge(n_iter=1000)
    removedCount = 0
    lastAcc = np.mean(cross_val_score(clf, x[:, selectedIndices], y,
                                     # fit_params={'sample_weight': sample_weights},
                                     scoring=regression_accuracy_scorer, cv=n_folds, n_jobs=n_folds))
    SICopy = np.copy(selectedIndices)
    for locus in selectedIndices:
        SICopyCopy = set(np.copy(SICopy))
        SICopyCopy.discard(locus)
        SICopyCopy = list(sorted(SICopyCopy))
        scores = cross_val_score(clf, x[:, SICopyCopy], y,
                                     # fit_params={'sample_weight': sample_weights},
                                     scoring=regression_accuracy_scorer, cv=n_folds, n_jobs=n_folds)
        newAcc = np.mean(scores) - np.std(scores)
        if newAcc >= lastAcc:
            lastAcc = newAcc
            SICopy = np.copy(SICopyCopy)
            removedCount+=1

    print(f"removed {removedCount} loci")
    return list(SICopy)

def detect_outliers(df):
    outlier_indices = []

    Q1 = np.percentile(df, 25)
    Q3 = np.percentile(df, 75)
    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR

    outlier_indices = df[(df < Q1 - outlier_step) |
                         (df > Q3 + outlier_step)].index

    return outlier_indices


def preprocess(scores, x, max_top_features = 300, LD_threshold = 0.7):
    assert(scores.shape[0] == x.shape[1])

    sorted_features_based_on_rank = feature_ranking(scores)
    list_of_selected_features = []

    for tf in sorted_features_based_on_rank:
        if len(list_of_selected_features) < max_top_features:
            shall_add = True
            for sf in list_of_selected_features:
                if rogers_huff_r_between(x[:, tf].reshape(1, x.shape[0]), x[:, sf].reshape(1, x.shape[0])) ** 2 >= LD_threshold:
                    shall_add = False
                    break
            if shall_add:
                list_of_selected_features.append(tf)
        else:
            break
    return np.array(list_of_selected_features)


def doIt(phenoIndex=0, genotype_file='genotype_full.txt', phenotype_file = 'phenotype.csv', n_process=8,
         totalRounds=10, maxSeconds=60 * 60 * 24, feature_bound=7, LD_threshold=0.8, exp=0):
    # Main parameters of the algorithm

    #These are not used actually, but leave the first two as they are
    timedBetaReductionStrategy = False
    roundBasedBetaReductionStrategy = False
    smallQueueSize = 100 #If there's no improvement for this many steps on validation metric, we reduce beta
    maxSecondsMiniStroke = 60 * 60 if timedBetaReductionStrategy else None #Max time to wait if there's no improvement in validation metric


    #These params are used
    runCount = totalRounds #number of times to run the algorithm
    poolSize = 10 #GA param, the number of chromosomes in the population
    maxAge = None #GA param, setting it to positive integers enables simulated annealing inside the GA
    maxIterations = 5000
    maxIterations *= poolSize
    max_top_features = 10000

    finalStats = []

    # Read the data and pre-process it

    genotypes = pd.read_csv(genotype_file, sep='\t', index_col=0)
    genotypes[genotypes == -1] = 0
    multi_pheno = pd.read_csv(phenotype_file, sep=',', index_col=0)
    #use only training data to generate this

    headers = genotypes.columns[:]
    phenoName = multi_pheno.columns[phenoIndex]


    y = multi_pheno.iloc[:, phenoIndex]

    # move the gene loci with NA traits
    genotypes = genotypes[~y.isna()]
    y = y[~y.isna()]

    # normlization
    scaled_Y = (y - y.min()) / (y.max() - y.min())
    # temp_Y = scaled_Y[~scaled_Y.isna()]
    # outliers_index = detect_outliers(temp_Y)
    # # set outliers as NAN
    # scaled_Y[outliers_index] = np.nan
    #
    x = genotypes.to_numpy()
    # y = scaled_Y[~np.isnan(scaled_Y)].to_numpy()
    y = scaled_Y.to_numpy()
    # y = norm.ppf((rankdata(y) - 0.5) / len(y))


    print("Data Shape after removing outliers: {}".format(x.shape))


    # Split train and test
    train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=0.1,
                                                            random_state=28213,
                                                            shuffle=True,
                                                            # stratify=digitized_for_stratification
                                                            )



    pcc_selected = []
    # pcc_selected = np.load(f"pcc_selected_i{phenoIndex}_{LD_threshold}.npy")
    pcc_scores = get_feature_to_target_correlation(train_X, train_Y)
    pcc_scores = (pcc_scores - np.min(pcc_scores)) / (np.max(pcc_scores) - np.min(pcc_scores))
    print("PearsonCC Finished")
    pcc_selected = preprocess(pcc_scores, train_X, max_top_features, LD_threshold)
    print(f"Feature No. after preprocess: {pcc_selected.shape}")
    np.save(f"pcc_selected_i{phenoIndex}_{LD_threshold}.npy", pcc_selected)



    selected_features_union = set()

    selected_features_union.update(pcc_selected.tolist())



    selected_features_union = sorted(selected_features_union)
    print(f"Number of selected features:\t{len(list(selected_features_union))}")


    feature_votes = np.zeros((train_X.shape[1]))
    feature_votes[selected_features_union] = 1


    clf = BayesianRidge(n_iter=1000)
    clf.fit(train_X[:, selected_features_union], train_Y)
    tMSE = mean_squared_error(test_Y, clf.predict(test_X[:, selected_features_union]))
    tAcc = pearson_correlation_coef(clf, test_X[:, selected_features_union], test_Y)

    print("{} Test Acc: {:3.4f}\t MSE: {:1.6f}".format(clf.__class__.__name__, tAcc, tMSE))

    clf = Ridge()
    clf.fit(train_X[:, selected_features_union], train_Y)
    tMSE = mean_squared_error(test_Y, clf.predict(test_X[:, selected_features_union]))
    tAcc = pearson_correlation_coef(clf, test_X[:, selected_features_union], test_Y)

    print("{}R Test Acc: {:3.4f}\t MSE: {:1.6f}".format(clf.__class__.__name__, tAcc, tMSE))



    bestResults = []
    for i in range(runCount):
        print("="*20, "Round {}".format(i), "="*20)
        print("="*47)
        startTime = time.time()
        currentSavedModelTrainingPercentage = 0
        timeLimit = True
        roundLimit = False


        print("Training Data Shape after feature ranking: {}".format(train_X.shape))
        print(f"Feature Bound: {feature_bound}")

        bestGenes = None
        bestStats = None


        filePrefixName = "{}_{}_"\
            .format(phenoName, i)
        print(filePrefixName)

        gene_selector = gs.GeneSelector(n_process=n_process, feature_bound=feature_bound)

        #Time out rules: 0 -> no timeout occured. 1 -> main timeout, 2 -> mini timeout
        for timedOut, trainWordScores, percentage, NOG, totalSeconds, savedPercentage in\
                gene_selector.just_do_it(train_X, train_Y, feature_votes,
                                         maxAge=maxAge, poolSize=poolSize,
                                         maxSeconds=maxSeconds, maxIdleRounds=maxIterations,
                                         maxSecondsMiniStroke=maxSecondsMiniStroke):

            currentSavedModelTrainingPercentage = savedPercentage
            clf2 = BayesianRidge(n_iter=10)
            clf2.fit(train_X[:, np.where(trainWordScores == 1)[0]], train_Y)
            tMSE2 = mean_squared_error(test_Y, clf2.predict(test_X[:, np.where(trainWordScores == 1)[0]]))
            tAcc2 = pearson_correlation_coef(clf2, test_X[:, np.where(trainWordScores == 1)[0]], test_Y)
            # clf = DecisionTreeRegressor(criterion="friedman_mse", max_depth=NOG*2)
            mRuns = 1
            acc_results = np.zeros((mRuns))
            mse_results = np.zeros((mRuns))

            for i in range(mRuns):
                # clf = ExtraTreesRegressor(random_state=0, n_estimators=max(1, int(np.ceil(NOG / feature_bound))),
                #                           max_depth=feature_bound, n_jobs=max(1, int(np.ceil(NOG / feature_bound))))
                # clf = SVR(kernel="poly", degree=3)
                # clf = ARDRegression(n_iter=500)
                clf = Ridge()
                clf.fit(train_X[:, np.where(trainWordScores == 1)[0]], train_Y)
                mse_results[i] = mean_squared_error(test_Y, clf.predict(test_X[:, np.where(trainWordScores == 1)[0]]))
                acc_results[i] = pearson_correlation_coef(clf, test_X[:, np.where(trainWordScores == 1)[0]], test_Y)


            # print("BR ACC Valid Score: {:3.2f}".format(vAcc), "Test Score: {:3.2f}".format(tAcc))

            print("Test Score: {}{}: ({:3.4f}, {:1.6f}), {}BR: ({:3.4f}, {:1.6f}{})".
                  format(bcolors.OKCYAN, clf.__class__.__name__, np.max(acc_results), mse_results[np.argmax(acc_results)],
                         bcolors.OKGREEN, tAcc2, tMSE2, bcolors.ENDC))


            if not roundBasedBetaReductionStrategy and timedOut == 3:
                roundLimit = not roundLimit
                timeLimit = False
                break

            with open(f"./outG{exp}_{LD_threshold}/" + filePrefixName +  f"_LD_{LD_threshold}_log_"
                      + '.csv', 'a', encoding='utf-8') as csvfile:
                spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow([NOG, percentage, tAcc2, tMSE2, totalSeconds.total_seconds()])

            bestGenes = trainWordScores
            bestStats = [NOG, percentage, tAcc2, tMSE2, totalSeconds.total_seconds()]

        
        if timeLimit:
            print("Time limit reached!")
        if roundLimit:
            print("Round limit reached!")
        print('\n=======================savedPercentage:{:0.2f}========================'
              '======='.format(currentSavedModelTrainingPercentage * 100))
        print("Best stats: [NOG, TrainAcc, tAcc, tMSE, totalSeconds.total_seconds()]", bestStats)
        print("Total Time:", str(time.time() - startTime))

        with open(f"./outG{exp}_{LD_threshold}/" +
                  "{}_LD_{}_RR{}__8fold_seed28213".format(phenoName, LD_threshold, i)
                  + '_feature_bound_{}_'.format(feature_bound) +
                  '_NOF_' + str(bestStats[0]) + '_MSE_' + str(bestStats[1]) +
                  '_TAcc{:1.5f}'.format(bestStats[3]) +
                  '_TMSE_{:1.5f}'.format(bestStats[4]) +
                  '__time_' + str(time.time() - startTime)
                  + '.csv', 'w', encoding='utf-8') as csvfile:
            spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(headers)
            spamwriter.writerow(bestGenes)
        bestResults.append(np.where(np.array(bestGenes)!=0)[0])
        finalStats.append([i, "pearsonCC", 0, 0, bestStats[0],
                           bestStats[1], bestStats[2], bestStats[3], time.time() - startTime])

    bestResultsUnionBeforeRemoving = list(sorted(set().union(*bestResults)))
    bestResultsIntersectionBeforeRemoving = list(sorted(set(bestResults[0]).intersection(*bestResults[1:])))

    bestResultsUnion = bestResultsUnionBeforeRemoving[:]#removeRedundantGenes(train_X, train_Y, bestResultsUnionBeforeRemoving, n_folds=n_process)

    print(f"Union Feature Count: before:{len(bestResultsUnionBeforeRemoving)} after:{len(bestResultsUnion)}")


    clf = BayesianRidge(n_iter=1000)
    clf.fit(train_X[:, bestResultsUnionBeforeRemoving], train_Y)
    tMSEb = mean_squared_error(test_Y, clf.predict(test_X[:, bestResultsUnionBeforeRemoving]))

    print("UNION BEFORE Test Score: {:1.6f}".format(tMSEb))

    clf = BayesianRidge(n_iter=1000)
    clf.fit(train_X[:, bestResultsUnion], train_Y)
    tMSE = mean_squared_error(test_Y, clf.predict(test_X[:, bestResultsUnion]))

    print("UNION AFTER Valid Test Score: {:1.6f}".format(tMSE))

    unionFeaturesToWrite = np.zeros((len(headers)), dtype='int16')
    unionFeaturesToWrite[list(sorted(bestResultsUnionBeforeRemoving))] = 1
    with open(f"./outG{exp}_{LD_threshold}/" +
              "{}_LD_{}_UNION__8fold_seed28213".format(phenoName, LD_threshold)
              +'_feature_bound_' + str(feature_bound)
              +'_NOF_beforeR_' + str(len(bestResultsUnionBeforeRemoving))
              +'_TMSEb_{:1.5f}'.format(tMSEb)
              + '_NOF_afterR_' + str(len(bestResultsUnion))
              +'_TMSEa_{:1.5f}'.format(tMSE)
              + '.csv', 'w', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(headers)
        spamwriter.writerow(unionFeaturesToWrite)

    if len(bestResultsIntersectionBeforeRemoving) > 0:

        bestResultsIntersection = bestResultsIntersectionBeforeRemoving[:]
        # removeRedundantGenes(train_X, train_Y, bestResultsIntersectionBeforeRemoving,
        #                                                n_folds=n_process)
        print(f"Intersection Feature Count: before: {len(bestResultsIntersectionBeforeRemoving)} after: {len(bestResultsIntersection)}")

        clf = BayesianRidge(n_iter=1000)
        clf.fit(train_X[:, bestResultsIntersectionBeforeRemoving], train_Y)
        tMSEb = mean_squared_error(test_Y, clf.predict(test_X[:, bestResultsIntersectionBeforeRemoving]))

        print("Intersection before Test Score: {:1.6f}".format(tMSEb))

        clf = BayesianRidge(n_iter=1000)
        clf.fit(train_X[:, bestResultsIntersection], train_Y)
        tMSE = mean_squared_error(test_Y, clf.predict(test_X[:, bestResultsIntersection]))

        print("Intersection after Test Score: {:1.6f}".format(tMSE))

        intersectFeaturesToWrite = np.zeros((len(headers)), dtype='int16')
        intersectFeaturesToWrite[list(sorted(bestResultsIntersectionBeforeRemoving))] = 1
        with open(f"./outG{exp}_{LD_threshold}/" +
                  "{}_LD_{}_Intersection__8fold_seed28213".format(phenoName, LD_threshold)
                  + '_NOF_beforeR_' + str(len(bestResultsIntersectionBeforeRemoving))
                  + '_feature_bound_' + str(feature_bound)
                  + '_TMSEb_{:1.5f}'.format(tMSEb)
                  + '_NOF_afterR_' + str(len(bestResultsIntersection))
                  + '_TMSEa_{:1.5f}'.format(tMSE)
                  + '.csv', 'w', encoding='utf-8') as csvfile:
            spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(headers)
            spamwriter.writerow(intersectFeaturesToWrite)




    with open(f"./outG{exp}_{LD_threshold}/" + '_' + phenoName + f"_LD_{LD_threshold}_log_"
              + '.csv', 'a', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        for r in finalStats:
            spamwriter.writerow(r)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ShiLab\'s GA model for gene selection')
    # parser.add_argument('-di', type=str, help='Path to the folder containing input dataset')
    parser.add_argument('-pi', type=int, help='Phenotype Index to run the code on (default=2)', default=0)
    parser.add_argument('-np', type=int, help='Number of processes (default=5)', default=5)
    # parser.add_argument('-fname', type=str, help='Filter Name FISHER | IG | PearsonCC (default=PearsonCC)',
    #                     choices=['IG', 'FISHER', 'PearsonCC'], default='PearsonCC')
    args = parser.parse_args()

    phenoIndex = args.pi# if args.pi else 2
    n_process = args.np# if args.np else 8
    # IGorF = args.fname# if args.fname else "FISHER"

    maxSeconds = 60 * 60 * 24  # Max GA runtime
    totalRounds = 3
    feature_bound = 7

    genotype_file = '../data/genotype_full.txt'
    phenotype_file = '../data/phenotype.csv'

    for exp in range(1):
        for LD_threshold in [0.2, 0.3, 0.4, 0.5]:
            for phenoIndex in [0, 1, 2, 3, 4]:
                phenos = pd.read_csv(phenotype_file, sep=',', index_col=0, nrows=2)
                phenoName = phenos.columns[phenoIndex]

                Path(f"outG{exp}_{LD_threshold}").mkdir(parents=True, exist_ok=True)
                with open(f"./outG{exp}_{LD_threshold}/" + '_' + phenoName + f"_LD_{LD_threshold}_log_" + '_featureBound_' + str(feature_bound) + "_"
                          + '.csv', 'a', encoding='utf-8') as csvfile:
                    spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow(['Round', "FilterName", "NOF",
                                         "Acc", "TACC", "TMSE", "TotalSecs"])

                doIt(phenoIndex=phenoIndex, genotype_file=genotype_file, phenotype_file=phenotype_file, n_process=n_process, LD_threshold=LD_threshold,
                     totalRounds=totalRounds, maxSeconds=maxSeconds, feature_bound=feature_bound, exp=exp)
