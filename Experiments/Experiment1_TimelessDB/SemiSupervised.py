import time

import numpy as np
import pandas as pd
from scipy.spatial import distance

import DA_Classifiers as DA_Classifiers


# Weight Calculation
def weightPerPersonMean(currentValues, personMean, currentClass, classes, trainFeatures, trainLabels):
    personValues = currentValues.copy()
    personValues['mean'].at[currentClass] = personMean
    return mccModelQDA(trainFeatures, trainLabels, personValues, classes, currentClass)


def weightPerPersonCov(currentValues, personCov, currentClass, classes, trainFeatures, trainLabels):
    personValues = currentValues.copy()
    personValues['cov'].at[currentClass] = personCov
    return mccModelQDA(trainFeatures, trainLabels, personValues, classes, currentClass)


# Matthews correlation coefficients

def mcc(TP, TN, FP, FN):
    mccValue = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    if np.isnan(mccValue) or mccValue < 0:
        return 0
    return mccValue


def mccModelQDA(testFeatures, testLabels, model, classes, currentClass):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    label = currentClass + 1
    for i in range(np.size(testLabels)):
        currentPredictor = DA_Classifiers.predictedModelQDAProb(testFeatures[i, :], model, classes)
        if testLabels[i] == label:
            TP += currentPredictor[currentClass]
            FP += np.sum(currentPredictor[currentPredictor != currentPredictor[currentClass]])
        else:
            TN += currentPredictor[testLabels[i] - 1]
            FN += np.sum(currentPredictor[currentPredictor != currentPredictor[testLabels[i] - 1]])

        # if currentPredictor == testLabels[i]:
        #     if currentPredictor == currentClass:
        #         TP += 1
        #     else:
        #         TN += 1
        # else:
        #     if testLabels[i] == currentClass:
        #         FN += 1
        #     else:
        #         FP += 1

    return mcc(TP, TN, FP, FN)


def reduce_dataset(x, y, classes):
    reduce_X = []
    reduce_Y = []
    for cl in range(1, classes + 1):
        reduce_X.append(np.mean(x[y == cl], axis=0))
        reduce_Y.append(cl)
    return np.array(reduce_X), np.array(reduce_Y)


def post_probabilities_Calculation_All(unlabeledGestures, model, classes, type_DA, numberGestures):
    post_probabilities = []
    if type_DA == 'LDA':
        LDACov = DA_Classifiers.LDA_Cov(model, classes)
        for i in range(numberGestures):
            post_probabilities.append(
                DA_Classifiers.predictedModelLDAProb(unlabeledGestures['mean'].loc[i].values[0], model, classes,
                                                     LDACov))
    elif type_DA == 'QDA':
        for i in range(numberGestures):
            post_probabilities.append(
                DA_Classifiers.predictedModelQDAProb(unlabeledGestures['mean'].loc[i].values[0], model, classes))
    unlabeledGestures['post_prob'] = post_probabilities
    return unlabeledGestures


def OurModelUnsupervisedAllProb_OneGesture(currentModel, unlabeledGestures, classes, trainFeatures,
                                           postProb_trainFeatures, fewModel, fewTrainFeatures, fewTrainLabels,
                                           type_DA):
    t = time.time()
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class'])

    # fews must be reduce before this method
    # trainFeatures, trainLabels = reduce_dataset(trainFeatures, trainLabels, classes)

    numberGestures = len(unlabeledGestures.index)
    if numberGestures != 0:
        unlabeledGestures = post_probabilities_Calculation_All(unlabeledGestures, currentModel, classes, type_DA,
                                                               numberGestures)

    gestureMean = np.mean(trainFeatures, axis=0)
    gestureCov = np.cov(trainFeatures, rowvar=False)

    # for cla in range(classes):
    #     adaptiveModel.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
    #     adaptiveModel.at[cla, 'mean'] = np.zeros((1, allFeatures))[0]

    wGestureMean = []
    wGestureCov = []

    for cla in range(classes):
        wGestureMean.append(weightPerPersonMean(fewModel, gestureMean, cla, classes, fewTrainFeatures, fewTrainLabels))
        if type_DA == 'QDA':
            wGestureCov.append(weightPerPersonCov(fewModel, gestureCov, cla, classes, fewTrainFeatures, fewTrainLabels))

    new_row = {'mean': gestureMean, 'cov': gestureCov, 'postProb': postProb_trainFeatures, 'wMean': wGestureMean,
               'wCov': wGestureCov}

    unlabeledGestures = unlabeledGestures.append(new_row, ignore_index=True)
    numberGestures += 1

    weightsMean = unlabeledGestures['postProb'].values * unlabeledGestures['wMean'].values
    weightsMean /= np.sum(weightsMean, axis=0)

    weightsCov = unlabeledGestures['postProb'].values * unlabeledGestures['wCov'].values
    weightsCov /= np.sum(weightsCov, axis=0)

    # sumWMean = np.sum(wPeopleMean, axis=0) + wFewMean
    # sumWCov = np.sum(wPeopleCov, axis=0) + wFewCov
    # wFewMean /= sumWMean
    # wFewCov /= sumWCov
    # wPeopleMean /= sumWMean
    # wPeopleCov /= sumWCov
    # wFewMean = np.nan_to_num(wFewMean, nan=1)
    # wFewCov = np.nan_to_num(wFewCov, nan=1)
    weightsMean = np.nan_to_num(weightsMean)
    weightsCov = np.nan_to_num(weightsCov)

    means = np.resize(unlabeledGestures['mean'], (classes, numberGestures)).T * weightsMean
    covs = np.resize(unlabeledGestures['cov'], (classes, numberGestures)).T * weightsCov
    for cla in range(classes):
        adaptiveModel.at[cla, 'class'] = cla + 1
        adaptiveModel.at[cla, 'mean'] = means[:, cla].sum()
        if type_DA == 'LDA':
            adaptiveModel.at[cla, 'cov'] = np.sum(unlabeledGestures['cov']) / numberGestures
        elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum()

    return adaptiveModel, time.time() - t, unlabeledGestures


def post_probabilities_Calculation(features, model, classes, type_DA):
    actualPredictor = np.zeros(classes)

    if type_DA == 'LDA':
        LDACov = DA_Classifiers.LDA_Cov(model, classes)
        for i in range(np.size(features, axis=0)):
            actualPredictor[DA_Classifiers.predictedModelLDA(features[i, :], model, classes, LDACov) - 1] += 1
    elif type_DA == 'QDA':
        for i in range(np.size(features, axis=0)):
            actualPredictor[DA_Classifiers.predictedModelQDA(features[i, :], model, classes) - 1] += 1

    return actualPredictor / actualPredictor.sum()
