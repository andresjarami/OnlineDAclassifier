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

    currentClass = currentClass + 1
    for i in range(np.size(testLabels)):
        currentPredictor = DA_Classifiers.predictedModelQDA(testFeatures[i, :], model, classes)

        if currentPredictor == testLabels[i]:
            if currentPredictor == currentClass:
                TP += 1
            else:
                TN += 1
        else:
            if testLabels[i] == currentClass:
                FN += 1
            else:
                FP += 1

    return mcc(TP, TN, FP, FN)


def reduce_dataset(x, y, classes):
    reduce_X = []
    reduce_Y = []
    for cl in range(1, classes + 1):
        reduce_X.append(np.mean(x[y == cl], axis=0))
        reduce_Y.append(cl)
    return np.array(reduce_X), np.array(reduce_Y)


def OurModelUnsupervisedAllProb_OneGesture(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels,
                                oneShotModel, typeModel, shotStart):
    t = time.time()

    trainFeatures, trainLabels = reduce_dataset(trainFeatures, trainLabels, classes)

    # peopleClass = len(preTrainedDataMatrix.index)


    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class'])

    # for cla in range(classes):
    #     adaptiveModel.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
    #     adaptiveModel.at[cla, 'mean'] = np.zeros((1, allFeatures))[0]

    wFewCov = np.ones(classes) * shotStart
    wFewMean = np.ones(classes) * shotStart


    wGestureMean = np.zeros(classes)
    wGestureCov = np.zeros(classes)

    for cla in range(classes):



        if preTrainedDataMatrix['prob'].loc[0][cla] != 0:
            personMean = preTrainedDataMatrix['mean'].loc[0]
            personCov = preTrainedDataMatrix['cov'].loc[0]
            wGestureMean[cla] = weightPerPersonMean(
                oneShotModel, personMean, cla, classes, trainFeatures, trainLabels) * \
                                       preTrainedDataMatrix['prob'].loc[0][cla]
            if typeModel == 'QDA':
                wGestureCov[cla] = weightPerPersonCov(
                    oneShotModel, personCov, cla, classes, trainFeatures, trainLabels) * \
                                          preTrainedDataMatrix['prob'].loc[0][cla]

        wCurrentCov[cla] = currentValues['weight_cov'].loc[cla]
        wCurrentMean[cla] = currentValues['weight_mean'].loc[cla]
        adaptiveModel['weight_cov'].loc[cla] = currentValues['weight_cov'].loc[cla] + wPeopleCov[:, cla].sum()
        adaptiveModel['weight_mean'].loc[cla] = currentValues['weight_mean'].loc[cla] + wPeopleMean[:, cla].sum()

    sumWMean = np.sum(wPeopleMean, axis=0) + wFewMean
    sumWCov = np.sum(wPeopleCov, axis=0) + wFewCov
    wFewMean /= sumWMean
    wFewCov /= sumWCov
    wPeopleMean /= sumWMean
    wPeopleCov /= sumWCov
    wFewMean = np.nan_to_num(wFewMean, nan=1)
    wFewCov = np.nan_to_num(wFewCov, nan=1)
    wPeopleMean = np.nan_to_num(wPeopleMean)
    wPeopleCov = np.nan_to_num(wPeopleCov)

    means = np.resize(preTrainedDataMatrix['mean'], (classes, len(preTrainedDataMatrix['mean']))).T * wPeopleMean
    covs = np.resize(preTrainedDataMatrix['cov'], (classes, len(preTrainedDataMatrix['cov']))).T * wPeopleCov
    for cla in range(classes):
        adaptiveModel.at[cla, 'class'] = cla + 1
        adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + oneShotModel['mean'].loc[cla] * wFewMean[cla]
        if typeModel == 'LDA':
            adaptiveModel.at[cla, 'cov'] = (np.sum(preTrainedDataMatrix['cov']) + oneShotModel['cov'].loc[cla]) / (
                    peopleClass + 1)
        elif typeModel == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + oneShotModel['cov'].loc[cla] * wFewCov[cla]

    return adaptiveModel, time.time() - t
