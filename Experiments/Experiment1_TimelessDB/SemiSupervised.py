import time
import math
import numpy as np
import pandas as pd
from scipy.spatial import distance

import DA_Classifiers as DA_Classifiers


# Weight Calculation
def weightPerPersonMean(currentValues, personMean, currentClass, classes, trainFeatures, trainLabels, typeModel):
    personValues = currentValues.copy()
    personValues['mean'].at[currentClass] = personMean
    if typeModel == 'LDA':
        weight = mccModelLDA(trainFeatures, trainLabels, personValues, classes, currentClass)
    elif typeModel == 'QDA':
        weight = mccModelQDA(trainFeatures, trainLabels, personValues, classes, currentClass)

    return weight


def weightPerPersonCov(currentValues, personCov, currentClass, classes, trainFeatures, trainLabels, typeModel):
    personValues = currentValues.copy()
    personValues['cov'].at[currentClass] = personCov
    if typeModel == 'LDA':
        weight = mccModelLDA(trainFeatures, trainLabels, personValues, classes, currentClass)
    elif typeModel == 'QDA':
        weight = mccModelQDA(trainFeatures, trainLabels, personValues, classes, currentClass)
    return weight


# Matthews correlation coefficients

def mcc(TP, TN, FP, FN):
    mccValue = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    if np.isscalar(mccValue):
        if np.isnan(mccValue) or mccValue < 0:
            mccValue = 0
    else:
        mccValue[np.isnan(mccValue)] = 0
        mccValue[mccValue < 0] = 0

    return mccValue


def mccModelLDA(testFeatures, testLabels, model, classes, currentClass):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    currentClass += 1
    LDACov = DA_Classifiers.LDA_Cov(model, classes)
    for i in range(np.size(testLabels)):
        currentPredictor = DA_Classifiers.predictedModelLDA(testFeatures[i, :], model, classes, LDACov)
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


def mccModelQDA(testFeatures, testLabels, model, classes, currentClass):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    currentClass += 1
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


##################################################
# Weight Calculation


# Matthews correlation coefficients


def reduce_dataset(x, y, classes):
    reduce_X = []
    reduce_Y = []
    for cl in range(1, classes + 1):
        reduce_X.append(np.mean(x[y == cl], axis=0))
        reduce_Y.append(cl)
    return np.array(reduce_X), np.array(reduce_Y)


def subsetTraining_One(trainFeatures, numSamples):
    return trainFeatures[np.random.choice(len(trainFeatures), size=numSamples)]


def postProbabilities_weights_Calculation(unlabeledGestures, model, classes, type_DA, numberGestures, fewTrainFeatures,
                                          fewTrainLabels, weights=True, post=True):
    if post:
        post_probabilities = []
    if weights:
        weightsMean = []
        weightsCov = []

    for i in range(numberGestures):
        if post:
            post_probabilities.append(
                post_probabilities_Calculation(unlabeledGestures['features'].loc[i], model, classes, type_DA))
        if weights:
            wGestureMean = []
            wGestureCov = []
            for cla in range(classes):
                wGestureMean.append(
                    weightPerPersonMean(model, unlabeledGestures['mean'].loc[i], cla, classes, fewTrainFeatures,
                                        fewTrainLabels, type_DA))
                # if type_DA == 'QDA':
                wGestureCov.append(
                    weightPerPersonCov(model, unlabeledGestures['cov'].loc[i], cla, classes, fewTrainFeatures,
                                       fewTrainLabels, type_DA))
            weightsMean.append(np.array(wGestureMean))
            weightsCov.append(np.array(wGestureCov))

    # if type_DA == 'LDA':
    #     LDACov = DA_Classifiers.LDA_Cov(model, classes)
    #     for i in range(numberGestures):
    #         post_probabilities.append(
    #             DA_Classifiers.predictedModelLDAProb(unlabeledGestures['mean'].loc[i], model, classes, LDACov))
    # elif type_DA == 'QDA':
    #     for i in range(numberGestures):
    #         post_probabilities.append(
    #             DA_Classifiers.predictedModelQDAProb(unlabeledGestures['mean'].loc[i], model, classes))

    if post:
        unlabeledGestures['postProb'] = post_probabilities
    if weights:
        unlabeledGestures['wMean'] = weightsMean
        unlabeledGestures['wCov'] = weightsCov
    return unlabeledGestures


def model_PostProb_MSDA(currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewModel,
                        fewTrainFeatures, fewTrainLabels, type_DA, k, N):
    t = time.time()
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])

    numberUnlabeledGestures = len(unlabeledGestures.index)
    if numberUnlabeledGestures == 0:
        currentModel.at[0, '# gestures'] = N
    elif k > numberUnlabeledGestures > 0:
        unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
                                                                  numberUnlabeledGestures, fewTrainFeatures,
                                                                  fewTrainLabels, weights=True, post=True)
    else:
        unlabeledGestures = unlabeledGestures.tail(k - 1).reset_index(drop=True)
        unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
                                                                  k - 1, fewTrainFeatures, fewTrainLabels, weights=True,
                                                                  post=True)

    gestureMean = np.mean(trainFeatures, axis=0)
    gestureCov = np.cov(trainFeatures, rowvar=False)

    wGestureMean = []
    wGestureCov = []

    for cla in range(classes):
        wGestureMean.append(
            weightPerPersonMean(currentModel, gestureMean, cla, classes, fewTrainFeatures, fewTrainLabels, type_DA))
        # if type_DA == 'QDA':
        wGestureCov.append(
            weightPerPersonCov(currentModel, gestureCov, cla, classes, fewTrainFeatures, fewTrainLabels, type_DA))

    trainFeatures = subsetTraining_One(trainFeatures, 50)
    new_row = {'mean': gestureMean, 'cov': gestureCov, 'postProb': postProb_trainFeatures, 'wMean': wGestureMean,
               'wCov': wGestureCov, 'features': trainFeatures}

    unlabeledGestures = unlabeledGestures.append(new_row, ignore_index=True)

    # print('after RQ1')
    # print(unlabeledGestures[['postProb', 'wMean', 'wCov']])

    if k > currentModel['# gestures'].loc[0] - N + 1:
        wFewMean = np.ones(classes) * N
        wFewCov = np.ones(classes) * N

        weightsMean = unlabeledGestures['postProb'].values * unlabeledGestures['wMean'].values
        sumMean = wFewMean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        wFewMean /= sumMean
        print('mean weights')
        print(weightsMean)
        print('mean few')
        print(wFewMean)
        # weightsMean = np.nan_to_num(weightsMean)
        means = np.resize(unlabeledGestures['mean'].values, (classes, numberUnlabeledGestures + 1)).T * weightsMean

        # if type_DA == 'QDA':
        weightsCov = unlabeledGestures['postProb'].values * unlabeledGestures['wCov'].values
        sumCov = wFewCov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        wFewCov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, numberUnlabeledGestures + 1)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * wFewMean[cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * wFewCov[cla]



    elif k == currentModel['# gestures'].loc[0] - N + 1:
        wFewMean = np.ones(classes) * N
        wFewCov = np.ones(classes) * N

        weightsMean = unlabeledGestures['postProb'].values * unlabeledGestures['wMean'].values
        # if type_DA == 'QDA':
        weightsCov = unlabeledGestures['postProb'].values * unlabeledGestures['wCov'].values

        for cla in range(classes):
            adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * wFewMean[cla] + weightsMean[0][cla] * \
                                              unlabeledGestures['mean'].loc[0]
            adaptiveModel.at[cla, 'wMean_J'] = wFewMean[cla] + weightsMean[0][cla]
            # if type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * wFewCov[cla] + weightsCov[0][cla] * \
                                             unlabeledGestures['cov'].loc[0]
            adaptiveModel.at[cla, 'wCov_J'] = wFewCov[cla] + weightsCov[0][cla]

        sumMean = wFewMean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        wFewMean /= sumMean
        print('mean weights')
        print(weightsMean)
        print('mean equal')
        print(wFewMean)
        # weightsMean = np.nan_to_num(weightsMean)
        means = np.resize(unlabeledGestures['mean'], (classes, k)).T * weightsMean

        # if type_DA == 'QDA':
        sumCov = wFewCov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        wFewCov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, k)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * wFewMean[cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            #
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * wFewCov[cla]


    elif k < currentModel['# gestures'].loc[0] - N + 1:
        wJMean = np.zeros(classes)
        JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))

        weightsMean = unlabeledGestures['postProb'].values * unlabeledGestures['wMean'].values
        # if type_DA == 'QDA':
        wJCov = np.zeros(classes)
        JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
        weightsCov = unlabeledGestures['postProb'].values * unlabeledGestures['wCov'].values

        for cla in range(classes):
            JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsMean[0][cla] * \
                                              unlabeledGestures['mean'].loc[0]
            wJMean[cla] = currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsMean[0][cla]
            # if type_DA == 'QDA':
            JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wCov_J'].loc[cla]
            adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsCov[0][cla] * \
                                             unlabeledGestures['cov'].loc[0]
            wJCov[cla] = currentModel['wCov_J'].loc[cla]
            adaptiveModel.at[cla, 'wCov_J'] = currentModel['wCov_J'].loc[cla] + weightsCov[0][cla]

        sumMean = wJMean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        wJMean /= sumMean
        print('mean weights')
        print(weightsMean)
        print('mean wJ')
        print(wJMean)
        # weightsMean = np.nan_to_num(weightsMean)
        means = np.resize(unlabeledGestures['mean'], (classes, k)).T * weightsMean

        # if type_DA == 'QDA':
        sumCov = wJCov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        wJCov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, k)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + JMean[cla, :] * wJMean[cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + JCov[cla, :, :] * wJCov[cla]

    adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
    return adaptiveModel, time.time() - t, unlabeledGestures


def model_MSDA(currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewModel,
               fewTrainFeatures, fewTrainLabels, type_DA, k, N):
    t = time.time()
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])

    numberUnlabeledGestures = len(unlabeledGestures.index)
    if numberUnlabeledGestures == 0:
        currentModel.at[0, '# gestures'] = N
    elif k > numberUnlabeledGestures > 0:
        unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
                                                                  numberUnlabeledGestures, fewTrainFeatures,
                                                                  fewTrainLabels, weights=True, post=False)
    else:
        unlabeledGestures = unlabeledGestures.tail(k - 1).reset_index(drop=True)
        unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
                                                                  k - 1, fewTrainFeatures, fewTrainLabels, weights=True,
                                                                  post=False)

    gestureMean = np.mean(trainFeatures, axis=0)
    gestureCov = np.cov(trainFeatures, rowvar=False)

    wGestureMean = []
    wGestureCov = []

    for cla in range(classes):
        wGestureMean.append(
            weightPerPersonMean(currentModel, gestureMean, cla, classes, fewTrainFeatures, fewTrainLabels, type_DA))
        # if type_DA == 'QDA':
        wGestureCov.append(
            weightPerPersonCov(currentModel, gestureCov, cla, classes, fewTrainFeatures, fewTrainLabels, type_DA))

    trainFeatures = subsetTraining_One(trainFeatures, 50)
    new_row = {'mean': gestureMean, 'cov': gestureCov, 'wMean': wGestureMean, 'wCov': wGestureCov,
               'features': trainFeatures}

    unlabeledGestures = unlabeledGestures.append(new_row, ignore_index=True)

    # print('after RQ1')
    # print(unlabeledGestures[['postProb', 'wMean', 'wCov']])

    if k > currentModel['# gestures'].loc[0] - N + 1:
        wFewMean = np.ones(classes) * N
        wFewCov = np.ones(classes) * N

        weightsMean = unlabeledGestures['wMean'].values
        sumMean = wFewMean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        wFewMean /= sumMean
        print('mean weights')
        print(weightsMean)
        print('mean few')
        print(wFewMean)
        # weightsMean = np.nan_to_num(weightsMean)
        means = np.resize(unlabeledGestures['mean'].values, (classes, numberUnlabeledGestures + 1)).T * weightsMean

        # if type_DA == 'QDA':
        weightsCov = unlabeledGestures['wCov'].values
        sumCov = wFewCov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        wFewCov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, numberUnlabeledGestures + 1)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * wFewMean[cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * wFewCov[cla]



    elif k == currentModel['# gestures'].loc[0] - N + 1:
        wFewMean = np.ones(classes) * N
        wFewCov = np.ones(classes) * N

        weightsMean = unlabeledGestures['wMean'].values
        # if type_DA == 'QDA':
        weightsCov = unlabeledGestures['wCov'].values

        for cla in range(classes):
            adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * wFewMean[cla] + weightsMean[0][cla] * \
                                              unlabeledGestures['mean'].loc[0]
            adaptiveModel.at[cla, 'wMean_J'] = wFewMean[cla] + weightsMean[0][cla]
            # if type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * wFewCov[cla] + weightsCov[0][cla] * \
                                             unlabeledGestures['cov'].loc[0]
            adaptiveModel.at[cla, 'wCov_J'] = wFewCov[cla] + weightsCov[0][cla]

        sumMean = wFewMean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        wFewMean /= sumMean
        print('mean weights')
        print(weightsMean)
        print('mean equal')
        print(wFewMean)
        # weightsMean = np.nan_to_num(weightsMean)
        means = np.resize(unlabeledGestures['mean'], (classes, k)).T * weightsMean

        # if type_DA == 'QDA':
        sumCov = wFewCov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        wFewCov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, k)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * wFewMean[cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            #
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * wFewCov[cla]


    elif k < currentModel['# gestures'].loc[0] - N + 1:
        wJMean = np.zeros(classes)
        JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))

        weightsMean = unlabeledGestures['wMean'].values
        # if type_DA == 'QDA':
        wJCov = np.zeros(classes)
        JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
        weightsCov = unlabeledGestures['wCov'].values

        for cla in range(classes):
            JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsMean[0][cla] * \
                                              unlabeledGestures['mean'].loc[0]
            wJMean[cla] = currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsMean[0][cla]
            # if type_DA == 'QDA':
            JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wCov_J'].loc[cla]
            adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsCov[0][cla] * \
                                             unlabeledGestures['cov'].loc[0]
            wJCov[cla] = currentModel['wCov_J'].loc[cla]
            adaptiveModel.at[cla, 'wCov_J'] = currentModel['wCov_J'].loc[cla] + weightsCov[0][cla]

        sumMean = wJMean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        wJMean /= sumMean
        print('mean weights')
        print(weightsMean)
        print('mean wJ')
        print(wJMean)
        # weightsMean = np.nan_to_num(weightsMean)
        means = np.resize(unlabeledGestures['mean'], (classes, k)).T * weightsMean

        # if type_DA == 'QDA':
        sumCov = wJCov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        wJCov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, k)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + JMean[cla, :] * wJMean[cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + JCov[cla, :, :] * wJCov[cla]

    adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
    return adaptiveModel, time.time() - t, unlabeledGestures


def model_PostProb(currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewModel,
                   fewTrainFeatures, fewTrainLabels, type_DA, k, N):
    t = time.time()
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])

    numberUnlabeledGestures = len(unlabeledGestures.index)
    if numberUnlabeledGestures == 0:
        currentModel.at[0, '# gestures'] = N
    elif k > numberUnlabeledGestures > 0:
        unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
                                                                  numberUnlabeledGestures, fewTrainFeatures,
                                                                  fewTrainLabels, weights=False, post=True)
    else:
        unlabeledGestures = unlabeledGestures.tail(k - 1).reset_index(drop=True)
        unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
                                                                  k - 1, fewTrainFeatures, fewTrainLabels,
                                                                  weights=False, post=True)

    gestureMean = np.mean(trainFeatures, axis=0)
    gestureCov = np.cov(trainFeatures, rowvar=False)

    # wGestureMean = []
    # wGestureCov = []

    # for cla in range(classes):
    #     wGestureMean.append(
    #         weightPerPersonMean(currentModel, gestureMean, cla, classes, fewTrainFeatures, fewTrainLabels,type_DA))
    #     # if type_DA == 'QDA':
    #     wGestureCov.append(
    #         weightPerPersonCov(currentModel, gestureCov, cla, classes, fewTrainFeatures, fewTrainLabels,type_DA))

    trainFeatures = subsetTraining_One(trainFeatures, 50)
    new_row = {'mean': gestureMean, 'cov': gestureCov, 'postProb': postProb_trainFeatures, 'features': trainFeatures}

    unlabeledGestures = unlabeledGestures.append(new_row, ignore_index=True)

    # print('after RQ1')
    # print(unlabeledGestures[['postProb', 'wMean', 'wCov']])

    if k > currentModel['# gestures'].loc[0] - N + 1:
        wFewMean = np.ones(classes) * N
        wFewCov = np.ones(classes) * N

        weightsMean = unlabeledGestures['postProb'].values
        sumMean = wFewMean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        wFewMean /= sumMean
        print('mean weights')
        print(weightsMean)
        print('mean few')
        print(wFewMean)
        # weightsMean = np.nan_to_num(weightsMean)
        means = np.resize(unlabeledGestures['mean'].values, (classes, numberUnlabeledGestures + 1)).T * weightsMean

        # if type_DA == 'QDA':
        weightsCov = unlabeledGestures['postProb'].values
        sumCov = wFewCov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        wFewCov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, numberUnlabeledGestures + 1)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * wFewMean[cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * wFewCov[cla]



    elif k == currentModel['# gestures'].loc[0] - N + 1:
        wFewMean = np.ones(classes) * N
        wFewCov = np.ones(classes) * N

        weightsMean = unlabeledGestures['postProb'].values
        # if type_DA == 'QDA':
        weightsCov = unlabeledGestures['postProb'].values

        for cla in range(classes):
            adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * wFewMean[cla] + weightsMean[0][cla] * \
                                              unlabeledGestures['mean'].loc[0]
            adaptiveModel.at[cla, 'wMean_J'] = wFewMean[cla] + weightsMean[0][cla]
            # if type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * wFewCov[cla] + weightsCov[0][cla] * \
                                             unlabeledGestures['cov'].loc[0]
            adaptiveModel.at[cla, 'wCov_J'] = wFewCov[cla] + weightsCov[0][cla]

        sumMean = wFewMean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        wFewMean /= sumMean
        print('mean weights')
        print(weightsMean)
        print('mean equal')
        print(wFewMean)
        # weightsMean = np.nan_to_num(weightsMean)
        means = np.resize(unlabeledGestures['mean'], (classes, k)).T * weightsMean

        # if type_DA == 'QDA':
        sumCov = wFewCov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        wFewCov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, k)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * wFewMean[cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            #
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * wFewCov[cla]


    elif k < currentModel['# gestures'].loc[0] - N + 1:
        wJMean = np.zeros(classes)
        JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))

        weightsMean = unlabeledGestures['postProb'].values
        # if type_DA == 'QDA':
        wJCov = np.zeros(classes)
        JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
        weightsCov = unlabeledGestures['postProb'].values

        for cla in range(classes):
            JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsMean[0][cla] * \
                                              unlabeledGestures['mean'].loc[0]
            wJMean[cla] = currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsMean[0][cla]
            # if type_DA == 'QDA':
            JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wCov_J'].loc[cla]
            adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsCov[0][cla] * \
                                             unlabeledGestures['cov'].loc[0]
            wJCov[cla] = currentModel['wCov_J'].loc[cla]
            adaptiveModel.at[cla, 'wCov_J'] = currentModel['wCov_J'].loc[cla] + weightsCov[0][cla]

        sumMean = wJMean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        wJMean /= sumMean
        print('mean weights')
        print(weightsMean)
        print('mean wJ')
        print(wJMean)
        # weightsMean = np.nan_to_num(weightsMean)
        means = np.resize(unlabeledGestures['mean'], (classes, k)).T * weightsMean

        # if type_DA == 'QDA':
        sumCov = wJCov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        wJCov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, k)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + JMean[cla, :] * wJMean[cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + JCov[cla, :, :] * wJCov[cla]

    adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
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
