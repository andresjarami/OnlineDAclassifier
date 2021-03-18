# %% Libraries
import time
import math
import numpy as np
import pandas as pd
from scipy.spatial import distance

import DA_Classifiers as DA_Classifiers


# %% Divergences
def KLdivergence(mean0, mean1, k, cov0, cov1):
    exp1 = np.trace(np.dot(np.linalg.inv(cov1), cov0))
    exp2 = np.dot(np.dot((mean1 - mean0).T, np.linalg.inv(cov1)), (mean1 - mean0))
    exp3 = np.log(np.linalg.det(cov1) / np.linalg.det(cov0))
    return 0.5 * (exp1 + exp2 - k + exp3)


def JSdivergence(mean0, mean1, k, cov0, cov1):
    meanM = (mean0 + mean1) / 2
    covM = (cov0 + cov1) / 2
    js = KLdivergence(mean0, meanM, k, cov0, covM) + KLdivergence(mean1, meanM, k, cov1, covM)
    # js /= np.log(2)
    return js / 2


# def weightPerPersonMean_KL(currentValues, personMean, currentClass, classes, trainFeatures, trainLabels, type_DA):
#     personValues = currentValues.copy()
#     personValues['mean'].at[currentClass] = personMean
#     return klDivergenceModel(trainFeatures, trainLabels, personValues, classes, currentClass)
#
#
# def weightPerPersonCov_KL(currentValues, personCov, currentClass, classes, trainFeatures, trainLabels, type_DA):
#     personValues = currentValues.copy()
#     personValues['cov'].at[currentClass] = personCov
#     return klDivergenceModel(trainFeatures, trainLabels, personValues, classes, currentClass)


# def predictedKL(mean, cov, model, classes):
#     d = np.zeros([classes])
#     for cla in range(classes):
#         d[cla] = JSdivergence(model['mean'].loc[cla], mean, np.size(mean), model['cov'].loc[cla], cov)
#         # d = d - d[np.argmin(d)]
#     # d / d.sum()
#     return np.argmin(d)


# def klDivergenceModel(testFeatures, testLabels, model, classes, currentClass):
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#     for i in range(classes):
#         meanAux = np.mean(testFeatures[testLabels == i + 1, :], axis=0)
#         covAux = np.cov(testFeatures[testLabels == i + 1, :], rowvar=False)
#         KLresult = predictedKL(meanAux, covAux, model, classes)
#         if KLresult == i:
#             if KLresult == currentClass:
#                 TP += 1
#             else:
#                 TN += 1
#         else:
#             if i == currentClass:
#                 FN += 1
#             else:
#                 FP += 1
#     return mcc(TP, TN, FP, FN)


def matrixDivergences(Features, Labels, model, classes):
    matrix = np.zeros((classes, classes))
    for i in range(classes):
        meanSamples = np.mean(Features[Labels == i + 1, :], axis=0)
        covSamples = np.cov(Features[Labels == i + 1, :], rowvar=False)
        for j in range(classes):
            covModel = model.loc[j, 'cov']
            meanModel = model.loc[j, 'mean']
            matrix[j, i] = JSdivergence(meanSamples, meanModel, np.size(meanSamples), covSamples, covModel)
    return matrix


def rowDivergences(Features, Labels, meanModel, covModel, classes):
    row = np.zeros((1, classes))
    for i in range(classes):
        meanSamples = np.mean(Features[Labels == i + 1, :], axis=0)
        covSamples = np.cov(Features[Labels == i + 1, :], rowvar=False)
        row[0, i] = JSdivergence(meanSamples, meanModel, np.size(meanSamples), covSamples, covModel)
    return row


def mcc_from_matrixDivergences(matrix, classes, currentClass):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for cla in range(classes):
        if np.argmin(matrix[:, cla]) == cla:
            if cla == currentClass:
                TP += 1
            else:
                TN += 1
        else:
            if cla == currentClass:
                FP += 1
            else:
                FN += 1
    return mcc(TP, TN, FP, FN)


def weight_MSDA_JS (currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures, labeledGesturesLabels,
                   type_DA):
    matrix_D = matrixDivergences(labeledGesturesFeatures, labeledGesturesLabels, currentModel, classes)

    wGestureMean = []
    wGestureCov = []
    for cla in range(classes):
        matrix_D_mean = matrix_D.copy()
        matrix_D_cov = matrix_D.copy()
        matrix_D_mean[cla, :] = rowDivergences(labeledGesturesFeatures, labeledGesturesLabels, gestureMean,
                                               currentModel['cov'].loc[cla], classes)
        matrix_D_cov[cla, :] = rowDivergences(labeledGesturesFeatures, labeledGesturesLabels,
                                              currentModel['mean'].loc[cla], gestureCov, classes)
        wGestureMean.append(mcc_from_matrixDivergences(matrix_D_mean, classes, cla))
        wGestureCov.append(mcc_from_matrixDivergences(matrix_D_cov, classes, cla))
    return wGestureMean, wGestureCov


# %% Weight Calculation
def weightPerPersonMean(currentValues, personMean, currentClass, classes, trainFeatures, trainLabels, type_DA):
    personValues = currentValues.copy()
    personValues['mean'].at[currentClass] = personMean
    if type_DA == 'LDA':
        weight = mccModelLDA(trainFeatures, trainLabels, personValues, classes, currentClass)
    elif type_DA == 'QDA':
        weight = mccModelQDA(trainFeatures, trainLabels, personValues, classes, currentClass)
    return weight


def weightPerPersonCov(currentValues, personCov, currentClass, classes, trainFeatures, trainLabels, type_DA):
    personValues = currentValues.copy()
    personValues['cov'].at[currentClass] = personCov
    if type_DA == 'LDA':
        weight = mccModelLDA(trainFeatures, trainLabels, personValues, classes, currentClass)
    elif type_DA == 'QDA':
        weight = mccModelQDA(trainFeatures, trainLabels, personValues, classes, currentClass)
    return weight


# %% Matthews correlation coefficients

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


# %% reduce random the datasets
def subsetTraining_One(trainFeatures, numSamples):
    return trainFeatures[np.random.choice(len(trainFeatures), size=numSamples)]


# %% Models

def model_PostProb_MSDA(currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewModel,
                        labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
    t = time.time()
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])

    numberUnlabeledGestures = len(unlabeledGestures.index)
    if numberUnlabeledGestures == 0:
        currentModel.at[0, '# gestures'] = shotStart
    elif samplesInMemory >= numberUnlabeledGestures > 0:
        unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
                                                                  numberUnlabeledGestures, labeledGesturesFeatures,
                                                                  labeledGesturesLabels, weights=True, post=True)
    else:
        unlabeledGestures = unlabeledGestures.tail(samplesInMemory).reset_index(drop=True)
        unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
                                                                  samplesInMemory, labeledGesturesFeatures,
                                                                  labeledGesturesLabels,
                                                                  weights=True,
                                                                  post=True)

    gestureMean = np.mean(trainFeatures, axis=0)
    gestureCov = np.cov(trainFeatures, rowvar=False)

    wGestureMean = []
    wGestureCov = []

    for cla in range(classes):
        wGestureMean.append(
            weightPerPersonMean(currentModel, gestureMean, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,
                                type_DA))

        wGestureCov.append(
            weightPerPersonCov(currentModel, gestureCov, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,
                               type_DA))

    trainFeatures = subsetTraining_One(trainFeatures, 50)
    new_row = {'mean': gestureMean, 'cov': gestureCov, 'postProb': postProb_trainFeatures, 'wMean': wGestureMean,
               'wCov': wGestureCov, 'features': trainFeatures}

    unlabeledGestures = unlabeledGestures.append(new_row, ignore_index=True)

    # print('after RQ1')
    # print(unlabeledGestures[['postProb', 'wMean', 'wCov']])

    if samplesInMemory > currentModel['# gestures'].loc[0] - shotStart:
        w_labeledGestures_Mean = np.ones(classes) * shotStart
        w_labeledGestures_Cov = np.ones(classes) * shotStart

        weightsMean = unlabeledGestures['postProb'].values * unlabeledGestures['wMean'].values
        sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        w_labeledGestures_Mean /= sumMean
        # print('mean weights')
        # print(weightsMean)
        # print('mean few')
        # print(w_labeledGestures_Mean)

        means = np.resize(unlabeledGestures['mean'].values, (classes, numberUnlabeledGestures + 1)).T * weightsMean

        # if type_DA == 'QDA':
        weightsCov = unlabeledGestures['postProb'].values * unlabeledGestures['wCov'].values
        sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        w_labeledGestures_Cov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, numberUnlabeledGestures + 1)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
                cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]



    elif samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
        w_labeledGestures_Mean = np.ones(classes) * shotStart
        w_labeledGestures_Cov = np.ones(classes) * shotStart

        weightsMean = unlabeledGestures['postProb'].values * unlabeledGestures['wMean'].values
        # if type_DA == 'QDA':
        weightsCov = unlabeledGestures['postProb'].values * unlabeledGestures['wCov'].values

        for cla in range(classes):
            adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
                cla] * \
                                              unlabeledGestures['mean'].loc[0]
            adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
            # if type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
                cla] * \
                                             unlabeledGestures['cov'].loc[0]
            adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures_Cov[cla] + weightsCov[0][cla]

        sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        w_labeledGestures_Mean /= sumMean
        # print('mean weights')
        # print(weightsMean)
        # print('mean equal')
        # print(w_labeledGestures_Mean)
        means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean

        # if type_DA == 'QDA':
        sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        w_labeledGestures_Cov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
                cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            #
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]


    elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
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
        # print('mean weights')
        # print(weightsMean)
        # print('mean wJ')
        # print(wJMean)
        means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean

        # if type_DA == 'QDA':
        sumCov = wJCov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        wJCov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov

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
               labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
    t = time.time()
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])

    numberUnlabeledGestures = len(unlabeledGestures.index)
    if numberUnlabeledGestures == 0:
        currentModel.at[0, '# gestures'] = shotStart
    elif samplesInMemory >= numberUnlabeledGestures > 0:
        unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
                                                                  numberUnlabeledGestures, labeledGesturesFeatures,
                                                                  labeledGesturesLabels, weights=True, post=False)
    else:
        unlabeledGestures = unlabeledGestures.tail(samplesInMemory).reset_index(drop=True)
        unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
                                                                  samplesInMemory, labeledGesturesFeatures,
                                                                  labeledGesturesLabels, weights=True, post=False)

    gestureMean = np.mean(trainFeatures, axis=0)
    gestureCov = np.cov(trainFeatures, rowvar=False)

    wGestureMean = []
    wGestureCov = []

    for cla in range(classes):
        wGestureMean.append(
            weightPerPersonMean(currentModel, gestureMean, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,
                                type_DA))
        # if type_DA == 'QDA':
        wGestureCov.append(
            weightPerPersonCov(currentModel, gestureCov, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,
                               type_DA))

    trainFeatures = subsetTraining_One(trainFeatures, 50)
    new_row = {'mean': gestureMean, 'cov': gestureCov, 'wMean': wGestureMean, 'wCov': wGestureCov,
               'features': trainFeatures}

    unlabeledGestures = unlabeledGestures.append(new_row, ignore_index=True)

    # print('after RQ1')
    # print(unlabeledGestures[['postProb', 'wMean', 'wCov']])

    if samplesInMemory > currentModel['# gestures'].loc[0] - shotStart:
        w_labeledGestures_Mean = np.ones(classes) * shotStart
        w_labeledGestures_Cov = np.ones(classes) * shotStart

        weightsMean = unlabeledGestures['wMean'].values
        sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        w_labeledGestures_Mean /= sumMean
        # print('mean weights')
        # print(weightsMean)
        # print('mean few')
        # print(w_labeledGestures_Mean)
        means = np.resize(unlabeledGestures['mean'].values, (classes, numberUnlabeledGestures + 1)).T * weightsMean

        # if type_DA == 'QDA':
        weightsCov = unlabeledGestures['wCov'].values
        sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        w_labeledGestures_Cov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, numberUnlabeledGestures + 1)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
                cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]



    elif samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
        w_labeledGestures_Mean = np.ones(classes) * shotStart
        w_labeledGestures_Cov = np.ones(classes) * shotStart

        weightsMean = unlabeledGestures['wMean'].values
        # if type_DA == 'QDA':
        weightsCov = unlabeledGestures['wCov'].values

        for cla in range(classes):
            adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
                cla] * \
                                              unlabeledGestures['mean'].loc[0]
            adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
            # if type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
                cla] * \
                                             unlabeledGestures['cov'].loc[0]
            adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures_Cov[cla] + weightsCov[0][cla]

        sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        w_labeledGestures_Mean /= sumMean
        # print('mean weights')
        # print(weightsMean)
        # print('mean equal')
        # print(w_labeledGestures_Mean)
        means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean

        # if type_DA == 'QDA':
        sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        w_labeledGestures_Cov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
                cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            #
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]


    elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
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
        # print('mean weights')
        # print(weightsMean)
        # print('mean wJ')
        # print(wJMean)
        means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean

        # if type_DA == 'QDA':
        sumCov = wJCov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        wJCov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov

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
                   labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
    t = time.time()
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])

    numberUnlabeledGestures = len(unlabeledGestures.index)
    if numberUnlabeledGestures == 0:
        currentModel.at[0, '# gestures'] = shotStart
    elif samplesInMemory >= numberUnlabeledGestures > 0:
        unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
                                                                  numberUnlabeledGestures, labeledGesturesFeatures,
                                                                  labeledGesturesLabels, weights=False, post=True)
    else:
        unlabeledGestures = unlabeledGestures.tail(samplesInMemory).reset_index(drop=True)
        unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
                                                                  samplesInMemory, labeledGesturesFeatures,
                                                                  labeledGesturesLabels,
                                                                  weights=False, post=True)

    gestureMean = np.mean(trainFeatures, axis=0)
    gestureCov = np.cov(trainFeatures, rowvar=False)

    # wGestureMean = []
    # wGestureCov = []

    # for cla in range(classes):
    #     wGestureMean.append(
    #         weightPerPersonMean(currentModel, gestureMean, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,type_DA))
    #     # if type_DA == 'QDA':
    #     wGestureCov.append(
    #         weightPerPersonCov(currentModel, gestureCov, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,type_DA))

    trainFeatures = subsetTraining_One(trainFeatures, 50)
    new_row = {'mean': gestureMean, 'cov': gestureCov, 'postProb': postProb_trainFeatures, 'features': trainFeatures}

    unlabeledGestures = unlabeledGestures.append(new_row, ignore_index=True)

    # print('after RQ1')
    # print(unlabeledGestures[['postProb', 'wMean', 'wCov']])

    if samplesInMemory > currentModel['# gestures'].loc[0] - shotStart:
        w_labeledGestures_Mean = np.ones(classes) * shotStart
        w_labeledGestures_Cov = np.ones(classes) * shotStart

        weightsMean = unlabeledGestures['postProb'].values
        sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        w_labeledGestures_Mean /= sumMean
        # print('mean weights')
        # print(weightsMean)
        # print('mean few')
        # print(w_labeledGestures_Mean)
        means = np.resize(unlabeledGestures['mean'].values, (classes, numberUnlabeledGestures + 1)).T * weightsMean

        # if type_DA == 'QDA':
        weightsCov = unlabeledGestures['postProb'].values
        sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        w_labeledGestures_Cov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, numberUnlabeledGestures + 1)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
                cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]



    elif samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
        w_labeledGestures_Mean = np.ones(classes) * shotStart
        w_labeledGestures_Cov = np.ones(classes) * shotStart

        weightsMean = unlabeledGestures['postProb'].values
        # if type_DA == 'QDA':
        weightsCov = unlabeledGestures['postProb'].values

        for cla in range(classes):
            adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
                cla] * \
                                              unlabeledGestures['mean'].loc[0]
            adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
            # if type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
                cla] * \
                                             unlabeledGestures['cov'].loc[0]
            adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures_Cov[cla] + weightsCov[0][cla]

        sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        w_labeledGestures_Mean /= sumMean
        # print('mean weights')
        # print(weightsMean)
        # print('mean equal')
        # print(w_labeledGestures_Mean)
        means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean

        # if type_DA == 'QDA':
        sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        w_labeledGestures_Cov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
                cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            #
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]


    elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
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
        # print('mean weights')
        # print(weightsMean)
        # print('mean wJ')
        # print(wJMean)
        means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean

        # if type_DA == 'QDA':
        sumCov = wJCov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        wJCov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov

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


def model_Baseline(currentModel, unlabeledGestures, classes, trainFeatures, trainLabels, postProb_trainFeatures,
                   fewModel, labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
    t = time.time()
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])

    numberUnlabeledGestures = len(unlabeledGestures.index)

    postProb_trainFeatures = np.zeros(classes)
    postProb_trainFeatures[int(np.mean(trainLabels)) - 1] = 1
    if numberUnlabeledGestures == 0:
        currentModel.at[0, '# gestures'] = shotStart
    else:
        unlabeledGestures = unlabeledGestures.tail(samplesInMemory).reset_index(drop=True)

    gestureMean = np.mean(trainFeatures, axis=0)
    gestureCov = np.cov(trainFeatures, rowvar=False)

    # wGestureMean = []
    # wGestureCov = []

    # for cla in range(classes):
    #     wGestureMean.append(
    #         weightPerPersonMean(currentModel, gestureMean, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,type_DA))
    #     # if type_DA == 'QDA':
    #     wGestureCov.append(
    #         weightPerPersonCov(currentModel, gestureCov, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,type_DA))

    trainFeatures = subsetTraining_One(trainFeatures, 50)
    new_row = {'mean': gestureMean, 'cov': gestureCov, 'postProb': postProb_trainFeatures, 'features': trainFeatures}

    unlabeledGestures = unlabeledGestures.append(new_row, ignore_index=True)

    # print('after RQ1')
    # print(unlabeledGestures[['postProb', 'wMean', 'wCov']])

    if samplesInMemory > currentModel['# gestures'].loc[0] - shotStart:
        w_labeledGestures_Mean = np.ones(classes) * shotStart
        w_labeledGestures_Cov = np.ones(classes) * shotStart

        weightsMean = unlabeledGestures['postProb'].values
        sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        w_labeledGestures_Mean /= sumMean
        # print('mean weights')
        # print(weightsMean)
        # print('mean few')
        # print(w_labeledGestures_Mean)
        means = np.resize(unlabeledGestures['mean'].values, (classes, numberUnlabeledGestures + 1)).T * weightsMean

        # if type_DA == 'QDA':
        weightsCov = unlabeledGestures['postProb'].values
        sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        w_labeledGestures_Cov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, numberUnlabeledGestures + 1)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
                cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]



    elif samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
        w_labeledGestures_Mean = np.ones(classes) * shotStart
        w_labeledGestures_Cov = np.ones(classes) * shotStart

        weightsMean = unlabeledGestures['postProb'].values
        # if type_DA == 'QDA':
        weightsCov = unlabeledGestures['postProb'].values

        for cla in range(classes):
            adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
                cla] * \
                                              unlabeledGestures['mean'].loc[0]
            adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
            # if type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
                cla] * \
                                             unlabeledGestures['cov'].loc[0]
            adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures_Cov[cla] + weightsCov[0][cla]

        sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        w_labeledGestures_Mean /= sumMean
        # print('mean weights')
        # print(weightsMean)
        # print('mean equal')
        # print(w_labeledGestures_Mean)
        means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean

        # if type_DA == 'QDA':
        sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        w_labeledGestures_Cov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
                cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            #
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]


    elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
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
        # print('mean weights')
        # print(weightsMean)
        # print('mean wJ')
        # print(wJMean)
        means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean

        # if type_DA == 'QDA':
        sumCov = wJCov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        wJCov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov

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


def model_MSDA_KL(currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewModel,
                  labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
    t = time.time()
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])

    numberUnlabeledGestures = len(unlabeledGestures.index)
    if numberUnlabeledGestures == 0:
        currentModel.at[0, '# gestures'] = shotStart
    elif samplesInMemory >= numberUnlabeledGestures > 0:
        unlabeledGestures = postProbabilities_weights_Calculation_KL(unlabeledGestures, currentModel, classes, type_DA,
                                                                     numberUnlabeledGestures, labeledGesturesFeatures,
                                                                     labeledGesturesLabels, weights=True, post=False)
    else:
        unlabeledGestures = unlabeledGestures.tail(samplesInMemory).reset_index(drop=True)
        unlabeledGestures = postProbabilities_weights_Calculation_KL(unlabeledGestures, currentModel, classes, type_DA,
                                                                     samplesInMemory, labeledGesturesFeatures,
                                                                     labeledGesturesLabels, weights=True, post=False)

    gestureMean = np.mean(trainFeatures, axis=0)
    gestureCov = np.cov(trainFeatures, rowvar=False)

    wGestureMean, wGestureCov = weight_MSDA_JS (currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures,
                                               labeledGesturesLabels, type_DA)

    trainFeatures = subsetTraining_One(trainFeatures, 50)
    new_row = {'mean': gestureMean, 'cov': gestureCov, 'wMean': wGestureMean, 'wCov': wGestureCov,
               'features': trainFeatures}

    unlabeledGestures = unlabeledGestures.append(new_row, ignore_index=True)

    # print('after RQ1')
    # print(unlabeledGestures[['postProb', 'wMean', 'wCov']])

    if samplesInMemory > currentModel['# gestures'].loc[0] - shotStart:
        w_labeledGestures_Mean = np.ones(classes) * shotStart
        w_labeledGestures_Cov = np.ones(classes) * shotStart

        weightsMean = unlabeledGestures['wMean'].values
        sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        w_labeledGestures_Mean /= sumMean
        # print('mean weights')
        # print(weightsMean)
        # print('mean few')
        # print(w_labeledGestures_Mean)
        means = np.resize(unlabeledGestures['mean'].values, (classes, numberUnlabeledGestures + 1)).T * weightsMean

        # if type_DA == 'QDA':
        weightsCov = unlabeledGestures['wCov'].values
        sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        w_labeledGestures_Cov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, numberUnlabeledGestures + 1)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
                cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]



    elif samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
        w_labeledGestures_Mean = np.ones(classes) * shotStart
        w_labeledGestures_Cov = np.ones(classes) * shotStart

        weightsMean = unlabeledGestures['wMean'].values
        # if type_DA == 'QDA':
        weightsCov = unlabeledGestures['wCov'].values

        for cla in range(classes):
            adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
                cla] * \
                                              unlabeledGestures['mean'].loc[0]
            adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
            # if type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
                cla] * \
                                             unlabeledGestures['cov'].loc[0]
            adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures_Cov[cla] + weightsCov[0][cla]

        sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        w_labeledGestures_Mean /= sumMean
        # print('mean weights')
        # print(weightsMean)
        # print('mean equal')
        # print(w_labeledGestures_Mean)
        means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean

        # if type_DA == 'QDA':
        sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        w_labeledGestures_Cov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
                cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            #
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]


    elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
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
        # print('mean weights')
        # print(weightsMean)
        # print('mean wJ')
        # print(wJMean)
        means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean

        # if type_DA == 'QDA':
        sumCov = wJCov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        wJCov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov

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


def model_PostProb_MSDA_JS(currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewModel,
                           labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
    t = time.time()
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])

    numberUnlabeledGestures = len(unlabeledGestures.index)
    if numberUnlabeledGestures == 0:
        currentModel.at[0, '# gestures'] = shotStart
    elif samplesInMemory >= numberUnlabeledGestures > 0:
        unlabeledGestures = postProbabilities_weights_Calculation_KL(unlabeledGestures, currentModel, classes, type_DA,
                                                                     numberUnlabeledGestures, labeledGesturesFeatures,
                                                                     labeledGesturesLabels, weights=True, post=True)
    else:
        unlabeledGestures = unlabeledGestures.tail(samplesInMemory).reset_index(drop=True)
        unlabeledGestures = postProbabilities_weights_Calculation_KL(unlabeledGestures, currentModel, classes, type_DA,
                                                                     samplesInMemory, labeledGesturesFeatures,
                                                                     labeledGesturesLabels, weights=True, post=True)
    # print(type_DA)
    # print(unlabeledGestures)
    gestureMean = np.mean(trainFeatures, axis=0)
    gestureCov = np.cov(trainFeatures, rowvar=False)

    wGestureMean, wGestureCov = weight_MSDA_JS (currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures,
                                               labeledGesturesLabels, type_DA)

    # wGestureMean = []
    # wGestureCov = []
    #
    # for cla in range(classes):
    #     wGestureMean.append(
    #         weightPerPersonMean_KL(currentModel, gestureMean, cla, classes, labeledGesturesFeatures,
    #                                labeledGesturesLabels,
    #                                type_DA))
    #
    #     wGestureCov.append(
    #         weightPerPersonCov_KL(currentModel, gestureCov, cla, classes, labeledGesturesFeatures,
    #                               labeledGesturesLabels,
    #                               type_DA))

    trainFeatures = subsetTraining_One(trainFeatures, 50)
    new_row = {'mean': gestureMean, 'cov': gestureCov, 'postProb': postProb_trainFeatures, 'wMean': wGestureMean,
               'wCov': wGestureCov, 'features': trainFeatures}

    unlabeledGestures = unlabeledGestures.append(new_row, ignore_index=True)

    # print('after RQ1')
    # print(unlabeledGestures[['postProb', 'wMean', 'wCov']])

    if samplesInMemory > currentModel['# gestures'].loc[0] - shotStart:
        # print('less than memmory')
        w_labeledGestures_Mean = np.ones(classes) * shotStart
        w_labeledGestures_Cov = np.ones(classes) * shotStart

        weightsMean = unlabeledGestures['postProb'].values * unlabeledGestures['wMean'].values
        sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        w_labeledGestures_Mean /= sumMean
        # print('mean weights')
        # print(weightsMean)
        # print('mean few')
        # print(w_labeledGestures_Mean)
        means = np.resize(unlabeledGestures['mean'].values, (classes, numberUnlabeledGestures + 1)).T * weightsMean

        # if type_DA == 'QDA':
        weightsCov = unlabeledGestures['postProb'].values * unlabeledGestures['wCov'].values
        sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        w_labeledGestures_Cov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, numberUnlabeledGestures + 1)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
                cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]



    elif samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
        # print('equal than memmory')
        w_labeledGestures_Mean = np.ones(classes) * shotStart
        w_labeledGestures_Cov = np.ones(classes) * shotStart

        weightsMean = unlabeledGestures['postProb'].values * unlabeledGestures['wMean'].values
        # if type_DA == 'QDA':
        weightsCov = unlabeledGestures['postProb'].values * unlabeledGestures['wCov'].values

        for cla in range(classes):
            adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
                cla] * \
                                              unlabeledGestures['mean'].loc[0]
            adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
            # if type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
                cla] * \
                                             unlabeledGestures['cov'].loc[0]
            adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures_Cov[cla] + weightsCov[0][cla]

        sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
        weightsMean = list(weightsMean) / sumMean
        w_labeledGestures_Mean /= sumMean
        # print('mean weights')
        # print(weightsMean)
        # print('mean equal')
        # print(w_labeledGestures_Mean)
        means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean

        # if type_DA == 'QDA':
        sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        w_labeledGestures_Cov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
                cla]
            # if type_DA == 'LDA':
            #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
            #             1 + currentModel['# gestures'].loc[0])
            #
            # elif type_DA == 'QDA':
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]


    elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
        # print('greater than memmory')
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
        # print('mean weights')
        # print(weightsMean)
        # print('mean wJ')
        # print(wJMean)
        means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean

        # if type_DA == 'QDA':
        sumCov = wJCov + np.sum(weightsCov, axis=0)
        weightsCov = list(weightsCov) / sumCov
        wJCov /= sumCov
        # weightsCov = np.nan_to_num(weightsCov)
        covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov

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


# %% Post Probabilities
def postProbabilities_weights_Calculation(unlabeledGestures, model, classes, type_DA, numberGestures,
                                          labeledGesturesFeatures,
                                          labeledGesturesLabels, weights=True, post=True):
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
                    weightPerPersonMean(model, unlabeledGestures['mean'].loc[i], cla, classes, labeledGesturesFeatures,
                                        labeledGesturesLabels, type_DA))
                wGestureCov.append(
                    weightPerPersonCov(model, unlabeledGestures['cov'].loc[i], cla, classes, labeledGesturesFeatures,
                                       labeledGesturesLabels, type_DA))

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


def postProbabilities_weights_Calculation_KL(unlabeledGestures, model, classes, type_DA, numberGestures,
                                             labeledGesturesFeatures, labeledGesturesLabels, weights=True, post=True):
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
            wGestureMean, wGestureCov = weight_MSDA_JS (model, unlabeledGestures['mean'].loc[i],
                                                       unlabeledGestures['cov'].loc[i], classes,
                                                       labeledGesturesFeatures, labeledGesturesLabels, type_DA)

            weightsMean.append(np.array(wGestureMean))
            weightsCov.append(np.array(wGestureCov))

    if post:
        unlabeledGestures['postProb'] = post_probabilities
    if weights:
        unlabeledGestures['wMean'] = weightsMean
        unlabeledGestures['wCov'] = weightsCov
    return unlabeledGestures
