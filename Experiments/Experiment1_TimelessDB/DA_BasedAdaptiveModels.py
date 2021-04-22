import time

import numpy as np
import pandas as pd
from scipy.spatial import distance

import DA_Classifiers as DA_Classifiers
import SemiSupervised as SemiSupervised


# Reduced Daily Recalibration of Myoelectric Prosthesis Classifiers Based on Domain Adaptation
# LIU IMPLEMENTATION
def weightDenominatorLiu(currentMean, preTrainedDataMatrix):
    weightDenominatorV = 0
    for i in range(len(preTrainedDataMatrix.index)):
        weightDenominatorV = weightDenominatorV + (
                1 / distance.mahalanobis(currentMean, preTrainedDataMatrix['mean'].loc[i],
                                         np.linalg.inv(preTrainedDataMatrix['cov'].loc[i])))
    return weightDenominatorV


def reTrainedMeanLiu(r, currentMean, preTrainedDataMatrix, weightDenominatorV, allFeatures):
    sumAllPreTrainedMean_Weighted = np.zeros((1, allFeatures))
    for i in range(len(preTrainedDataMatrix.index)):
        sumAllPreTrainedMean_Weighted = np.add(sumAllPreTrainedMean_Weighted, preTrainedDataMatrix['mean'].loc[i] * (
                1 / distance.mahalanobis(currentMean, preTrainedDataMatrix['mean'].loc[i],
                                         np.linalg.inv(preTrainedDataMatrix['cov'].loc[i]))))

    reTrainedMeanValue = np.add((1 - r) * currentMean, (r / weightDenominatorV) * sumAllPreTrainedMean_Weighted)
    return reTrainedMeanValue


def reTrainedCovLiu(r, currentMean, currentCov, preTrainedDataMatrix, weightDenominatorV, allFeatures):
    sumAllPreTrainedCov_Weighted = np.zeros((allFeatures, allFeatures))
    for i in range(len(preTrainedDataMatrix.index)):
        sumAllPreTrainedCov_Weighted = np.add(sumAllPreTrainedCov_Weighted, preTrainedDataMatrix['cov'][i] * (
                1 / distance.mahalanobis(currentMean, preTrainedDataMatrix['mean'][i],
                                         np.linalg.inv(preTrainedDataMatrix['cov'][i]))))

    reTrainedCovValue = np.add((1 - r) * currentCov, (r / weightDenominatorV) * sumAllPreTrainedCov_Weighted)
    return reTrainedCovValue


def LiuModel(currentValues, preTrainedDataMatrix, classes, allFeatures):
    trainedModel = pd.DataFrame(columns=['cov', 'mean', 'class'])
    r = 0.5
    for cla in range(0, classes):
        preTrainedMatrix_Class = pd.DataFrame(
            preTrainedDataMatrix[['cov', 'mean']].loc[(preTrainedDataMatrix['class'] == cla + 1)])
        preTrainedMatrix_Class = preTrainedMatrix_Class.reset_index(drop=True)
        currentCov = currentValues['cov'].loc[cla]
        currentMean = currentValues['mean'].loc[cla]
        weightDenominatorV = weightDenominatorLiu(currentMean, preTrainedMatrix_Class)
        trainedModel.at[cla, 'cov'] = reTrainedCovLiu(r, currentMean, currentCov, preTrainedMatrix_Class,
                                                      weightDenominatorV, allFeatures)
        trainedModel.at[cla, 'mean'] = \
            reTrainedMeanLiu(r, currentMean, preTrainedMatrix_Class, weightDenominatorV, allFeatures)[0]
        trainedModel.at[cla, 'class'] = cla + 1

    return trainedModel


# VIDOVIC IMPLEMENTATION
def VidovicModel(currentValues, preTrainedDataMatrix, classes, allFeatures):
    trainedModelL = pd.DataFrame(columns=['cov', 'mean', 'class'])
    trainedModelQ = pd.DataFrame(columns=['cov', 'mean', 'class'])

    preTrainedCov = np.zeros((allFeatures, allFeatures))
    preTrainedMean = np.zeros((1, allFeatures))

    for cla in range(0, classes):
        preTrainedMatrix_Class = pd.DataFrame(
            preTrainedDataMatrix[['cov', 'mean']].loc[(preTrainedDataMatrix['class'] == cla + 1)])
        preTrainedMatrix_Class = preTrainedMatrix_Class.reset_index(drop=True)
        for i in range(len(preTrainedMatrix_Class.index)):
            preTrainedCov += preTrainedDataMatrix['cov'][i]
            preTrainedMean += preTrainedDataMatrix['mean'][i]
        preTrainedCov = preTrainedCov / len(preTrainedMatrix_Class.index)
        preTrainedMean = preTrainedMean / len(preTrainedMatrix_Class.index)
        currentCov = currentValues['cov'].loc[cla]
        currentMean = currentValues['mean'].loc[cla]
        trainedModelL.at[cla, 'cov'] = (1 - 0.8) * preTrainedCov + 0.8 * currentCov
        trainedModelL.at[cla, 'mean'] = (1 - 0.8) * preTrainedMean[0] + 0.8 * currentMean
        trainedModelQ.at[cla, 'cov'] = (1 - 0.9) * preTrainedCov + 0.9 * currentCov
        trainedModelQ.at[cla, 'mean'] = (1 - 0.7) * preTrainedMean[0] + 0.7 * currentMean

        trainedModelL.at[cla, 'class'] = cla + 1
        trainedModelQ.at[cla, 'class'] = cla + 1

    return trainedModelL, trainedModelQ


###### OUR TECHNIQUE

def OurModel(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, step,
             type_DA, k, shotStart):
    t = time.time()
    numSamples = 50
    trainFeatures, trainLabels = subsetTraining(trainFeatures, trainLabels, numSamples, classes)

    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'weight_mean', 'weight_cov', '# gestures'])

    for cla in range(classes):
        adaptiveModel.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
        adaptiveModel.at[cla, 'mean'] = np.zeros((1, allFeatures))[0]

    if type_DA == 'LDA':
        wTarget = mccModelLDA_ALL(trainFeatures, trainLabels, currentValues, classes, step)
    elif type_DA == 'QDA':
        wTarget = mccModelQDA_ALL(trainFeatures, trainLabels, currentValues, classes, step)
    wTargetCov = wTarget.copy()
    wTargetMean = wTarget.copy()

    for cla in range(classes):
        preTrainedMatrix_Class = pd.DataFrame(
            preTrainedDataMatrix[['cov', 'mean']].loc[(preTrainedDataMatrix['class'] == cla + 1)])
        preTrainedMatrix_Class = preTrainedMatrix_Class.reset_index(drop=True)
        currentCov = currentValues['cov'].loc[cla]
        currentMean = currentValues['mean'].loc[cla]
        peopleClass = len(preTrainedMatrix_Class.index)

        wPeopleMean = np.zeros(peopleClass)
        wPeopleCov = np.zeros(peopleClass)

        for i in range(peopleClass):
            personMean = preTrainedMatrix_Class['mean'].loc[i]
            personCov = preTrainedMatrix_Class['cov'].loc[i]
            wPeopleMean[i] = weightPerPersonMean(currentValues, personMean, cla, classes, trainFeatures, trainLabels,
                                                 step, type_DA)
            wPeopleCov[i] = weightPerPersonCov(currentValues, personCov, cla, classes, trainFeatures, trainLabels, step,
                                               type_DA)

        sumWMean = np.sum(wPeopleMean)

        if (sumWMean != 0) and (sumWMean + wTargetMean[cla] != 0):
            wTargetMean[cla] = wTargetMean[cla] / (wTargetMean[cla] + np.mean(wPeopleMean[wPeopleMean != 0]) * k)
            wPeopleMean = (wPeopleMean / sumWMean) * (1 - wTargetMean[cla])

        else:
            wTargetMean[cla] = 1
            wPeopleMean = np.zeros(peopleClass)

        sumWCov = np.sum(wPeopleCov)
        if (sumWCov != 0) and (sumWCov + wTargetCov[cla] != 0):

            wTargetCov[cla] = wTargetCov[cla] / (wTargetCov[cla] + np.mean(wPeopleCov[wPeopleCov != 0]) * k)
            wPeopleCov = (wPeopleCov / sumWCov) * (1 - wTargetCov[cla])

        else:
            wTargetCov[cla] = 1
            wPeopleCov = np.zeros(peopleClass)

        adaptiveModel.at[cla, 'cov'] = np.sum(preTrainedMatrix_Class['cov'] * wPeopleCov) + currentCov * wTargetCov[cla]
        adaptiveModel.at[cla, 'mean'] = np.sum(preTrainedMatrix_Class['mean'] * wPeopleMean) + currentMean * \
                                        wTargetMean[
                                            cla]
        adaptiveModel.at[cla, 'class'] = cla + 1
        adaptiveModel.at[cla, 'weight_mean'] = currentValues.loc[cla, 'weight_mean']
        adaptiveModel.at[cla, 'weight_cov'] = currentValues.loc[cla, 'weight_cov']

    adaptiveModel.at[0, '# gestures'] = shotStart
    trainingTime = time.time() - t
    return adaptiveModel, wTargetMean, wTargetMean.mean(), wTargetCov, wTargetCov.mean(), trainingTime


#### Our Technique with JS divergence

def weight_MSDA_JS_model(currentModel, gestureMean, gestureCov, cla, classes, labeledGesturesFeatures,
                         labeledGesturesLabels, type_DA, matrix_D):
    matrix_D_mean = matrix_D.copy()
    matrix_D_cov = matrix_D.copy()
    # if type_DA == 'LDA':
    #     LDACov = DA_Classifiers.LDA_Cov(currentModel, classes)
    #     matrix_D_mean[cla, :] = SemiSupervised.rowDivergences(labeledGesturesFeatures, labeledGesturesLabels,
    #                                                           gestureMean, LDACov, classes)
    #     auxCurrentModel = currentModel.copy()
    #     auxCurrentModel['cov'].at[cla] = gestureCov
    #     auxLDACov = DA_Classifiers.LDA_Cov(auxCurrentModel, classes)
    #     matrix_D_cov[cla, :] = SemiSupervised.rowDivergences(labeledGesturesFeatures, labeledGesturesLabels,
    #                                                          currentModel['mean'].loc[cla], auxLDACov, classes)
    # elif type_DA == 'QDA':
    matrix_D_mean[cla, :] = SemiSupervised.rowDivergences(labeledGesturesFeatures, labeledGesturesLabels, gestureMean,
                                                          currentModel['cov'].loc[cla], classes)
    matrix_D_cov[cla, :] = SemiSupervised.rowDivergences(labeledGesturesFeatures, labeledGesturesLabels,
                                                         currentModel['mean'].loc[cla], gestureCov, classes)
    if np.argmin(matrix_D_mean[:, cla]) != cla or np.argmin(matrix_D_mean[cla, :]) != cla:
        weightMean = 0
    else:
        JS_2 = 0
        for i in range(classes):
            if i != cla:
                JS_2 += matrix_D_mean[cla, i] / (matrix_D_mean[cla, i] + matrix_D_mean[i, cla])

        # JS_1 = second_smallest(matrix_D_mean[:, cla])
        # JS_2 = second_smallest(matrix_D_mean[cla, :])
        # weightMean = JS_2 / (JS_1 + JS_2)

        weightMean = JS_2 / (classes - 1)

    if np.argmin(matrix_D_cov[:, cla]) != cla or np.argmin(matrix_D_cov[cla, :]) != cla:
        weightCov = 0
    else:
        JS_2 = 0
        for i in range(classes):
            if i != cla:
                JS_2 += matrix_D_cov[cla, i] / (matrix_D_cov[cla, i] + matrix_D_cov[i, cla])
        # JS_1 = second_smallest(matrix_D_cov[:, cla])
        # JS_2 = second_smallest(matrix_D_cov[cla, :])
        # weightCov = JS_2 / (JS_1 + JS_2)

        weightCov = JS_2 / (classes - 1)
    # return SemiSupervised.mcc_from_matrixDivergences(matrix_D_mean, classes, cla), \
    #        SemiSupervised.mcc_from_matrixDivergences(matrix_D_cov, classes, cla)
    return weightMean, weightCov


def second_smallest(numbers):
    m1, m2 = float('inf'), float('inf')
    for x in numbers:
        if x <= m1:
            m1, m2 = x, m1
        elif x < m2:
            m2 = x
    return m2


def weight_MSDA_JS_matrixDivergences(classes, matrix_D):
    wGesture = []
    for cla in range(classes):
        wGesture.append(SemiSupervised.mcc_from_matrixDivergences(matrix_D, classes, cla))
    return np.array(wGesture)


def OurModel2(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, step,
              type_DA, k):
    t = time.time()
    # numSamples = 50
    # trainFeatures, trainLabels = subsetTraining(trainFeatures, trainLabels, numSamples, classes)

    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'weight_mean', 'weight_cov'])

    for cla in range(classes):
        adaptiveModel.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
        adaptiveModel.at[cla, 'mean'] = np.zeros((1, allFeatures))[0]

    matrix_D = SemiSupervised.matrixDivergences(trainFeatures, trainLabels, currentValues, classes, type_DA)
    wTarget = weight_MSDA_JS_matrixDivergences(classes, matrix_D)

    # if type_DA == 'LDA':
    #     wTarget = mccModelLDA_ALL(trainFeatures, trainLabels, currentValues, classes, step)
    # elif type_DA == 'QDA':
    #     wTarget = mccModelQDA_ALL(trainFeatures, trainLabels, currentValues, classes, step)

    wTargetCov = wTarget.copy()
    wTargetMean = wTarget.copy()
    for cla in range(classes):
        preTrainedMatrix_Class = pd.DataFrame(
            preTrainedDataMatrix[['cov', 'mean']].loc[(preTrainedDataMatrix['class'] == cla + 1)])
        preTrainedMatrix_Class = preTrainedMatrix_Class.reset_index(drop=True)
        currentCov = currentValues['cov'].loc[cla]
        currentMean = currentValues['mean'].loc[cla]
        peopleClass = len(preTrainedMatrix_Class.index)

        wPeopleMean = np.zeros(peopleClass)
        wPeopleCov = np.zeros(peopleClass)

        for i in range(peopleClass):
            personMean = preTrainedMatrix_Class['mean'].loc[i]
            personCov = preTrainedMatrix_Class['cov'].loc[i]
            # wPeopleMean[i] = weightPerPersonMean(currentValues, personMean, cla, classes, trainFeatures, trainLabels,
            #                                      step, type_DA)
            # wPeopleCov[i] = weightPerPersonCov(currentValues, personCov, cla, classes, trainFeatures, trainLabels, step,
            #                                    type_DA)

            wPeopleMean[i], wPeopleCov[i] = weight_MSDA_JS_model(currentValues, personMean, personCov, cla, classes,
                                                                 trainFeatures, trainLabels, type_DA, matrix_D)

        sumWMean = np.sum(wPeopleMean)
        sumWCov = np.sum(wPeopleCov)
        if sumWMean != 0:
            wPeopleMean = (wPeopleMean / sumWMean)
            peopleMean = np.sum(preTrainedMatrix_Class['mean'] * wPeopleMean)
        else:
            print('no people mean', cla)
            peopleMean = currentMean

        if sumWCov != 0:
            wPeopleCov = (wPeopleCov / sumWCov)
            peopleCov = np.sum(preTrainedMatrix_Class['cov'] * wPeopleCov)
        else:
            print('no people cov', cla)
            peopleCov = currentCov

        wPeopleMeanTotal, wPeopleCovTotal = weight_MSDA_JS_model(currentValues, peopleMean, peopleCov, cla, classes,
                                                                 trainFeatures, trainLabels, type_DA, matrix_D)

        wTargetMean[cla] = 1 - wPeopleMeanTotal
        wTargetCov[cla] = 1 - wPeopleCovTotal
        print('wTargetMean', cla, ' ', wTargetMean[cla])
        print('wTargetCov', cla, ' ', wTargetCov[cla])

        adaptiveModel.at[cla, 'cov'] = peopleCov * wPeopleCovTotal + currentCov * wTargetCov[cla]
        adaptiveModel.at[cla, 'mean'] = peopleMean * wPeopleMeanTotal + currentMean * wTargetMean[cla]
        adaptiveModel.at[cla, 'class'] = cla + 1
        adaptiveModel.at[cla, 'weight_mean'] = currentValues.loc[cla, 'weight_mean']
        adaptiveModel.at[cla, 'weight_cov'] = currentValues.loc[cla, 'weight_cov']

        # if (sumWMean != 0) and (sumWMean + wTargetMean[cla] != 0):
        #     wTargetMean[cla] = wTargetMean[cla] / (wTargetMean[cla] + np.mean(wPeopleMean[wPeopleMean != 0]) * k)
        #     wPeopleMean = (wPeopleMean / sumWMean) * (1 - wTargetMean[cla])
        #
        # else:
        #     wTargetMean[cla] = 1
        #     wPeopleMean = np.zeros(peopleClass)
        #
        # sumWCov = np.sum(wPeopleCov)
        # if (sumWCov != 0) and (sumWCov + wTargetCov[cla] != 0):
        #
        #     wTargetCov[cla] = wTargetCov[cla] / (wTargetCov[cla] + np.mean(wPeopleCov[wPeopleCov != 0]) * k)
        #     wPeopleCov = (wPeopleCov / sumWCov) * (1 - wTargetCov[cla])
        #
        # else:
        #     wTargetCov[cla] = 1
        #     wPeopleCov = np.zeros(peopleClass)
        #
        # adaptiveModel.at[cla, 'cov'] = np.sum(preTrainedMatrix_Class['cov'] * wPeopleCov) + currentCov * wTargetCov[cla]
        # adaptiveModel.at[cla, 'mean'] = np.sum(preTrainedMatrix_Class['mean'] * wPeopleMean) + currentMean * \
        #                                 wTargetMean[cla]
        # adaptiveModel.at[cla, 'class'] = cla + 1
        # adaptiveModel.at[cla, 'weight_mean'] = currentValues.loc[cla, 'weight_mean']
        # adaptiveModel.at[cla, 'weight_cov'] = currentValues.loc[cla, 'weight_cov']

    trainingTime = time.time() - t
    return adaptiveModel, wTargetMean, wTargetMean.mean(), wTargetCov, wTargetCov.mean(), trainingTime


# %% Weight Calculation
def weightPerPersonMean(currentValues, personMean, currentClass, classes, trainFeatures, trainLabels, step,
                        type_DA):
    personValues = currentValues.copy()
    personValues['mean'].at[currentClass] = personMean
    if type_DA == 'LDA':
        weight = mccModelLDA(trainFeatures, trainLabels, personValues, classes, currentClass, step)
    elif type_DA == 'QDA':
        weight = mccModelQDA(trainFeatures, trainLabels, personValues, classes, currentClass, step)

    return weight


def weightPerPersonCov(currentValues, personCov, currentClass, classes, trainFeatures, trainLabels, step,
                       type_DA):
    personValues = currentValues.copy()
    personValues['cov'].at[currentClass] = personCov
    if type_DA == 'LDA':
        weight = mccModelLDA(trainFeatures, trainLabels, personValues, classes, currentClass, step)
    elif type_DA == 'QDA':
        weight = mccModelQDA(trainFeatures, trainLabels, personValues, classes, currentClass, step)
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


def mccModelLDA(testFeatures, testLabels, model, classes, currentClass, step):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    currentClass = currentClass + 1
    LDACov = DA_Classifiers.LDA_Cov(model, classes)
    for i in range(0, np.size(testLabels), step):
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


def mccModelQDA(testFeatures, testLabels, model, classes, currentClass, step):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    currentClass = currentClass + 1
    for i in range(0, np.size(testLabels), step):
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


def mccModelLDA_ALL(testFeatures, testLabels, model, classes, step):
    TP = np.zeros([classes])
    TN = np.zeros([classes])
    FP = np.zeros([classes])
    FN = np.zeros([classes])

    LDACov = DA_Classifiers.LDA_Cov(model, classes)

    for i in range(0, np.size(testLabels), step):
        currentPredictor = DA_Classifiers.predictedModelLDA(testFeatures[i, :], model, classes, LDACov)

        if currentPredictor == testLabels[i]:
            TP[int(testLabels[i] - 1)] += 1
            for j in range(classes):
                if j != int(testLabels[i] - 1):
                    TN[j] += 1
        else:
            FN[int(testLabels[i] - 1)] += 1
            for j in range(classes):
                if j != int(testLabels[i] - 1):
                    FP[j] += 1
    return mcc(TP, TN, FP, FN)


def mccModelQDA_ALL(testFeatures, testLabels, model, classes, step):
    TP = np.zeros([classes])
    TN = np.zeros([classes])
    FP = np.zeros([classes])
    FN = np.zeros([classes])

    for i in range(0, np.size(testLabels), step):
        currentPredictor = DA_Classifiers.predictedModelQDA(testFeatures[i, :], model, classes)

        if currentPredictor == testLabels[i]:
            TP[int(testLabels[i] - 1)] += 1
            for j in range(classes):
                if j != int(testLabels[i] - 1):
                    TN[j] += 1
        else:
            FN[int(testLabels[i] - 1)] += 1
            for j in range(classes):
                if j != int(testLabels[i] - 1):
                    FP[j] += 1

    return mcc(TP, TN, FP, FN)


def subsetTraining(trainFeatures, trainLabels, numSamples, classes):
    idx = []
    for cla in range(classes):
        aux = np.where(trainLabels == cla + 1)[0]

        if len(aux) > numSamples:
            modNumber = np.ceil(len(aux) / numSamples)
            idxAux = []
            [idxAux.append(a) for a in aux if a % modNumber == 1 and len(idxAux) < numSamples]
            if len(idxAux) < numSamples:
                [idxAux.append(a) for a in aux if a % modNumber == 2 and len(idxAux) < numSamples]
            idx.extend(idxAux)
        else:
            idx.extend(list(aux))
    return trainFeatures[idx], trainLabels[idx]

# def KLdivergence(mean0, mean1, k, cov0, cov1):
#     exp1 = np.trace(np.dot(np.linalg.inv(cov1), cov0))
#     exp2 = np.dot(np.dot((mean1 - mean0).T, np.linalg.inv(cov1)), (mean1 - mean0))
#     exp3 = np.log(np.linalg.det(cov1) / np.linalg.det(cov0))
#     return 0.5 * (exp1 + exp2 - k + exp3)
#
#
# def JSDdivergence(mean0, mean1, k, cov0, cov1):
#     meanM = (mean0 + mean1) / 2
#     covM = (cov0 + cov1) / 2
#     js = KLdivergence(mean0, meanM, k, cov0, covM) + KLdivergence(mean1, meanM, k, cov1, covM)
#     # js /= np.log(2)
#     return np.sqrt(js / 2)


# %% Unsupervised Methods
#
# def OurModelUnsupervisedAllProb(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels,
#                                 oneShotModel, type_DA, shotStart):
#     t = time.time()
#
#     trainFeatures, trainLabels = reduce_dataset(trainFeatures, trainLabels, classes)
#
#     type_DAWeights = 'QDA'
#     peopleClass = len(preTrainedDataMatrix.index)
#     # if typeDatabase == 'Nina5':
#     #     preTrainedDataMatrix2 = pd.DataFrame(columns=['cov', 'mean', 'class', 'prob', 'samples'])
#     #     i2 = 0
#     #     idxSetBase = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 1])
#     #
#     #     for j in range(int(peopleClass / classes)):
#     #         idxSet = classes * j + idxSetBase
#     #
#     #         for i in idxSet:
#     #             preTrainedDataMatrix2.at[i2] = preTrainedDataMatrix.loc[i]
#     #             i2 += 1
#     #
#     #     preTrainedDataMatrix = preTrainedDataMatrix2.copy()
#
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class'])
#
#     for cla in range(classes):
#         adaptiveModel.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
#         adaptiveModel.at[cla, 'mean'] = np.zeros((1, allFeatures))[0]
#
#     wFewCov = np.ones(classes) * shotStart
#     wFewMean = np.ones(classes) * shotStart
#
#     # print('allpeople', preTrainedDataMatrix[['class', 'prob']])
#
#     wPeopleMean = np.zeros((peopleClass, classes))
#     wPeopleCov = np.zeros((peopleClass, classes))
#
#     for cla in range(classes):
#         # wTargetMean[cla] = weightPerPersonMean(oneShotModel, currentValues['mean'].loc[cla], cla, classes,
#         #                                        trainFeatures, trainLabels, step=1, type_DA=type_DAWeights)
#         # wTargetCov[cla] = weightPerPersonCov(oneShotModel, currentValues['cov'].loc[cla], cla, classes,
#         #                                      trainFeatures, trainLabels, step=1, type_DA=type_DAWeights)
#
#         for person in range(peopleClass):
#             # wPeopleMean[i] = JSDdivergence(currentMean, personMean, 8, currentCov, personCov)
#
#             if preTrainedDataMatrix['prob'].loc[person][cla] != 0:
#                 personMean = preTrainedDataMatrix['mean'].loc[person]
#                 personCov = preTrainedDataMatrix['cov'].loc[person]
#                 wPeopleMean[person, cla] = weightPerPersonMean(
#                     oneShotModel, personMean, cla, classes, trainFeatures, trainLabels, step=1,
#                     type_DA=type_DAWeights) * \
#                                            preTrainedDataMatrix['prob'].loc[person][cla]
#                 if type_DA == 'QDA':
#                     wPeopleCov[person, cla] = weightPerPersonCov(
#                         oneShotModel, personCov, cla, classes, trainFeatures, trainLabels, step=1,
#                         type_DA=type_DAWeights) * preTrainedDataMatrix['prob'].loc[person][cla]
#
#     sumWMean = np.sum(wPeopleMean, axis=0) + wFewMean
#     sumWCov = np.sum(wPeopleCov, axis=0) + wFewCov
#     wFewMean /= sumWMean
#     wFewCov /= sumWCov
#     wPeopleMean /= sumWMean
#     wPeopleCov /= sumWCov
#     wFewMean = np.nan_to_num(wFewMean, nan=1)
#     wFewCov = np.nan_to_num(wFewCov, nan=1)
#     wPeopleMean = np.nan_to_num(wPeopleMean)
#     wPeopleCov = np.nan_to_num(wPeopleCov)
#
#     means = np.resize(preTrainedDataMatrix['mean'], (classes, len(preTrainedDataMatrix['mean']))).T * wPeopleMean
#     covs = np.resize(preTrainedDataMatrix['cov'], (classes, len(preTrainedDataMatrix['cov']))).T * wPeopleCov
#     for cla in range(classes):
#         adaptiveModel.at[cla, 'class'] = cla + 1
#         adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + oneShotModel['mean'].loc[cla] * wFewMean[cla]
#         if type_DA == 'LDA':
#             adaptiveModel.at[cla, 'cov'] = (np.sum(preTrainedDataMatrix['cov']) + oneShotModel['cov'].loc[cla]) / (
#                     peopleClass + 1)
#         elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + oneShotModel['cov'].loc[cla] * wFewCov[cla]
#
#     return adaptiveModel, time.time() - t
#
#
# def reduce_dataset(x, y, classes):
#     reduce_X = []
#     reduce_Y = []
#     for cl in range(1, classes + 1):
#         reduce_X.append(np.mean(x[y == cl], axis=0))
#         reduce_Y.append(cl)
#     return np.array(reduce_X), np.array(reduce_Y)
#
#
# def OurModelUnsupervisedAllProb_shot(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures,
#                                      trainLabels, oneShotModel, type_DA, shotStart):
#     t = time.time()
#     trainFeatures, trainLabels = reduce_dataset(trainFeatures, trainLabels, classes)
#
#     type_DAWeights = 'QDA'
#     peopleClass = len(preTrainedDataMatrix.index)
#
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'weight_mean', 'weight_cov'])
#
#     for cla in range(classes):
#         adaptiveModel.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
#         adaptiveModel.at[cla, 'mean'] = np.zeros((1, allFeatures))[0]
#
#     wCurrentCov = np.zeros(classes)
#     wCurrentMean = np.zeros(classes)
#
#     # print('allpeople', preTrainedDataMatrix[['class', 'prob']])
#
#     wPeopleMean = np.zeros((peopleClass, classes))
#     wPeopleCov = np.zeros((peopleClass, classes))
#
#     for cla in range(classes):
#
#         for person in range(peopleClass):
#
#             if preTrainedDataMatrix['prob'].loc[person][cla] != 0:
#                 personMean = preTrainedDataMatrix['mean'].loc[person]
#                 personCov = preTrainedDataMatrix['cov'].loc[person]
#                 wPeopleMean[person, cla] = weightPerPersonMean(
#                     oneShotModel, personMean, cla, classes, trainFeatures, trainLabels, step=1,
#                     type_DA=type_DAWeights) * \
#                                            preTrainedDataMatrix['prob'].loc[person][cla]
#                 if type_DA == 'QDA':
#                     wPeopleCov[person, cla] = weightPerPersonCov(
#                         oneShotModel, personCov, cla, classes, trainFeatures, trainLabels, step=1,
#                         type_DA=type_DAWeights) * preTrainedDataMatrix['prob'].loc[person][cla]
#
#         wCurrentCov[cla] = currentValues['weight_cov'].loc[cla]
#         wCurrentMean[cla] = currentValues['weight_mean'].loc[cla]
#         adaptiveModel['weight_cov'].loc[cla] = currentValues['weight_cov'].loc[cla] + wPeopleCov[:, cla].sum()
#         adaptiveModel['weight_mean'].loc[cla] = currentValues['weight_mean'].loc[cla] + wPeopleMean[:, cla].sum()
#
#     sumWMean = np.sum(wPeopleMean, axis=0) + wCurrentMean
#     sumWCov = np.sum(wPeopleCov, axis=0) + wCurrentCov
#     wCurrentMean /= sumWMean
#     wCurrentCov /= sumWCov
#     wPeopleMean /= sumWMean
#     wPeopleCov /= sumWCov
#     wCurrentMean = np.nan_to_num(wCurrentMean, nan=1)
#     wCurrentCov = np.nan_to_num(wCurrentCov, nan=1)
#     wPeopleMean = np.nan_to_num(wPeopleMean)
#     wPeopleCov = np.nan_to_num(wPeopleCov)
#
#     means = np.resize(preTrainedDataMatrix['mean'], (classes, len(preTrainedDataMatrix['mean']))).T * wPeopleMean
#     covs = np.resize(preTrainedDataMatrix['cov'], (classes, len(preTrainedDataMatrix['cov']))).T * wPeopleCov
#     for cla in range(classes):
#         adaptiveModel.at[cla, 'class'] = cla + 1
#         adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + currentValues['mean'].loc[cla] * wCurrentMean[cla]
#         if type_DA == 'LDA':
#             adaptiveModel.at[cla, 'cov'] = (np.sum(preTrainedDataMatrix['cov']) + currentValues['cov'].loc[cla]) / (
#                     peopleClass + 1)
#         else:
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + currentValues['cov'].loc[cla] * wCurrentCov[cla]
#
#     return adaptiveModel, time.time() - t
#
#
# def OurModelUnsupervisedAllProb_noRQ1(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures,
#                                       trainLabels, oneShotModel, type_DA, shotStart):
#     t = time.time()
#     peopleClass = len(preTrainedDataMatrix.index)
#
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class'])
#
#     for cla in range(classes):
#         adaptiveModel.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
#         adaptiveModel.at[cla, 'mean'] = np.zeros((1, allFeatures))[0]
#
#     wFewCov = np.ones(classes) * shotStart
#     wFewMean = np.ones(classes) * shotStart
#
#     # print('allpeople', preTrainedDataMatrix[['class', 'prob']])
#
#     wPeopleMean = np.zeros((peopleClass, classes))
#     wPeopleCov = np.zeros((peopleClass, classes))
#
#     for cla in range(classes):
#
#         for person in range(peopleClass):
#
#             if preTrainedDataMatrix['prob'].loc[person][cla] != 0:
#                 wPeopleMean[person, cla] = preTrainedDataMatrix['prob'].loc[person][cla]
#                 wPeopleCov[person, cla] = preTrainedDataMatrix['prob'].loc[person][cla]
#     sumWMean = np.sum(wPeopleMean, axis=0) + wFewMean
#     sumWCov = np.sum(wPeopleCov, axis=0) + wFewCov
#     wFewMean /= sumWMean
#     wFewCov /= sumWCov
#     wPeopleMean /= sumWMean
#     wPeopleCov /= sumWCov
#     wFewMean = np.nan_to_num(wFewMean, nan=1)
#     wFewCov = np.nan_to_num(wFewCov, nan=1)
#     wPeopleMean = np.nan_to_num(wPeopleMean)
#     wPeopleCov = np.nan_to_num(wPeopleCov)
#
#     means = np.resize(preTrainedDataMatrix['mean'], (classes, len(preTrainedDataMatrix['mean']))).T * wPeopleMean
#     covs = np.resize(preTrainedDataMatrix['cov'], (classes, len(preTrainedDataMatrix['cov']))).T * wPeopleCov
#     for cla in range(classes):
#         adaptiveModel.at[cla, 'class'] = cla + 1
#         adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + oneShotModel['mean'].loc[cla] * wFewMean[cla]
#         if type_DA == 'LDA':
#             adaptiveModel.at[cla, 'cov'] = (np.sum(preTrainedDataMatrix['cov']) + oneShotModel['cov'].loc[cla]) / (
#                     peopleClass + 1)
#         else:
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + oneShotModel['cov'].loc[cla] * wFewCov[cla]
#
#     return adaptiveModel, time.time() - t
#
#
# def OurModelUnsupervisedAllProb_noRQ1_shot(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures,
#                                            trainLabels, oneShotModel, type_DA, shotStart):
#     t = time.time()
#     peopleClass = len(preTrainedDataMatrix.index)
#
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'weight_mean', 'weight_cov'])
#
#     for cla in range(classes):
#         adaptiveModel.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
#         adaptiveModel.at[cla, 'mean'] = np.zeros((1, allFeatures))[0]
#
#     wCurrentCov = np.zeros(classes)
#     wCurrentMean = np.zeros(classes)
#
#     # print('allpeople', preTrainedDataMatrix[['class', 'prob']])
#
#     wPeopleMean = np.zeros((peopleClass, classes))
#     wPeopleCov = np.zeros((peopleClass, classes))
#
#     for cla in range(classes):
#
#         for person in range(peopleClass):
#
#             if preTrainedDataMatrix['prob'].loc[person][cla] != 0:
#                 wPeopleMean[person, cla] = preTrainedDataMatrix['prob'].loc[person][cla]
#                 wPeopleCov[person, cla] = preTrainedDataMatrix['prob'].loc[person][cla]
#
#         wCurrentCov[cla] = currentValues['weight_cov'].loc[cla]
#         wCurrentMean[cla] = currentValues['weight_mean'].loc[cla]
#         adaptiveModel['weight_cov'].loc[cla] = currentValues['weight_cov'].loc[cla] + wPeopleCov[:, cla].sum()
#         adaptiveModel['weight_mean'].loc[cla] = currentValues['weight_mean'].loc[cla] + wPeopleMean[:, cla].sum()
#
#     sumWMean = np.sum(wPeopleMean, axis=0) + wCurrentMean
#     sumWCov = np.sum(wPeopleCov, axis=0) + wCurrentCov
#     wCurrentMean /= sumWMean
#     wCurrentCov /= sumWCov
#     wPeopleMean /= sumWMean
#     wPeopleCov /= sumWCov
#     wCurrentMean = np.nan_to_num(wCurrentMean, nan=1)
#     wCurrentCov = np.nan_to_num(wCurrentCov, nan=1)
#     wPeopleMean = np.nan_to_num(wPeopleMean)
#     wPeopleCov = np.nan_to_num(wPeopleCov)
#
#     means = np.resize(preTrainedDataMatrix['mean'], (classes, len(preTrainedDataMatrix['mean']))).T * wPeopleMean
#     covs = np.resize(preTrainedDataMatrix['cov'], (classes, len(preTrainedDataMatrix['cov']))).T * wPeopleCov
#     for cla in range(classes):
#         adaptiveModel.at[cla, 'class'] = cla + 1
#         adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + currentValues['mean'].loc[cla] * wCurrentMean[cla]
#         if type_DA == 'LDA':
#             adaptiveModel.at[cla, 'cov'] = (np.sum(preTrainedDataMatrix['cov']) + currentValues['cov'].loc[cla]) / (
#                     peopleClass + 1)
#         else:
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + currentValues['cov'].loc[cla] * wCurrentCov[cla]
#
#     return adaptiveModel, time.time() - t
#
# # def OurModelUnsupervisedAllProb_onlyRQ1(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures,
# #                                         trainLabels,
# #                                         oneShotModel, type_DA):
# #     type_DAWeights = 'QDA'
# #     peopleClass = len(preTrainedDataMatrix.index)
# #     # if typeDatabase == 'Nina5':
# #     #     preTrainedDataMatrix2 = pd.DataFrame(columns=['cov', 'mean', 'class', 'prob', 'samples'])
# #     #     i2 = 0
# #     #     idxSetBase = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 1])
# #     #
# #     #     for j in range(int(peopleClass / classes)):
# #     #         idxSet = classes * j + idxSetBase
# #     #
# #     #         for i in idxSet:
# #     #             preTrainedDataMatrix2.at[i2] = preTrainedDataMatrix.loc[i]
# #     #             i2 += 1
# #     #
# #     #     preTrainedDataMatrix = preTrainedDataMatrix2.copy()
# #
# #     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class'])
# #
# #     for cla in range(classes):
# #         adaptiveModel.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
# #         adaptiveModel.at[cla, 'mean'] = np.zeros((1, allFeatures))[0]
# #
# #     wTargetCov = np.zeros(classes)
# #     wTargetMean = np.zeros(classes)
# #
# #     wPeopleMean = np.zeros((peopleClass, classes))
# #     wPeopleCov = np.zeros((peopleClass, classes))
# #
# #     for cla in range(classes):
# #         wTargetMean[cla] = weightPerPersonMean(oneShotModel, currentValues['mean'].loc[cla], cla, classes,
# #                                                trainFeatures, trainLabels, step=1, type_DA=type_DAWeights)
# #         wTargetCov[cla] = weightPerPersonCov(oneShotModel, currentValues['cov'].loc[cla], cla, classes,
# #                                              trainFeatures, trainLabels, step=1, type_DA=type_DAWeights)
# #
# #         for person in range(peopleClass):
# #             # wPeopleMean[i] = JSDdivergence(currentMean, personMean, 8, currentCov, personCov)
# #
# #             # if preTrainedDataMatrix['prob'].loc[person][cla] != 0:
# #             personMean = preTrainedDataMatrix['mean'].loc[person]
# #             personCov = preTrainedDataMatrix['cov'].loc[person]
# #             wPeopleMean[person, cla] = weightPerPersonMean(
# #                 oneShotModel, personMean, cla, classes, trainFeatures, trainLabels, step=1, type_DA=type_DAWeights)
# #             wPeopleCov[person, cla] = weightPerPersonCov(
# #                 oneShotModel, personCov, cla, classes, trainFeatures, trainLabels, step=1, type_DA=type_DAWeights)
# #     sumWMean = np.sum(wPeopleMean, axis=0) + wTargetMean
# #     sumWCov = np.sum(wPeopleCov, axis=0) + wTargetCov
# #     wTargetMean /= sumWMean
# #     wTargetCov /= sumWCov
# #     wPeopleMean /= sumWMean
# #     wPeopleCov /= sumWCov
# #     wTargetMean = np.nan_to_num(wTargetMean, nan=1)
# #     wTargetCov = np.nan_to_num(wTargetCov, nan=1)
# #     wPeopleMean = np.nan_to_num(wPeopleMean)
# #     wPeopleCov = np.nan_to_num(wPeopleCov)
# #     # print('mean weights', wPeopleMean)
# #     # print(wTargetMean)
# #     # print('cov weights', wPeopleCov)
# #     # print(wTargetCov)
# #     means = np.resize(preTrainedDataMatrix['mean'], (classes, len(preTrainedDataMatrix['mean']))).T * wPeopleMean
# #     covs = np.resize(preTrainedDataMatrix['cov'], (classes, len(preTrainedDataMatrix['cov']))).T * wPeopleCov
# #     for cla in range(classes):
# #         adaptiveModel.at[cla, 'class'] = cla + 1
# #         adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + currentValues['mean'].loc[cla] * wTargetMean[cla]
# #         if type_DA == 'LDA':
# #             adaptiveModel.at[cla, 'cov'] = (np.sum(preTrainedDataMatrix['cov']) + currentValues['cov'].loc[cla]) / (
# #                     peopleClass + 1)
# #         else:
# #             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + currentValues['cov'].loc[cla] * wTargetCov[cla]
# #
# #     return adaptiveModel, wTargetMean, wTargetMean.mean(), wTargetCov, wTargetCov.mean(), 0
