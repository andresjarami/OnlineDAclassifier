import math
import time

import numpy as np

import scipy.stats as stats
from sklearn.metrics import confusion_matrix


import pandas as pd
from scipy.spatial import distance

import DA_Classifiers as DA_Classifiers


# from numpy.random import default_rng
#
# rng = default_rng()


# LDA Calssifier
def LDA_Discriminant(x, covariance, mean):
    det = np.linalg.det(covariance)
    if det > 0:
        invCov = np.linalg.inv(covariance)
        discriminant = np.dot(np.dot(x, invCov), mean) - 0.5 * np.dot(np.dot(mean, invCov), mean)
    else:
        discriminant = float('NaN')
    return discriminant


def LDA_Discriminant_pseudo(x, covariance, mean):
    invCov = np.linalg.pinv(covariance)
    return np.dot(np.dot(x, invCov), mean) - 0.5 * np.dot(np.dot(mean, invCov), mean)


def LDA_Cov(trainedModel, classes):
    LDACov = trainedModel['cov'].sum() / classes
    return LDACov


def predictedModelLDA(sample, model, classes, LDACov):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = LDA_Discriminant(sample, LDACov, model['mean'].loc[cl])
        if math.isnan(d[cl]):
            return predictedModelLDA_pseudo(sample, model, classes, LDACov)
    return np.argmax(d) + 1


def predictedModelLDA_pseudo(sample, model, classes, LDACov):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = LDA_Discriminant_pseudo(sample, LDACov, model['mean'].loc[cl])
    return np.argmax(d) + 1


def accuracyModelLDA(testFeatures, testLabels, model, classes):
    t = 0
    true = 0
    count = 0
    LDACov = LDA_Cov(model, classes)
    for i in range(0, np.size(testLabels)):
        auxt = time.time()
        currentPredictor = predictedModelLDA(testFeatures[i, :], model, classes, LDACov)
        t += (time.time() - auxt)
        if currentPredictor == testLabels[i]:
            true += 1
            count += 1
        else:
            count += 1

    return true / count, t / np.size(testLabels)


# QDA Classifier
def QDA_Discriminant(x, covariance, mean):
    det = np.linalg.det(covariance)
    if det > 0:
        discriminant = -.5 * np.log(det) - .5 * np.dot(np.dot((x - mean), np.linalg.inv(covariance)), (x - mean).T)
    else:
        discriminant = float('NaN')
    return discriminant


def QDA_Discriminant_pseudo(x, covariance, mean):
    return -.5 * np.dot(np.dot((x - mean), np.linalg.pinv(covariance)), (x - mean).T)


def predictedModelQDA(sample, model, classes):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = QDA_Discriminant(sample, model['cov'].loc[cl], model['mean'].loc[cl])
        if math.isnan(d[cl]):
            return predictedModelQDA_pseudo(sample, model, classes)
    return np.argmax(d) + 1


def predictedModelQDA_pseudo(sample, model, classes):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = QDA_Discriminant_pseudo(sample, model['cov'].loc[cl], model['mean'].loc[cl])
    return np.argmax(d) + 1


def accuracyModelQDA(testFeatures, testLabels, model, classes):
    t = 0
    true = 0
    count = 0
    for i in range(0, np.size(testLabels)):
        auxt = time.time()
        actualPredictor = predictedModelQDA(testFeatures[i, :], model, classes)
        t += (time.time() - auxt)
        if actualPredictor == testLabels[i]:
            true += 1
            count += 1
        else:
            count += 1
    return true / count, t / np.size(testLabels)


def scoreModelLDA_ClassificationUnsupervised(testFeatures, model, classes, testLabels, testRepetitions):
    LDACov = LDA_Cov(model, classes)
    auxLabel = testLabels[0]
    auxRep = testRepetitions[0]

    actualPredictor = np.zeros(classes)

    actualFeatures = []
    idx = 0
    preTrainedDataMatrix = pd.DataFrame(columns=['mean', 'cov', 'class', 'prob', 'samples'])

    for i in range(np.size(testRepetitions)):
        if auxLabel != testLabels[i] or auxRep != testRepetitions[i] or i == np.size(testRepetitions) - 1:
            auxLabel = testLabels[i]
            auxRep = testRepetitions[i]

            actualFeatures = np.array(actualFeatures)
            preTrainedDataMatrix.at[idx, 'mean'] = np.mean(actualFeatures, axis=0)
            preTrainedDataMatrix.at[idx, 'cov'] = np.cov(actualFeatures, rowvar=False)
            preTrainedDataMatrix.at[idx, 'class'] = actualPredictor.argmax() + 1
            preTrainedDataMatrix.at[idx, 'prob'] = actualPredictor / actualPredictor.sum()
            preTrainedDataMatrix.at[idx, 'samples'] = len(actualFeatures)
            idx += 1
            actualPredictor = np.zeros(classes)
            actualFeatures = []

        actualPredictor[predictedModelLDA(testFeatures[i, :], model, classes, LDACov) - 1] += 1
        actualFeatures.append(testFeatures[i, :])

    return preTrainedDataMatrix


def scoreModelQDA_ClassificationUnsupervised(testFeatures, model, classes, testLabels, testRepetitions):
    auxLabel = testLabels[0]
    auxRep = testRepetitions[0]

    actualPredictor = np.zeros(classes)
    actualFeatures = []
    idx = 0
    preTrainedDataMatrix = pd.DataFrame(columns=['mean', 'cov', 'class', 'prob', 'samples'])

    for i in range(np.size(testRepetitions)):
        if auxLabel != testLabels[i] or auxRep != testRepetitions[i] or i == np.size(testRepetitions) - 1:
            auxLabel = testLabels[i]
            auxRep = testRepetitions[i]

            actualFeatures = np.array(actualFeatures)
            preTrainedDataMatrix.at[idx, 'mean'] = np.mean(actualFeatures, axis=0)
            preTrainedDataMatrix.at[idx, 'cov'] = np.cov(actualFeatures, rowvar=False)
            preTrainedDataMatrix.at[idx, 'class'] = actualPredictor.argmax() + 1
            preTrainedDataMatrix.at[idx, 'prob'] = actualPredictor / actualPredictor.sum()
            preTrainedDataMatrix.at[idx, 'samples'] = len(actualFeatures)
            idx += 1
            actualPredictor = np.zeros(classes)
            actualFeatures = []

        actualPredictor[predictedModelQDA(testFeatures[i, :], model, classes) - 1] += 1
        actualFeatures.append(testFeatures[i, :])
    return preTrainedDataMatrix


##### GA
def accuracyModelLDAconfusionMatrix(testFeatures, testLabels, model, classes):
    t = 0
    true = 0
    prob = 0
    LDACov = LDA_Cov(model, classes)
    currentPredictor = np.zeros(np.size(testLabels))
    for i in range(np.size(testLabels)):
        auxt = time.time()
        currentPredictor[i], currentProb = predictedModelLDAProb(testFeatures[i, :], model, classes, LDACov)
        t += (time.time() - auxt)
        if currentPredictor[i] == testLabels[i]:
            true += 1
            prob += currentProb
    cm = np.array(confusion_matrix(testLabels, currentPredictor))
    pre = np.zeros(classes)
    recall = np.zeros(classes)
    w = np.zeros(classes)
    for cla in range(classes):
        pre[cla] = cm[cla, cla] / cm[cla, :].sum()
        recall[cla] = cm[cla, cla] / cm[:, cla].sum()
        w[cla] = pre[cla] * recall[cla]

    print(cm, pre, recall)
    return true / np.size(testLabels), prob, t / np.size(testLabels), w


def accuracyModelLDAMix(testFeatures, testLabels, model1, model2, classes, w1, w2):
    t = 0
    true = 0
    LDACov1 = LDA_Cov(model1, classes)
    LDACov2 = LDA_Cov(model2, classes)
    currentPredictor = np.zeros(np.size(testLabels))
    for i in range(np.size(testLabels)):
        auxt = time.time()
        _, currentProb1 = predictedModelLDAProbALL(testFeatures[i, :], model1, classes, LDACov1)
        _, currentProb2 = predictedModelLDAProbALL(testFeatures[i, :], model2, classes, LDACov2)
        currentProb = currentProb1 * w1 + currentProb2 * w2
        currentPredictor[i] = np.argmax(currentProb) + 1
        t += (time.time() - auxt)
        if currentPredictor[i] == testLabels[i]:
            true += 1

    cm = np.array(confusion_matrix(testLabels, currentPredictor))
    pre = np.zeros(classes)
    recall = np.zeros(classes)
    w = np.zeros(classes)
    for cla in range(classes):
        pre[cla] = cm[cla, cla] / cm[cla, :].sum()
        recall[cla] = cm[cla, cla] / cm[:, cla].sum()
        w[cla] = pre[cla] * recall[cla]

    print(cm)
    return true / np.size(testLabels), 0, t / np.size(testLabels), 0


def accuracyModelQDAconfusionMatrix(testFeatures, testLabels, model, classes):
    t = 0
    true = 0
    prob = 0
    currentPredictor = np.zeros(np.size(testLabels))
    for i in range(np.size(testLabels)):
        auxt = time.time()
        currentPredictor[i], currentProb = predictedModelQDAProb(testFeatures[i, :], model, classes)
        t += (time.time() - auxt)
        if currentPredictor[i] == testLabels[i]:
            true += 1
            prob += currentProb
    cm = np.array(confusion_matrix(testLabels, currentPredictor))
    pre = np.zeros(classes)
    recall = np.zeros(classes)
    w = np.zeros(classes)
    for cla in range(classes):
        pre[cla] = cm[cla, cla] / cm[cla, :].sum()
        recall[cla] = cm[cla, cla] / cm[:, cla].sum()
        w[cla] = pre[cla] * recall[cla]

    print(cm, pre, recall)
    return true / np.size(testLabels), prob, t / np.size(testLabels), w


def accuracyModelQDAMix(testFeatures, testLabels, model1, model2, classes, w1, w2):
    t = 0
    true = 0
    prob = 0
    currentPredictor = np.zeros(np.size(testLabels))

    for i in range(np.size(testLabels)):
        auxt = time.time()
        _, currentProb1 = predictedModelQDAProbALL(testFeatures[i, :], model1, classes)
        _, currentProb2 = predictedModelQDAProbALL(testFeatures[i, :], model2, classes)
        currentProb = currentProb1 * w1 + currentProb2 * w2
        currentPredictor[i] = np.argmax(currentProb) + 1
        t += (time.time() - auxt)
        if currentPredictor[i] == testLabels[i]:
            true += 1

    cm = np.array(confusion_matrix(testLabels, currentPredictor))
    pre = np.zeros(classes)
    recall = np.zeros(classes)
    w = np.zeros(classes)
    for cla in range(classes):
        pre[cla] = cm[cla, cla] / cm[cla, :].sum()
        recall[cla] = cm[cla, cla] / cm[:, cla].sum()
        w[cla] = pre[cla] * recall[cla]

    print(cm)
    return true / np.size(testLabels), 0, t / np.size(testLabels), 0


def accuracyModelQDAProb(testFeatures, testLabels, model, classes):
    t = 0
    true = 0
    prob = 0
    for i in range(0, np.size(testLabels)):
        auxt = time.time()
        actualPredictor, currentProb = predictedModelQDAProb(testFeatures[i, :], model, classes)
        t += (time.time() - auxt)
        if actualPredictor == testLabels[i]:
            true += 1
            prob += currentProb

    return true / np.size(testLabels), prob, t / np.size(testLabels)


def predictedModelQDAProb(sample, model, classes):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = QDA_Discriminant(sample, model['cov'].loc[cl], model['mean'].loc[cl])
        if math.isnan(d[cl]):
            return predictedModelQDA_pseudoProb(sample, model, classes)
    d = d - d[np.argmin(d)]
    return np.argmax(d) + 1, d[np.argmax(d)] / d.sum()


def predictedModelQDA_pseudoProb(sample, model, classes):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = QDA_Discriminant_pseudo(sample, model['cov'].loc[cl], model['mean'].loc[cl])
    d = d - d[np.argmin(d)]
    return np.argmax(d) + 1, d[np.argmax(d)] / d.sum()


def accuracyModelLDAProb(testFeatures, testLabels, model, classes):
    t = 0
    true = 0
    prob = 0
    LDACov = LDA_Cov(model, classes)
    for i in range(np.size(testLabels)):
        auxt = time.time()
        currentPredictor, currentProb = predictedModelLDAProb(testFeatures[i, :], model, classes, LDACov)
        t += (time.time() - auxt)
        if currentPredictor == testLabels[i]:
            true += 1
            prob += currentProb

    return true / np.size(testLabels), prob, t / np.size(testLabels)


def predictedModelLDAProb(sample, model, classes, LDACov):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = LDA_Discriminant(sample, LDACov, model['mean'].loc[cl])
        if math.isnan(d[cl]):
            return predictedModelLDA_pseudoProb(sample, model, classes, LDACov)
    d = d - d[np.argmin(d)]
    return np.argmax(d) + 1, d[np.argmax(d)] / d.sum()


def predictedModelLDA_pseudoProb(sample, model, classes, LDACov):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = LDA_Discriminant_pseudo(sample, LDACov, model['mean'].loc[cl])
    d = d - d[np.argmin(d)]
    return np.argmax(d) + 1, d[np.argmax(d)] / d.sum()


def predictedModelLDAProbALL(sample, model, classes, LDACov):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = LDA_Discriminant(sample, LDACov, model['mean'].loc[cl])
        if math.isnan(d[cl]):
            return predictedModelLDA_pseudoProbALL(sample, model, classes, LDACov)
    d = d - d[np.argmin(d)]
    return np.argmax(d) + 1, d / d.sum()


def predictedModelLDA_pseudoProbALL(sample, model, classes, LDACov):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = LDA_Discriminant_pseudo(sample, LDACov, model['mean'].loc[cl])
    d = d - d[np.argmin(d)]
    return np.argmax(d) + 1, d / d.sum()


def predictedModelQDAProbALL(sample, model, classes):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = QDA_Discriminant(sample, model['cov'].loc[cl], model['mean'].loc[cl])
        if math.isnan(d[cl]):
            return predictedModelQDA_pseudoProbALL(sample, model, classes)
    d = d - d[np.argmin(d)]
    return np.argmax(d) + 1, d / d.sum()


def predictedModelQDA_pseudoProbALL(sample, model, classes):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = QDA_Discriminant_pseudo(sample, model['cov'].loc[cl], model['mean'].loc[cl])
    d = d - d[np.argmin(d)]
    return np.argmax(d) + 1, d / d.sum()


def accuracyModelQDAProbabilities(testFeatures, testLabels, model, classes):
    testFeaturesNew = []
    testLabelsNew = []
    testProbsNew = []
    print('samples total:', np.size(testLabels))
    for i in range(np.size(testLabels)):

        currentPredictor, currentProb = predictedModelQDAProb(testFeatures[i, :], model, classes)

        if currentPredictor == testLabels[i]:
            testFeaturesNew.append(list(testFeatures[i, :]))
            testLabelsNew.append(testLabels[i])
            testProbsNew.append(currentProb)
        # else:
        #     testFeaturesNew.append(list(testFeatures[i, :]))
        #     testLabelsNew.append(testLabels[i])
        #     testProbsNew.append(0.5)

    testFeaturesNew = np.array(testFeaturesNew)
    testLabelsNew = np.array(testLabelsNew)
    testProbsNew = np.array(testProbsNew)
    print('samples total:', np.size(testLabelsNew))
    return testFeaturesNew, testLabelsNew, testProbsNew


#######CLASSIFICATION

def scoreModelLDA_Classification(testFeatures, testLabels, model, classes, testRepetitions):
    true = 0
    count = 0
    LDACov = LDA_Cov(model, classes)
    auxRep = testRepetitions[0]
    auxClass = testLabels[0]
    actualPredictor = np.empty(0)

    for i in range(0, np.size(testLabels)):
        if auxRep != testRepetitions[i] or auxClass != testLabels[i] or i == np.size(testLabels) - 1:
            auxRep = testRepetitions[i]
            auxClass = testLabels[i]

            if stats.mode(actualPredictor)[0][0] == testLabels[i - 1]:
                true += 1
                count += 1
            else:
                count += 1
            actualPredictor = np.empty(0)
        actualPredictor = np.append(actualPredictor, predictedModelLDA(testFeatures[i, :], model, classes, LDACov))

    return true / count


def scoreModelQDA_Classification(testFeatures, testLabels, model, classes, testRepetitions):
    true = 0
    count = 0
    auxRep = testRepetitions[0]
    auxClass = testLabels[0]
    actualPredictor = np.empty(0)

    for i in range(0, np.size(testLabels)):
        if auxRep != testRepetitions[i] or auxClass != testLabels[i] or i == np.size(testLabels) - 1:
            auxRep = testRepetitions[i]
            auxClass = testLabels[i]
            if stats.mode(actualPredictor)[0][0] == testLabels[i - 1]:
                true += 1
                count += 1
            else:
                count += 1
            actualPredictor = np.empty(0)
        actualPredictor = np.append(actualPredictor, predictedModelQDA(testFeatures[i, :], model, classes))

    return true / count




##############################################################################################
#############################################################################################




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
             typeModel, k):
    t = time.time()
    numSamples=50
    trainFeatures, trainLabels = subsetTraining(trainFeatures, trainLabels, numSamples, classes)

    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class'])

    for cla in range(classes):
        adaptiveModel.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
        adaptiveModel.at[cla, 'mean'] = np.zeros((1, allFeatures))[0]

    if typeModel == 'LDA':
        wTarget = mccModelLDA_ALL(trainFeatures, trainLabels, currentValues, classes, step)
    elif typeModel == 'QDA':
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
            wPeopleMean[i] = weightPerPersonMean(currentValues, personMean, cla, classes
                                                 , trainFeatures, trainLabels, step, typeModel)
            wPeopleCov[i] = weightPerPersonCov(currentValues, personCov, cla, classes
                                               , trainFeatures, trainLabels, step, typeModel)

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

    trainingTime = time.time() - t
    return adaptiveModel, wTargetMean, wTargetMean.mean(), wTargetCov, wTargetCov.mean(), trainingTime


# Weight Calculation
def weightPerPersonMean(currentValues, personMean, currentClass, classes, trainFeatures, trainLabels, step,
                        typeModel):
    personValues = currentValues.copy()
    personValues['mean'].at[currentClass] = personMean
    if typeModel == 'LDA':
        weight = mccModelLDA(trainFeatures, trainLabels, personValues, classes, currentClass, step)
    elif typeModel == 'QDA':
        weight = mccModelQDA(trainFeatures, trainLabels, personValues, classes, currentClass, step)

    return weight


def weightPerPersonCov(currentValues, personCov, currentClass, classes, trainFeatures, trainLabels, step,
                       typeModel):
    personValues = currentValues.copy()
    personValues['cov'].at[currentClass] = personCov
    if typeModel == 'LDA':
        weight = mccModelLDA(trainFeatures, trainLabels, personValues, classes, currentClass, step)
    elif typeModel == 'QDA':
        weight = mccModelQDA(trainFeatures, trainLabels, personValues, classes, currentClass, step)
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


###### GENETIC ALGORITHM all parameters


def mutation(weights, classes):
    weights[np.random.randint(2 * classes)] = np.random.rand()
    return list(weights)


def mutationIdx(weights, idx):
    weights[idx] = np.random.rand()
    return list(weights)


def crossover(weights, classes, numberCrossover):
    weightsLast = []
    mutations = []
    lenWeights = len(weights)
    for i in range(0, lenWeights, 2):

        for idx in range(2 * classes):
            mutations.append(mutationIdx(weights[i, :].copy(), idx))
            mutations.append(mutationIdx(weights[(i + 1) % lenWeights, :].copy(), idx))

        for j in range(numberCrossover):
            idx = np.random.randint(1, 2 * classes - 1)
            weightsLast.append(list(np.hstack((weights[i, :idx], weights[(i + j + 1) % lenWeights, idx:]))))
            weightsLast.append(list(np.hstack((weights[(i + j + 1) % lenWeights, :idx], weights[i, idx:]))))

            # mutations.append(mutation(weights[i, :].copy(), classes))
            # mutations.append(mutation(weights[(i + j + 1) % lenWeights, :].copy(), classes))

    if mutations == []:
        return np.around(np.vstack((weights, np.array(weightsLast))), decimals=2)
    else:
        return np.around(np.vstack((weights, np.array(weightsLast), np.array(mutations))), decimals=2)
    # if mutations == []:
    #     return np.vstack((np.array(weightsLast),weights))
    # else:
    #     return np.vstack((np.array(weightsLast), np.array(mutations),weights))


def fitness(weightsArray, currentValues, classes, trainFeatures, trainLabels, typeModel, numSamples,
            selectedChromosomes, modelPK):
    trainFeatures, trainLabels = subsetTraining(trainFeatures, trainLabels, numSamples, classes)
    y = []
    for i in range(len(weightsArray)):
        y.append(
            fitnessModel(currentValues, classes, trainFeatures, trainLabels, typeModel, weightsArray[i, :], modelPK)[0])

    idx = np.argsort(np.array(y) * -1, kind='mergesort')[:selectedChromosomes]

    y = np.array(y)
    # print('eval: ', y[list(idx)], list(idx))

    return weightsArray[list(idx), :]


def fitnessModel(currentValues, classes, trainFeatures, trainLabels, typeModel, weightsArray, modelPK):
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class'])

    for cla in range(classes):
        adaptiveModel.at[cla, 'mean'] = currentValues['mean'].loc[cla] * weightsArray[cla] + modelPK['mean'].loc[
            cla] * (1 - weightsArray[cla])
        adaptiveModel.at[cla, 'cov'] = currentValues['cov'].loc[cla] * weightsArray[cla + classes] + modelPK['cov'].loc[
            cla] * (1 - weightsArray[cla + classes])
        adaptiveModel.at[cla, 'class'] = cla + 1

    if typeModel == 'LDA':
        acc, prob, _ = DA_Classifiers.accuracyModelLDAProb(trainFeatures, trainLabels, adaptiveModel, classes)
        return prob, acc, adaptiveModel
    elif typeModel == 'QDA':
        acc, prob, _ = DA_Classifiers.accuracyModelQDAProb(trainFeatures, trainLabels, adaptiveModel, classes)
        return prob, acc, adaptiveModel


def subsetTraining(trainFeatures, trainLabels, numSamples, classes):
    idx = []
    for cla in range(classes):
        idx.extend(list(np.random.choice(np.where(trainLabels == cla + 1)[0], size=numSamples)))
    return trainFeatures[idx], trainLabels[idx]


def weightsMCC(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels,
               typeModel, numSamples):
    step = 1
    trainFeatures, trainLabels = subsetTraining(trainFeatures, trainLabels, numSamples, classes)
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class'])

    for cla in range(classes):
        preTrainedMatrix_Class = pd.DataFrame(
            preTrainedDataMatrix[['cov', 'mean']].loc[(preTrainedDataMatrix['class'] == cla + 1)])
        preTrainedMatrix_Class = preTrainedMatrix_Class.reset_index(drop=True)

        peopleClass = len(preTrainedMatrix_Class.index)

        wPeopleMean = np.zeros(peopleClass)
        wPeopleCov = np.zeros(peopleClass)

        for i in range(peopleClass):
            personMean = preTrainedMatrix_Class['mean'].loc[i]
            personCov = preTrainedMatrix_Class['cov'].loc[i]
            wPeopleMean[i] = weightPerPersonMean(currentValues, personMean, cla, classes
                                                 , trainFeatures, trainLabels, step, typeModel)
            wPeopleCov[i] = weightPerPersonCov(currentValues, personCov, cla, classes
                                               , trainFeatures, trainLabels, step, typeModel)

        if wPeopleMean.sum() != 0:
            wPeopleMean = wPeopleMean / wPeopleMean.sum()
            adaptiveModel.at[cla, 'mean'] = np.sum(preTrainedMatrix_Class['mean'] * wPeopleMean)
        else:
            adaptiveModel.at[cla, 'mean'] = np.sum(preTrainedMatrix_Class['mean']) / peopleClass
            print('do not find any similar MEAN for the class', cla + 1, 'type: ', typeModel)
        if wPeopleCov.sum() != 0:
            wPeopleCov = wPeopleCov / wPeopleCov.sum()
            adaptiveModel.at[cla, 'cov'] = np.sum(preTrainedMatrix_Class['cov'] * wPeopleCov)
        else:
            adaptiveModel.at[cla, 'cov'] = np.sum(preTrainedMatrix_Class['cov']) / peopleClass
            print('do not find any similar COV for the class', cla + 1, 'type: ', typeModel)
        adaptiveModel.at[cla, 'class'] = cla + 1

    return adaptiveModel


def joinModels(currentValues, classes, trainFeatures, trainLabels, typeModel, modelPK):
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class'])

    if typeModel == 'LDA':
        acc, _, _, w1 = DA_Classifiers.accuracyModelLDAconfusionMatrix(trainFeatures, trainLabels, currentValues,
                                                                       classes)
        accpk, _, _, w2 = DA_Classifiers.accuracyModelLDAconfusionMatrix(trainFeatures, trainLabels, modelPK, classes)
        # print('training accuracy (initial) LDA: ', acc, accpk)
    elif typeModel == 'QDA':
        acc, _, _, w1 = DA_Classifiers.accuracyModelQDAconfusionMatrix(trainFeatures, trainLabels, currentValues,
                                                                       classes)
        accpk, _, _, w2 = DA_Classifiers.accuracyModelQDAconfusionMatrix(trainFeatures, trainLabels, modelPK, classes)
        # print('training accuracy (initial) QDA: ', acc, accpk)
    w1[np.isnan(w1)] = 0
    w2[np.isnan(w2)] = 0
    print('training accuracy (initial)', typeModel, acc, accpk, w1, w2)
    auxw2 = w2.copy()
    auxw1 = w1.copy()
    w1 = auxw1 / (auxw1 + auxw2)
    w2 = auxw2 / (auxw1 + auxw2)

    for cla in range(classes):
        adaptiveModel.at[cla, 'cov'] = currentValues.loc[cla, 'cov'] * w1[cla] + modelPK.loc[cla, 'cov'] * w2[cla]
        adaptiveModel.at[cla, 'mean'] = currentValues.loc[cla, 'mean'] * w1[cla] + modelPK.loc[cla, 'mean'] * w2[cla]
        adaptiveModel.at[cla, 'class'] = cla + 1
    return adaptiveModel


def GeneticModel(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, typeModel,
                 modelPK):
    t = time.time()
    numSamplesMCC = 30

    numSamplesGA = 20
    selectedChromosomes = 6
    iteractions = 50
    numberCrossover = 1

    # # weights = np.around(np.random.rand(selectedChromosomes, classes * 2), decimals=2)
    # # weights = np.around(np.ones((selectedChromosomes, classes * 2)), decimals=2)
    # weights = np.vstack((np.ones((1, classes * 2)), np.random.rand(selectedChromosomes - 1, classes * 2)))
    # print('weights initial: ', weights)

    if typeModel == 'LDA':
        acc, _, _, w1 = DA_Classifiers.accuracyModelLDAconfusionMatrix(trainFeatures, trainLabels, currentValues,
                                                                       classes)
        accpk, _, _, w2 = DA_Classifiers.accuracyModelLDAconfusionMatrix(trainFeatures, trainLabels, modelPK, classes)
        print('training accuracy (initial) LDA: ', acc, accpk)
    elif typeModel == 'QDA':
        acc, _, _, w1 = DA_Classifiers.accuracyModelQDAconfusionMatrix(trainFeatures, trainLabels, currentValues,
                                                                       classes)
        accpk, _, _, w2 = DA_Classifiers.accuracyModelQDAconfusionMatrix(trainFeatures, trainLabels, modelPK, classes)
        print('training accuracy (initial) QDA: ', acc, accpk)

    # for i in range(iteractions):
    #     weightsArray = crossover(weights, classes, numberCrossover)
    #     print('weightsArray', weightsArray[0, :])
    #     weights = fitness(weightsArray, currentValues, classes, trainFeatures, trainLabels, typeModel, numSamplesGA,
    #                       selectedChromosomes, modelPK)
    # weights = weights.mean(axis=0)
    #
    # prob, acc, adaptiveModel = fitnessModel(currentValues, classes, trainFeatures, trainLabels, typeModel, weights,
    #                                         modelPK)
    # print('training accuracy (final): ', prob, acc, weights)

    # if typeModel == 'LDA':
    #     adaptiveModel, _ = VidovicModel(currentValues, modelPK, classes, allFeatures)
    # elif typeModel == 'QDA':
    #     _, adaptiveModel = VidovicModel(currentValues, modelPK, classes, allFeatures)
    auxw2 = w2.copy()
    auxw1 = w1.copy()
    w1 = auxw1 / (auxw1 + auxw2)
    w2 = auxw2 / (auxw1 + auxw2)
    trainingTime = time.time() - t
    print('w: ', w1, w2)
    return modelPK, trainingTime, w1, w2


def OurModelModified(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels,
                     typeModel, numSamples):
    t = time.time()
    step = 1
    trainFeatures, trainLabels = subsetTraining(trainFeatures, trainLabels, numSamples, classes)
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class'])

    if typeModel == 'LDA':
        wTarget = mccModelLDA_ALL(trainFeatures, trainLabels, currentValues, classes, step)
    elif typeModel == 'QDA':
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
            wPeopleMean[i] = weightPerPersonMean(currentValues, personMean, cla, classes
                                                 , trainFeatures, trainLabels, step, typeModel)
            wPeopleCov[i] = weightPerPersonCov(currentValues, personCov, cla, classes
                                               , trainFeatures, trainLabels, step, typeModel)

        sumWMean = np.sum(wPeopleMean)

        if (sumWMean != 0) and (sumWMean + wTargetMean[cla] != 0):
            wTargetMean[cla] = wTargetMean[cla] / (wTargetMean[cla] + np.mean(wPeopleMean[wPeopleMean != 0]))
            wPeopleMean = (wPeopleMean / sumWMean) * (1 - wTargetMean[cla])

        else:
            wTargetMean[cla] = 1
            wPeopleMean = np.zeros(peopleClass)

        sumWCov = np.sum(wPeopleCov)
        if (sumWCov != 0) and (sumWCov + wTargetCov[cla] != 0):

            wTargetCov[cla] = wTargetCov[cla] / (wTargetCov[cla] + np.mean(wPeopleCov[wPeopleCov != 0]))
            wPeopleCov = (wPeopleCov / sumWCov) * (1 - wTargetCov[cla])

        else:
            wTargetCov[cla] = 1
            wPeopleCov = np.zeros(peopleClass)

        adaptiveModel.at[cla, 'cov'] = np.sum(preTrainedMatrix_Class['cov'] * wPeopleCov) + currentCov * wTargetCov[cla]
        adaptiveModel.at[cla, 'mean'] = np.sum(preTrainedMatrix_Class['mean'] * wPeopleMean) + currentMean * \
                                        wTargetMean[cla]
        adaptiveModel.at[cla, 'class'] = cla + 1

    trainingTime = time.time() - t
    return adaptiveModel, wTargetMean, wTargetMean.mean(), wTargetCov, wTargetCov.mean(), trainingTime


###### Few-class

def OurModelFewClass(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, step,
                     typeModel, k):
    t = time.time()

    peopleClass = len(preTrainedDataMatrix.loc[(preTrainedDataMatrix['class'] == 1)])
    wPeopleMeanNoClass = []
    wPeopleCovNoClass = []

    deletedClasses = 1
    NoClass = 1
    currentValuesNoClass = currentValues.loc[currentValues['class'] != NoClass]
    currentValuesNoClass = currentValuesNoClass.reset_index(drop=True)

    step = 1
    numSamples = 30
    trainFeatures, trainLabels = subsetTraining(trainFeatures, trainLabels, numSamples, classes)

    idxCla = 0
    for cla in range(classes):
        if cla != NoClass - 1:
            currentValuesNoClass.at[idxCla, 'class'] = idxCla + 1
            trainLabels[trainLabels == cla + 1] = idxCla + 1
            idxCla += 1
        else:
            trainFeatures = trainFeatures[trainLabels != cla + 1, :]
            trainLabels = trainLabels[trainLabels != cla + 1]

    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class'])
    # for cla in range(classes):
    #     adaptiveModel.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
    #     adaptiveModel.at[cla, 'mean'] = np.zeros((1, allFeatures))[0]

    if typeModel == 'LDA':
        wTarget = mccModelLDA_ALL(trainFeatures, trainLabels, currentValuesNoClass, classes - deletedClasses, step)
    elif typeModel == 'QDA':
        wTarget = mccModelQDA_ALL(trainFeatures, trainLabels, currentValuesNoClass, classes - deletedClasses, step)
    wTargetCov = wTarget.copy()
    wTargetMean = wTarget.copy()

    idxCla = 0
    for cla in range(classes):
        if cla != NoClass - 1:
            preTrainedMatrix_Class = pd.DataFrame(
                preTrainedDataMatrix[['cov', 'mean']].loc[(preTrainedDataMatrix['class'] == cla + 1)])
            preTrainedMatrix_Class = preTrainedMatrix_Class.reset_index(drop=True)
            currentCov = currentValuesNoClass['cov'].loc[idxCla]
            currentMean = currentValuesNoClass['mean'].loc[idxCla]
            # peopleClass = len(preTrainedMatrix_Class.index)

            wPeopleMean = np.zeros(peopleClass)
            wPeopleCov = np.zeros(peopleClass)

            for i in range(peopleClass):
                personMean = preTrainedMatrix_Class['mean'].loc[i]
                personCov = preTrainedMatrix_Class['cov'].loc[i]
                wPeopleMean[i] = weightPerPersonMean(currentValuesNoClass, personMean, idxCla, classes - deletedClasses
                                                     , trainFeatures, trainLabels, step, typeModel)
                wPeopleCov[i] = weightPerPersonCov(currentValuesNoClass, personCov, idxCla, classes - deletedClasses
                                                   , trainFeatures, trainLabels, step, typeModel)
            wPeopleMeanNoClass.append(list(wPeopleMean))
            wPeopleCovNoClass.append(list(wPeopleCov))

            sumWMean = np.sum(wPeopleMean)
            if sumWMean != 0:
                wTargetMean[idxCla] = wTargetMean[idxCla] / (
                        wTargetMean[idxCla] + np.mean(wPeopleMean[wPeopleMean != 0]) * k)
                wPeopleMean = (wPeopleMean / sumWMean) * (1 - wTargetMean[idxCla])

            else:
                wTargetMean[idxCla] = 1
                wPeopleMean = np.zeros(peopleClass)

            sumWCov = np.sum(wPeopleCov)
            if sumWCov != 0:

                wTargetCov[idxCla] = wTargetCov[idxCla] / (
                        wTargetCov[idxCla] + np.mean(wPeopleCov[wPeopleCov != 0]) * k)
                wPeopleCov = (wPeopleCov / sumWCov) * (1 - wTargetCov[idxCla])

            else:
                wTargetCov[idxCla] = 1
                wPeopleCov = np.zeros(peopleClass)

            adaptiveModel.at[cla, 'cov'] = np.sum(preTrainedMatrix_Class['cov'] * wPeopleCov) + currentCov * wTargetCov[
                idxCla]
            adaptiveModel.at[cla, 'mean'] = np.sum(preTrainedMatrix_Class['mean'] * wPeopleMean) + currentMean * \
                                            wTargetMean[idxCla]
            adaptiveModel.at[cla, 'class'] = cla + 1

            idxCla += 1

    # wPeopleMean = np.mean(np.array(wPeopleMeanNoClass), axis=0)
    # wPeopleCov = np.mean(np.array(wPeopleCovNoClass), axis=0)
    # # print('all Mean', wPeopleMeanNoClass)
    # print('Mean', wPeopleMean)

    idx = np.argsort(np.array(wPeopleMean) * -1, kind='mergesort')[:1]
    # wPeopleMeanNew = np.zeros(peopleClass)
    # wPeopleMeanNew[list(idx)] = wPeopleMean[list(idx)]
    # wPeopleMean = wPeopleMeanNew
    # print('Mean', wPeopleMean)

    # print('all Cov', wPeopleCovNoClass)
    # print('Cov', wPeopleCov)
    #
    # idx = np.argsort(np.array(wPeopleCov) * -1, kind='mergesort')[:1]
    # wPeopleCovNew = np.zeros(peopleClass)
    # wPeopleCovNew[list(idx)] = wPeopleCov[list(idx)]
    # wPeopleCov = wPeopleCovNew
    # print('Cov', wPeopleCov)

    # sumWMean = np.sum(wPeopleMean)
    # if sumWMean == 0:
    #     wPeopleMean = np.ones(peopleClass) * (1 / peopleClass)
    #     print('do not find Mean')
    # else:
    #     wPeopleMean = wPeopleMean / sumWMean
    #
    # sumWCov = np.sum(wPeopleCov)
    # if sumWCov == 0:
    #     wPeopleCov = np.ones(peopleClass) * (1 / peopleClass)
    #     print('do not find Cov')
    # else:
    #     wPeopleCov = wPeopleCov / sumWCov

    d = np.zeros(allFeatures)

    # for cl in range(classes):
    #     if cl != NoClass - 1:
    #         preTrainedMatrix_Class = pd.DataFrame(
    #             preTrainedDataMatrix[['cov', 'mean']].loc[(preTrainedDataMatrix['class'] == cl + 1)])
    #         preTrainedMatrix_Class = preTrainedMatrix_Class.reset_index(drop=True)
    #         d += currentValues.loc[cl, 'mean'] - preTrainedMatrix_Class.loc[int(idx), 'mean']

    for cl in range(classes):
        if cl != NoClass - 1:
            preTrainedMatrix_Class = pd.DataFrame(
                preTrainedDataMatrix[['cov', 'mean']].loc[(preTrainedDataMatrix['class'] == cl + 1)])
            preTrainedMatrix_Class = preTrainedMatrix_Class.reset_index(drop=True)
            d += currentValues.loc[cl, 'mean'] - preTrainedMatrix_Class.loc[int(idx), 'mean']
        else:
            preTrainedMatrix_Class = pd.DataFrame(
                preTrainedDataMatrix[['cov', 'mean']].loc[(preTrainedDataMatrix['class'] == cl + 1)])
            preTrainedMatrix_Class = preTrainedMatrix_Class.reset_index(drop=True)
            d += (classes - deletedClasses) * preTrainedMatrix_Class.loc[int(idx), 'mean']

    cla = NoClass - 1
    preTrainedMatrix_Class = pd.DataFrame(
        preTrainedDataMatrix[['cov', 'mean']].loc[(preTrainedDataMatrix['class'] == cla + 1)])
    preTrainedMatrix_Class = preTrainedMatrix_Class.reset_index(drop=True)

    adaptiveModel = currentValues.copy()

    # adaptiveModel.at[cla, 'mean'] = preTrainedMatrix_Class.loc[int(idx), 'mean']
    adaptiveModel.at[cla, 'mean'] = d / (classes - deletedClasses)
    # adaptiveModel.at[cla, 'cov'] = np.sum(preTrainedMatrix_Class['cov'] * wPeopleCov)
    adaptiveModel.at[cla, 'class'] = cla + 1
    adaptiveModel = adaptiveModel.sort_values(by=['class'])
    adaptiveModel = adaptiveModel.reset_index(drop=True)
    trainingTime = time.time() - t
    return adaptiveModel, wTargetMean, wTargetMean.mean(), wTargetCov, wTargetCov.mean(), trainingTime


def KLdivergence(mean0, mean1, k, cov0, cov1):
    exp1 = np.trace(np.dot(np.linalg.inv(cov1), cov0))
    exp2 = np.dot(np.dot((mean1 - mean0).T, np.linalg.inv(cov1)), (mean1 - mean0))
    exp3 = np.log(np.linalg.det(cov1) / np.linalg.det(cov0))
    return 0.5 * (exp1 + exp2 - k + exp3)


def JSDdivergence(mean0, mean1, k, cov0, cov1):
    meanM = (mean0 + mean1) / 2
    covM = (cov0 + cov1) / 2
    js = KLdivergence(mean0, meanM, k, cov0, covM) + KLdivergence(mean1, meanM, k, cov1, covM)
    # js /= np.log(2)
    return np.sqrt(js / 2)


#### Data augmentation

def currentDistributionValues(trainFeatures, trainLabels, classes, allFeatures):
    currentValues = pd.DataFrame(columns=['cov', 'mean', 'class'])
    trainLabelsAux = trainLabels[np.newaxis]
    Matrix = np.hstack((trainFeatures, trainLabelsAux.T))
    for cla in range(classes):
        currentValues.at[cla, 'cov'] = np.cov(Matrix[np.where((Matrix[:, allFeatures] == cla + 1)), 0:allFeatures][0],
                                              rowvar=False)
        currentValues.at[cla, 'mean'] = np.mean(Matrix[np.where((Matrix[:, allFeatures] == cla + 1)), 0:allFeatures][0],
                                                axis=0)
        currentValues.at[cla, 'class'] = cla + 1

    return currentValues


def currentDistributionValuesEM(trainFeatures, trainLabels, classes, allFeatures, trainFeaturesGenNew,
                                trainLabelsGenNew, trainProbsGenNew):
    if trainLabelsGenNew is None:
        currentValues = pd.DataFrame(columns=['cov', 'mean', 'class'])

        for cla in range(classes):
            FeaturesCL = trainFeatures[trainLabels == cla + 1]
            ProbsCL = trainProbsGenNew[trainLabels == cla + 1]
            currentValues.at[cla, 'cov'] = covariaceMatrixProbs(FeaturesCL, ProbsCL)
            currentValues.at[cla, 'mean'] = meanMatrixProbs(FeaturesCL, ProbsCL)
            currentValues.at[cla, 'class'] = cla + 1

        return currentValues

    else:
        currentValues = pd.DataFrame(columns=['cov', 'mean', 'class'])

        for cla in range(classes):
            FeaturesCL = np.vstack(
                (trainFeatures[trainLabels == cla + 1], trainFeaturesGenNew[trainLabelsGenNew == cla + 1]))
            ProbsCL = np.hstack(
                (np.ones(len(trainLabels[trainLabels == cla + 1])), trainProbsGenNew[trainLabelsGenNew == cla + 1]))
            currentValues.at[cla, 'cov'] = covariaceMatrixProbs(FeaturesCL, ProbsCL)
            currentValues.at[cla, 'mean'] = meanMatrixProbs(FeaturesCL, ProbsCL)
            currentValues.at[cla, 'class'] = cla + 1

        return currentValues


def covariaceMatrixProbs(FeaturesMatrix, Probs):
    mean = FeaturesMatrix.mean(axis=0)
    substraction = FeaturesMatrix - mean
    features = len(FeaturesMatrix.T)
    covarianceMatrix = np.zeros((features, features))
    lenM = Probs.sum()
    for i in range(features):
        for j in range(features):
            covarianceMatrix[i, j] = (Probs * (substraction[:, i] * substraction[:, j])).sum() / (lenM - 1)
    return covarianceMatrix


def meanMatrixProbs(FeaturesMatrix, Probs):
    return (FeaturesMatrix.T * Probs).sum(axis=1) / Probs.sum()


def DAugResultsDataframe(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels, classes, allFeatures,
                         results, testFeatures, testLabels, idx, person, subset, featureSet, nameFile, printR,
                         trainFeaturesGen, trainLabelsGen):
    # PKmodel = currentDistributionValues(trainFeaturesGen, trainLabelsGen, classes, allFeatures)
    #
    # trainFeaturesNew, trainLabelsNew, trainProbsNew = DA_Classifiers.accuracyModelQDAProbabilities(
    #     trainFeatures, trainLabels, PKmodel, classes)
    # currentValuesNew = currentDistributionValuesEM(trainFeaturesNew, trainLabelsNew, classes, allFeatures,
    #                                                None, None, trainProbsNew)

    numSamples = 2000
    print('training samples: ', len(trainLabels))
    trainFeaturesGen, trainLabelsGen = subsetTraining(trainFeaturesGen, trainLabelsGen, numSamples, classes)
    trainFeaturesNew, trainLabelsNew, trainProbsNew = DA_Classifiers.accuracyModelQDAProbabilities(
        trainFeaturesGen, trainLabelsGen, currentValues, classes)

    # trainFeatures = np.vstack((trainFeatures, trainFeaturesGenNew))
    # trainLabels = np.hstack((trainLabels, trainLabelsGenNew))
    # currentValuesNew = currentDistributionValues(trainFeatures, trainLabels, classes, allFeatures)

    currentValuesNew = currentDistributionValuesEM(trainFeatures, trainLabels, classes, allFeatures,
                                                   trainFeaturesNew, trainLabelsNew, trainProbsNew)

    # modelPKLDA = adaptive.weightsMCC(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures,
    #                                  trainLabels, 'LDA', 30)
    # modelPKQDA = adaptive.weightsMCC(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures,
    #                                  trainLabels, 'QDA', 30)
    #
    # propModelLDA = adaptive.joinModels(currentValues, classes, trainFeatures, trainLabels, 'LDA', modelPKLDA)
    # propModelQDA = adaptive.joinModels(currentValues, classes, trainFeatures, trainLabels, 'QDA', modelPKQDA)

    # GeneticModelLDA, results.at[idx, 'timeTrainingPropLDA'], wL1, wL2 = adaptive.GeneticModel(
    #     currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, 'LDA', propModelLDA)
    # GeneticModelQDA, results.at[idx, 'timeTrainingPropQDA'], wQ1, wQ2 = adaptive.GeneticModel(
    #     currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, 'QDA', propModelQDA)

    results.at[idx, 'person'] = person
    results.at[idx, 'subset'] = subset
    results.at[idx, '# shots'] = np.size(subset)
    results.at[idx, 'Feature Set'] = featureSet
    # LDA results
    print('indLDA')
    results.at[idx, 'AccLDAInd'], _, results.at[idx, 'tIndL'], _ = DA_Classifiers.accuracyModelLDAconfusionMatrix(
        testFeatures, testLabels, currentValues, classes)
    print('propLDA')
    results.at[idx, 'AccLDAProp'], _, results.at[idx, 'tPropL'], _ = DA_Classifiers.accuracyModelLDAconfusionMatrix(
        testFeatures, testLabels, currentValuesNew, classes)
    print('indQDA')
    results.at[idx, 'AccQDAInd'], _, results.at[idx, 'tIndQ'], _ = DA_Classifiers.accuracyModelQDAconfusionMatrix(
        testFeatures, testLabels, currentValues, classes)
    print('propQDA')
    results.at[idx, 'AccQDAProp'], _, results.at[idx, 'tPropQ'], _ = DA_Classifiers.accuracyModelQDAconfusionMatrix(
        testFeatures, testLabels, currentValuesNew, classes)

    # print('modelLDA')
    # results.at[idx, 'AccLDAGenetic'], _, _, _ = DA_Classifiers.accuracyModelLDAconfusionMatrix(
    #     testFeatures, testLabels, GeneticModelLDA, classes)
    #
    # results.at[idx, 'AccLDAMulti'], results.at[idx, 'tGenL'] = DA_Classifiers.accuracyModelLDA(testFeatures, testLabels,
    #                                                                                            pkValues,
    #                                                                                            classes)
    # print('mixLDA')
    # results.at[idx, 'AccLDAMix'], results.at[idx, 'tMixL'] = DA_Classifiers.accuracyModelLDAMix(
    #     testFeatures, testLabels, currentValues, GeneticModelLDA, classes, wL1, wL2)
    #
    # print('mixQDA')
    # results.at[idx, 'AccQDAMix'], results.at[idx, 'tMixQ'] = DA_Classifiers.accuracyModelQDAMix(
    #     testFeatures, testLabels, currentValues, GeneticModelQDA, classes, wQ1, wQ2)
    '''
    # results.at[idx, 'AccLDAProp'], results.at[idx, 'tCLPropL'] = DA_Classifiers.accuracyModelLDA(testFeatures,
    #                                                                                              testLabels, propModel,
    #                                                                                              classes)
    # results.at[idx, 'AccLDALiu'], _ = DA_Classifiers.accuracyModelLDA(testFeatures, testLabels, liuModel, classes)
    # results.at[idx, 'AccLDAVidovic'], _ = DA_Classifiers.accuracyModelLDA(testFeatures, testLabels, vidovicModelL,
    #                                                                       classes)

    ## QDA results


    results.at[idx, 'AccQDAInd'], results.at[idx, 'tIndQ'] = DA_Classifiers.accuracyModelQDA(testFeatures, testLabels,
                                                                                             currentValues,
                                                                                             classes)


    results.at[idx, 'AccQDAMulti'], results.at[idx, 'tGenQ'] = DA_Classifiers.accuracyModelQDA(testFeatures, testLabels,
                                                                                               pkValues,
                                                                                               classes)

    results.at[idx, 'AccQDAProp'], results.at[idx, 'tCLPropQ'] = DA_Classifiers.accuracyModelQDA(testFeatures,
                                                                                                 testLabels, propModel,
                                                                                                 classes)
    results.at[idx, 'AccQDALiu'], _ = DA_Classifiers.accuracyModelQDA(testFeatures, testLabels, liuModel, classes)
    results.at[idx, 'AccQDAVidovic'], _ = DA_Classifiers.accuracyModelQDA(testFeatures, testLabels, vidovicModelQ,
                                                                          classes)

    '''

    if nameFile is not None:
        results.to_csv(nameFile)
    if printR:
        print(featureSet)
        print('Results: person= ', person, ' shot set= ', subset)
        print(results.loc[idx])

    idx += 1

    return results, idx


def OurModelModified2(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, step,
                      typeModel, k):
    t = time.time()

    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class'])

    for cla in range(classes):
        adaptiveModel.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
        adaptiveModel.at[cla, 'mean'] = np.zeros((1, allFeatures))[0]

    if typeModel == 'LDA':
        wTarget = mccModelLDA_ALL(trainFeatures, trainLabels, currentValues, classes, step)
    elif typeModel == 'QDA':
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
            wPeopleMean[i] = weightPerPersonMean(currentValues, personMean, cla, classes
                                                 , trainFeatures, trainLabels, step, typeModel)
            wPeopleCov[i] = weightPerPersonCov(currentValues, personCov, cla, classes
                                               , trainFeatures, trainLabels, step, typeModel)

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

        print('peoplecov', wPeopleCov)

        adaptiveModel.at[cla, 'cov'] = np.sum(preTrainedMatrix_Class['cov'] * wPeopleCov) + currentCov * wTargetCov[cla]

        # adaptiveModel.at[cla, 'mean'] = np.sum(preTrainedMatrix_Class['mean'] * wPeopleMean) + currentMean * \
        #                                 wTargetMean[cla]
        adaptiveModel.at[cla, 'mean'] = currentValues['mean'].loc[cla]
        adaptiveModel.at[cla, 'class'] = cla + 1

    trainingTime = time.time() - t
    return adaptiveModel, wTargetMean, wTargetMean.mean(), wTargetCov, wTargetCov.mean(), trainingTime


def OurModelNok(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, step,
                typeModel, k):
    t = time.time()

    step = 1
    numSamples = 20
    trainFeatures, trainLabels = subsetTraining(trainFeatures, trainLabels, numSamples, classes)

    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class'])

    for cla in range(classes):
        adaptiveModel.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
        adaptiveModel.at[cla, 'mean'] = np.zeros((1, allFeatures))[0]

    if typeModel == 'LDA':
        wTarget = mccModelLDA_ALLProbs(trainFeatures, trainLabels, currentValues, classes, step)
    elif typeModel == 'QDA':
        wTarget = mccModelQDA_ALLProbs(trainFeatures, trainLabels, currentValues, classes, step)
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
            wPeopleMean[i] = weightPerPersonMeanProb(currentValues, personMean, cla, classes
                                                     , trainFeatures, trainLabels, step, typeModel)
            wPeopleCov[i] = weightPerPersonCovProb(currentValues, personCov, cla, classes
                                                   , trainFeatures, trainLabels, step, typeModel)

        sumWMean = np.sum(wPeopleMean)
        if sumWMean != 0:
            wPeopleMean = (wPeopleMean / sumWMean)
            adaptiveModel.at[cla, 'mean'] = np.sum(preTrainedMatrix_Class['mean'] * wPeopleMean)

            # usedDistributions = len(wPeopleMean[wPeopleMean != 0])
            # wTargetMean[cla] = 1 - (sumWMean / usedDistributions)
            # wPeopleMean = (wPeopleMean / sumWMean) * (1 - wTargetMean[cla])
            # print('distrib', usedDistributions, sumWMean, wTargetMean[cla])

            # wPeopleMean = (wPeopleMean / sumWMean)
            # meanPK = np.sum(preTrainedMatrix_Class['mean'] * wPeopleMean)
            # wPKmean = weightPerPersonMeanProb(currentValues, meanPK, cla, classes, trainFeatures, trainLabels, step,
            #                                   typeModel)
            # wTargetMean[cla] = wTargetMean[cla] / (wTargetMean[cla] + wPKmean)
            # wPKmean = 1 - wTargetMean[cla]

        else:
            wTargetMean[cla] = 1

            wPeopleMean = np.zeros(peopleClass)
            adaptiveModel.at[cla, 'mean'] = currentMean

            # wPKmean = 1 - wTargetMean[cla]
            # meanPK = 0

        sumWCov = np.sum(wPeopleCov)
        if sumWCov != 0:
            wPeopleCov = (wPeopleCov / sumWCov)
            adaptiveModel.at[cla, 'cov'] = np.sum(preTrainedMatrix_Class['cov'] * wPeopleCov)

            # usedDistributions = len(wPeopleCov[wPeopleCov != 0])
            # wTargetCov[cla] = 1 - (sumWCov / usedDistributions)
            # wPeopleCov = (wPeopleCov / sumWCov) * (1 - wTargetCov[cla])

            # wPeopleCov = (wPeopleCov / sumWCov)
            # covPK = np.sum(preTrainedMatrix_Class['cov'] * wPeopleCov)
            # wPKcov = weightPerPersonCovProb(currentValues, covPK, cla, classes, trainFeatures, trainLabels, step,
            #                                 typeModel)
            # wTargetCov[cla] = wTargetCov[cla] / (wTargetCov[cla] + wPKcov)
            # wPKcov = 1 - wTargetCov[cla]

        else:
            wTargetCov[cla] = 1

            wPeopleCov = np.zeros(peopleClass)
            adaptiveModel.at[cla, 'cov'] = currentCov

            # wPKcov = 1 - wTargetCov[cla]
            # covPK = 0

        # # adaptiveModel.at[cla, 'cov'] = np.sum(preTrainedMatrix_Class['cov'] * wPeopleCov) + currentCov * wTargetCov[cla]
        # # adaptiveModel.at[cla, 'mean'] = np.sum(preTrainedMatrix_Class['mean'] * wPeopleMean) + currentMean * \
        # #                                 wTargetMean[cla]
        # adaptiveModel.at[cla, 'class'] = cla + 1

        # adaptiveModel.at[cla, 'cov'] = covPK * wPKcov + currentCov * wTargetCov[cla]
        # adaptiveModel.at[cla, 'mean'] = meanPK * wPKmean + currentMean * wTargetMean[cla]
        # adaptiveModel.at[cla, 'class'] = cla + 1

    print('FIRST')
    print('weight mean current:', wTargetMean)
    print('weight cov current:', wTargetCov)

    wTargetNew = mccModelQDA_ALLProbs(trainFeatures, trainLabels, adaptiveModel, classes, step)
    wTargetMean = wTarget / (wTarget + wTargetNew)
    wTargetCov = wTarget / (wTarget + wTargetNew)

    print('weight current:', wTargetMean)
    trainingTime = time.time() - t
    return adaptiveModel, wTargetMean, wTargetMean.mean(), wTargetCov, wTargetCov.mean(), trainingTime


def mccModelLDA_ALLProbs(testFeatures, testLabels, model, classes, step):
    TP = np.zeros([classes])
    TN = np.zeros([classes])
    FP = np.zeros([classes])
    FN = np.zeros([classes])

    LDACov = DA_Classifiers.LDA_Cov(model, classes)

    for i in range(0, np.size(testLabels), step):
        currentPredictor, currentProb = DA_Classifiers.predictedModelLDAProb(testFeatures[i, :], model, classes, LDACov)

        if currentPredictor == testLabels[i]:
            TP[int(testLabels[i] - 1)] += currentProb
            for j in range(classes):
                if j != int(testLabels[i] - 1):
                    TN[j] += currentProb
        else:
            FP[int(testLabels[i] - 1)] += currentProb
            for j in range(classes):
                if j != int(testLabels[i] - 1):
                    FN[j] += currentProb

    return mcc(TP, TN, FP, FN)


def mccModelQDA_ALLProbs(testFeatures, testLabels, model, classes, step):
    TP = np.zeros([classes])
    TN = np.zeros([classes])
    FP = np.zeros([classes])
    FN = np.zeros([classes])

    for i in range(0, np.size(testLabels), step):
        currentPredictor, currentProb = DA_Classifiers.predictedModelQDAProb(testFeatures[i, :], model, classes)

        if currentPredictor == testLabels[i]:
            TP[int(testLabels[i] - 1)] += currentProb
            for j in range(classes):
                if j != int(testLabels[i] - 1):
                    TN[j] += currentProb
        else:
            FP[int(testLabels[i] - 1)] += currentProb
            for j in range(classes):
                if j != int(testLabels[i] - 1):
                    FN[j] += currentProb

    return mcc(TP, TN, FP, FN)


def mccModelLDAProb(testFeatures, testLabels, model, classes, currentClass, step):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    currentClass = currentClass + 1
    LDACov = DA_Classifiers.LDA_Cov(model, classes)
    for i in range(0, np.size(testLabels), step):
        currentPredictor, currentProb = DA_Classifiers.predictedModelLDAProb(testFeatures[i, :], model, classes, LDACov)
        if currentPredictor == testLabels[i]:
            if currentPredictor == currentClass:
                TP += currentProb
            else:
                TN += currentProb
        else:
            if currentPredictor == currentClass:
                FP += currentProb
            else:
                FN += currentProb
    return mcc(TP, TN, FP, FN)


def mccModelQDAProb(testFeatures, testLabels, model, classes, currentClass, step):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    currentClass = currentClass + 1
    for i in range(0, np.size(testLabels), step):
        currentPredictor, currentProb = DA_Classifiers.predictedModelQDAProb(testFeatures[i, :], model, classes)

        if currentPredictor == testLabels[i]:
            if currentPredictor == currentClass:
                TP += currentProb
            else:
                TN += currentProb
        else:
            if currentPredictor == currentClass:
                FP += currentProb
            else:
                FN += currentProb

    return mcc(TP, TN, FP, FN)


def weightPerPersonMeanProb(currentValues, personMean, currentClass, classes, trainFeatures, trainLabels, step,
                            typeModel):
    personValues = currentValues.copy()
    personValues['mean'].at[currentClass] = personMean
    if typeModel == 'LDA':
        weight = mccModelLDAProb(trainFeatures, trainLabels, personValues, classes, currentClass, step)
    elif typeModel == 'QDA':
        weight = mccModelQDAProb(trainFeatures, trainLabels, personValues, classes, currentClass, step)

    return weight


def weightPerPersonCovProb(currentValues, personCov, currentClass, classes, trainFeatures, trainLabels, step,
                           typeModel):
    personValues = currentValues.copy()
    personValues['cov'].at[currentClass] = personCov
    if typeModel == 'LDA':
        weight = mccModelLDAProb(trainFeatures, trainLabels, personValues, classes, currentClass, step)
    elif typeModel == 'QDA':
        weight = mccModelQDAProb(trainFeatures, trainLabels, personValues, classes, currentClass, step)
    return weight


# def OurModelUnsupervised(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels,
#                          oneShotModel, step, typeModel, k):
#     t = time.time()
#
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class'])
#
#     for cla in range(classes):
#         adaptiveModel.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
#         adaptiveModel.at[cla, 'mean'] = np.zeros((1, allFeatures))[0]
#
#     # if typeModel == 'LDA':
#     #     wTarget = mccModelLDA_ALL(trainFeatures, trainLabels, currentValues, classes, step)
#     # elif typeModel == 'QDA':
#     #     wTarget = mccModelQDA_ALL(trainFeatures, trainLabels, currentValues, classes, step)
#     # wTargetCov = wTarget.copy()
#     # wTargetMean = wTarget.copy()
#
#     wTargetCov = np.ones(classes)
#     wTargetMean = np.ones(classes)
#
#     print('allpeople', preTrainedDataMatrix[['class', 'prob']])
#     for cla in range(classes):
#         preTrainedMatrix_Class = pd.DataFrame(
#             preTrainedDataMatrix[['cov', 'mean', 'prob']].loc[(preTrainedDataMatrix['class'] == cla + 1)])
#         if preTrainedMatrix_Class.empty:
#             adaptiveModel.at[cla, 'cov'] = currentValues['cov'].loc[cla]
#             adaptiveModel.at[cla, 'mean'] = currentValues['mean'].loc[cla]
#             adaptiveModel.at[cla, 'class'] = cla + 1
#         else:
#
#             preTrainedMatrix_Class = preTrainedMatrix_Class.reset_index(drop=True)
#             currentCov = currentValues['cov'].loc[cla]
#             currentMean = currentValues['mean'].loc[cla]
#             peopleClass = len(preTrainedMatrix_Class.index)
#
#             wPeopleMean = np.zeros(peopleClass)
#             wPeopleCov = np.zeros(peopleClass)
#
#             for i in range(peopleClass):
#                 personMean = preTrainedMatrix_Class['mean'].loc[i]
#                 personCov = preTrainedMatrix_Class['cov'].loc[i]
#                 # wPeopleMean[i] = JSDdivergence(currentMean, personMean, 8, currentCov, personCov)
#
#                 wPeopleMean[i] = weightPerPersonMean(oneShotModel, personMean, cla, classes
#                                                      , trainFeatures, trainLabels, step, typeModel) * \
#                                  preTrainedMatrix_Class['prob'].loc[i]
#                 wPeopleCov[i] = weightPerPersonCov(oneShotModel, personCov, cla, classes
#                                                    , trainFeatures, trainLabels, step, typeModel) * \
#                                 preTrainedMatrix_Class['prob'].loc[i]
#
#             wTargetMean[cla] = weightPerPersonMean(oneShotModel, currentMean, cla, classes, trainFeatures, trainLabels,
#                                                    step, typeModel)
#             wTargetCov[cla] = weightPerPersonCov(oneShotModel, currentCov, cla, classes, trainFeatures, trainLabels,
#                                                  step, typeModel)
#
#             sumWMean = np.sum(wPeopleMean)
#             # print('before class', cla + 1, wPeopleMean)
#             if (sumWMean != 0):
#                 wPeopleMean = wPeopleMean / (sumWMean + wTargetMean[cla])
#                 wTargetMean[cla] = wTargetMean[cla] / (sumWMean + wTargetMean[cla])
#
#                 # wTargetMean[cla] = wTargetMean[cla] / (wTargetMean[cla] + np.mean(wPeopleMean[wPeopleMean != 0]) * k)
#                 # wPeopleMean = (wPeopleMean / sumWMean) * (1 - wTargetMean[cla])
#
#             else:
#                 wTargetMean[cla] = 1
#                 wPeopleMean = np.zeros(peopleClass)
#
#             sumWCov = np.sum(wPeopleCov)
#             if (sumWCov != 0):
#                 wPeopleCov = wPeopleCov / (sumWCov + wTargetCov[cla])
#                 wTargetCov[cla] = wTargetCov[cla] / (sumWCov + wTargetCov[cla])
#
#                 # wTargetCov[cla] = wTargetCov[cla] / (wTargetCov[cla] + np.mean(wPeopleCov[wPeopleCov != 0]) * k)
#                 # wPeopleCov = (wPeopleCov / sumWCov) * (1 - wTargetCov[cla])
#
#             else:
#                 wTargetCov[cla] = 1
#                 wPeopleCov = np.zeros(peopleClass)
#
#             if typeModel == 'LDA':
#                 adaptiveModel.at[cla, 'cov'] = (np.sum(preTrainedMatrix_Class['cov']) + currentCov) / (peopleClass + 1)
#             else:
#                 adaptiveModel.at[cla, 'cov'] = np.sum(preTrainedMatrix_Class['cov'] * wPeopleCov) + currentCov * \
#                                                wTargetCov[cla]
#
#             print('class', cla + 1, wPeopleMean, wTargetMean[cla])
#             print('class cov', cla + 1, wPeopleCov, wTargetCov[cla])
#             adaptiveModel.at[cla, 'mean'] = np.sum(preTrainedMatrix_Class['mean'] * wPeopleMean) + currentMean * \
#                                             wTargetMean[cla]
#             adaptiveModel.at[cla, 'class'] = cla + 1
#
#     trainingTime = time.time() - t
#     return adaptiveModel, wTargetMean, wTargetMean.mean(), wTargetCov, wTargetCov.mean(), trainingTime


def OurModelUnsupervisedAllProb(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels,
                                oneShotModel, step, typeModel, typeDatabase):
    typeModelWeights='QDA'
    peopleClass = len(preTrainedDataMatrix.index)
    if typeDatabase == 'Nina5':
        preTrainedDataMatrix2 = pd.DataFrame(columns=['cov', 'mean', 'class', 'prob', 'samples'])
        i2 = 0
        idxSetBase = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 1])

        for j in range(int(peopleClass / classes)):
            idxSet = classes * j + idxSetBase

            for i in idxSet:
                preTrainedDataMatrix2.at[i2] = preTrainedDataMatrix.loc[i]
                i2 += 1

        preTrainedDataMatrix = preTrainedDataMatrix2.copy()

    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class'])

    for cla in range(classes):
        adaptiveModel.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
        adaptiveModel.at[cla, 'mean'] = np.zeros((1, allFeatures))[0]

    wTargetCov = np.zeros(classes)
    wTargetMean = np.zeros(classes)

    print('allpeople', preTrainedDataMatrix[['class', 'prob']])

    wPeopleMean = np.zeros((peopleClass, classes))
    wPeopleCov = np.zeros((peopleClass, classes))

    for cla in range(classes):
        wTargetMean[cla] = weightPerPersonMean(oneShotModel, currentValues['mean'].loc[cla], cla, classes,
                                               trainFeatures, trainLabels, step, typeModelWeights)
        wTargetCov[cla] = weightPerPersonCov(oneShotModel, currentValues['cov'].loc[cla], cla, classes,
                                             trainFeatures, trainLabels, step, typeModelWeights)

        for person in range(peopleClass):
            # wPeopleMean[i] = JSDdivergence(currentMean, personMean, 8, currentCov, personCov)

            if preTrainedDataMatrix['prob'].loc[person][cla] != 0:
                personMean = preTrainedDataMatrix['mean'].loc[person]
                personCov = preTrainedDataMatrix['cov'].loc[person]
                wPeopleMean[person, cla] = weightPerPersonMean(
                    oneShotModel, personMean, cla, classes, trainFeatures, trainLabels, step, typeModelWeights) * \
                                           preTrainedDataMatrix['prob'].loc[person][cla]
                wPeopleCov[person, cla] = weightPerPersonCov(
                    oneShotModel, personCov, cla, classes, trainFeatures, trainLabels, step, typeModelWeights) * \
                                          preTrainedDataMatrix['prob'].loc[person][cla]
    sumWMean = np.sum(wPeopleMean, axis=0) + wTargetMean
    sumWCov = np.sum(wPeopleCov, axis=0) + wTargetCov
    wTargetMean /= sumWMean
    wTargetCov /= sumWCov
    wPeopleMean /= sumWMean
    wPeopleCov /= sumWCov
    wTargetMean=np.nan_to_num(wTargetMean,nan=1)
    wTargetCov = np.nan_to_num(wTargetCov,nan=1)
    wPeopleMean = np.nan_to_num(wPeopleMean)
    wPeopleCov = np.nan_to_num(wPeopleCov)
    print('mean weights', wPeopleMean)
    print(wTargetMean)
    print('cov weights', wPeopleCov)
    print(wTargetCov)
    means = np.resize(preTrainedDataMatrix['mean'], (classes, len(preTrainedDataMatrix['mean']))).T * wPeopleMean
    covs = np.resize(preTrainedDataMatrix['cov'], (classes, len(preTrainedDataMatrix['cov']))).T * wPeopleCov
    for cla in range(classes):
        adaptiveModel.at[cla, 'class'] = cla + 1
        adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + currentValues['mean'].loc[cla] * wTargetMean[cla]
        if typeModel == 'LDA':
            adaptiveModel.at[cla, 'cov'] = (np.sum(preTrainedDataMatrix['cov']) + currentValues['cov'].loc[cla]) / (
                    peopleClass + 1)
        else:
            adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + currentValues['cov'].loc[cla] * wTargetCov[cla]

    return adaptiveModel, wTargetMean, wTargetMean.mean(), wTargetCov, wTargetCov.mean(), 0
