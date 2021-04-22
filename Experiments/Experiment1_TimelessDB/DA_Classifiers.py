import math
import time

import numpy as np
import pandas as pd


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


def predictedModelLDA_Prob(sample, model, classes, LDACov):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = LDA_Discriminant(sample, LDACov, model['mean'].loc[cl])
        if math.isnan(d[cl]):
            return predictedModelLDA_pseudo_Prob(sample, model, classes, LDACov)
    if np.any(d < 0):
        d = d - d.min()
    return d / d.sum()


def predictedModelLDA_pseudo_Prob(sample, model, classes, LDACov):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = LDA_Discriminant_pseudo(sample, LDACov, model['mean'].loc[cl])
    if np.any(d < 0):
        d = d - d.min()
    return d / d.sum()


def accuracyModelLDA(testFeatures, testLabels, model, classes):
    t = 0
    true = 0
    count = 0
    LDACov = LDA_Cov(model, classes)
    precision = np.zeros((2, classes))
    recall = np.zeros((2, classes))
    for i in range(np.size(testLabels)):
        auxt = time.time()
        currentPredictor = predictedModelLDA(testFeatures[i, :], model, classes, LDACov)
        t += (time.time() - auxt)
        if currentPredictor == testLabels[i]:
            true += 1
            count += 1
            recall[0, int(testLabels[i] - 1)] += 1
            precision[0, int(currentPredictor - 1)] += 1
        else:
            count += 1
            recall[1, int(testLabels[i] - 1)] += 1
            precision[1, int(currentPredictor - 1)] += 1

    return true / count, np.nan_to_num(precision[0, :] / precision.sum(axis=0)), np.nan_to_num(
        recall[0, :] / recall.sum(axis=0)), t / np.size(testLabels)


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


def predictedModelQDA_Prob(sample, model, classes):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = QDA_Discriminant(sample, model['cov'].loc[cl], model['mean'].loc[cl])
        if math.isnan(d[cl]):
            return predictedModelQDA_pseudo_Prob(sample, model, classes)
    if np.any(d < 0):
        d = d - d.min()
    return d / d.sum()


def predictedModelQDA_pseudo_Prob(sample, model, classes):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = QDA_Discriminant_pseudo(sample, model['cov'].loc[cl], model['mean'].loc[cl])
    if np.any(d < 0):
        d = d - d.min()
    return d / d.sum()


def accuracyModelQDA(testFeatures, testLabels, model, classes):
    t = 0
    true = 0
    count = 0
    precision = np.zeros((2, classes))
    recall = np.zeros((2, classes))
    for i in range(0, np.size(testLabels)):
        auxt = time.time()
        currentPredictor = predictedModelQDA(testFeatures[i, :], model, classes)
        t += (time.time() - auxt)
        if currentPredictor == testLabels[i]:
            true += 1
            count += 1
            recall[0, int(testLabels[i] - 1)] += 1
            precision[0, int(currentPredictor - 1)] += 1
        else:
            count += 1
            recall[1, int(testLabels[i] - 1)] += 1
            precision[1, int(currentPredictor - 1)] += 1
    return true / count, np.nan_to_num(precision[0, :] / precision.sum(axis=0)), np.nan_to_num(
        recall[0, :] / recall.sum(axis=0)), t / np.size(testLabels)


######

# def accuracyModelLDAconfusionMatrix(testFeatures, testLabels, model, classes):
#     t = 0
#     true = 0
#     prob = 0
#     LDACov = LDA_Cov(model, classes)
#     currentPredictor = np.zeros(np.size(testLabels))
#     for i in range(np.size(testLabels)):
#         auxt = time.time()
#         currentPredictor[i], currentProb = predictedModelLDAProb(testFeatures[i, :], model, classes, LDACov)
#         t += (time.time() - auxt)
#         if currentPredictor[i] == testLabels[i]:
#             true += 1
#             prob += currentProb
#     cm = np.array(confusion_matrix(testLabels, currentPredictor))
#     pre = np.zeros(classes)
#     recall = np.zeros(classes)
#     w = np.zeros(classes)
#     for cla in range(classes):
#         pre[cla] = cm[cla, cla] / cm[cla, :].sum()
#         recall[cla] = cm[cla, cla] / cm[:, cla].sum()
#         w[cla] = pre[cla] * recall[cla]
#
#     print(cm, pre, recall)
#     return true / np.size(testLabels), prob, t / np.size(testLabels), w
#
#
# def accuracyModelQDAconfusionMatrix(testFeatures, testLabels, model, classes):
#     t = 0
#     true = 0
#     prob = 0
#     currentPredictor = np.zeros(np.size(testLabels))
#     for i in range(np.size(testLabels)):
#         auxt = time.time()
#         currentPredictor[i], currentProb = predictedModelQDAProb(testFeatures[i, :], model, classes)
#         t += (time.time() - auxt)
#         if currentPredictor[i] == testLabels[i]:
#             true += 1
#             prob += currentProb
#     cm = np.array(confusion_matrix(testLabels, currentPredictor))
#     pre = np.zeros(classes)
#     recall = np.zeros(classes)
#     w = np.zeros(classes)
#     for cla in range(classes):
#         pre[cla] = cm[cla, cla] / cm[cla, :].sum()
#         recall[cla] = cm[cla, cla] / cm[:, cla].sum()
#         w[cla] = pre[cla] * recall[cla]
#
#     print(cm, pre, recall)
#     return true / np.size(testLabels), prob, t / np.size(testLabels), w
#
#


######## CLASSIFICATION ( ALREADY KNOW THERE IS A GESTURE)
def scoreModel_ClassificationUnsupervised(testFeatures, model, classes, testLabels, testRepetitions, type_DA):
    auxLabel = testLabels[0]
    auxRep = testRepetitions[0]
    actualPredictor = np.zeros(classes)
    actualFeatures = []
    idx = 0
    preTrainedDataMatrix = pd.DataFrame(columns=['mean', 'cov', 'class', 'prob', 'samples'])

    if type_DA == 'LDA':
        LDACov = LDA_Cov(model, classes)
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
    elif type_DA == 'QDA':
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


######## DA classifiers (probabilities)


def predictedModelLDAProb(sample, model, classes, LDACov):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = LDA_Discriminant(sample, LDACov, model['mean'].loc[cl])
        if math.isnan(d[cl]):
            return predictedModelLDA_pseudoProb(sample, model, classes, LDACov)
    d = d - d[np.argmin(d)]
    return d / d.sum()


def predictedModelLDA_pseudoProb(sample, model, classes, LDACov):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = LDA_Discriminant_pseudo(sample, LDACov, model['mean'].loc[cl])
    d = d - d[np.argmin(d)]
    return d / d.sum()


def predictedModelQDAProb(sample, model, classes):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = QDA_Discriminant(sample, model['cov'].loc[cl], model['mean'].loc[cl])
        if math.isnan(d[cl]):
            return predictedModelQDA_pseudoProb(sample, model, classes)
    d = d - d[np.argmin(d)]
    return d / d.sum(), np.argmax(d) + 1


def predictedModelQDA_pseudoProb(sample, model, classes):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = QDA_Discriminant_pseudo(sample, model['cov'].loc[cl], model['mean'].loc[cl])
    d = d - d[np.argmin(d)]
    return d / d.sum(), np.argmax(d) + 1
