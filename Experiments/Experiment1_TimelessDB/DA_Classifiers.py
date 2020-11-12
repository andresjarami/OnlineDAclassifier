import math
import time

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import confusion_matrix


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
