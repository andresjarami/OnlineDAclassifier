import math

import numpy as np


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


def LDA_Cov(trainedModel):
    classes = trainedModel.shape[0]
    LDACov = trainedModel['cov'].sum() / classes
    return LDACov

def LDA_Cov_weights(trainedModel):
    classes = trainedModel.shape[0]
    sumCov = np.sum(trainedModel['cov'] * (trainedModel['N'] - 1))
    LDACov = sumCov / (trainedModel['N'].sum() - classes)
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
    true = 0
    # LDACov = LDA_Cov(model, classes)
    LDACov = model.loc[0, 'LDAcov']
    precision = np.zeros((2, classes))
    recall = np.zeros((2, classes))
    for i in range(np.size(testLabels)):
        currentPredictor = predictedModelLDA(testFeatures[i, :], model, classes, LDACov)
        if currentPredictor == testLabels[i]:
            true += 1
            recall[0, int(testLabels[i] - 1)] += 1
            precision[0, int(currentPredictor - 1)] += 1
        else:
            recall[1, int(testLabels[i] - 1)] += 1
            precision[1, int(currentPredictor - 1)] += 1

    return true / np.size(testLabels), np.nan_to_num(precision[0, :] / precision.sum(axis=0)), np.nan_to_num(
        recall[0, :] / recall.sum(axis=0))


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
    true = 0
    precision = np.zeros((2, classes))
    recall = np.zeros((2, classes))
    for i in range(np.size(testLabels)):
        currentPredictor = predictedModelQDA(testFeatures[i, :], model, classes)
        if currentPredictor == testLabels[i]:
            true += 1
            recall[0, int(testLabels[i] - 1)] += 1
            precision[0, int(currentPredictor - 1)] += 1
        else:
            recall[1, int(testLabels[i] - 1)] += 1
            precision[1, int(currentPredictor - 1)] += 1
    return true / np.size(testLabels), np.nan_to_num(precision[0, :] / precision.sum(axis=0)), np.nan_to_num(
        recall[0, :] / recall.sum(axis=0))




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

def covariance(data,Features,mean):
    cov = np.zeros((Features, Features))
    for i in range(len(data)):
        x_mean = np.resize(data[i, :] - mean, (len(data[i, :]), 1))
        cov += np.dot(x_mean, x_mean.T.conj())
    return cov/(len(data) - 1)

def mean(data,Features):
    mean = np.zeros(Features)
    for i in range(len(data)):
        mean += data[i, :]
    return mean/(len(data))
