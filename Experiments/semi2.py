# %% Libraries
import time
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
# from scipy.spatial import distance

import DA_Classifiers as DA_Classifiers
import DA_BasedAdaptiveModels as DA_BasedAdaptiveModels


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


# %% Models
def updateModel(classes, weights, model, chunk_mean, chunk_cov, chunk_N, type_DA):
    for cla in range(classes):
        mean = model.loc[cla, 'mean']
        cov = model.loc[cla, 'cov']
        N = model.loc[cla, 'N']
        w = weights[cla]

        model.at[cla, 'mean'] = (N * mean + chunk_N * w * chunk_mean) / \
                                (N + chunk_N * w)
        aux = np.resize(chunk_mean - mean, (len(chunk_mean), 1))
        model.at[cla, 'cov'] = (1 / (N + chunk_N * w - 1)) * \
                               (cov * (N - 1) + chunk_cov * w * (chunk_N - 1) +
                                np.dot(aux, aux.T.conj()) * (N * chunk_N * w) / (N + chunk_N * w))
        model.at[cla, 'N'] = N + chunk_N * w
    if type_DA == 'LDA':
        model.at[0, 'LDAcov'] = DA_Classifiers.LDA_Cov_weights(model)
    return model

def updateModel2(classes, weights, model, chunk_mean, chunk_cov, chunk_N, type_DA):
    for cla in range(classes):
        mean = model.loc[cla, 'mean']
        cov = model.loc[cla, 'cov']
        N = model.loc[cla, 'N']
        w = weights[cla]

        model.at[cla, 'mean'] = (N * mean + chunk_N * w * chunk_mean) / \
                                (N + chunk_N * w)
        # aux = np.resize(chunk_mean - mean, (len(chunk_mean), 1))
        # model.at[cla, 'cov'] = (1 / (N + chunk_N * w - 1)) * \
        #                        (cov * (N - 1) + chunk_cov * w * (chunk_N - 1) +
        #                         np.dot(aux, aux.T.conj()) * (N * chunk_N * w) / (N + chunk_N * w))
        model.at[cla, 'cov'] = (N * cov + chunk_N * w * chunk_cov) / \
                                (N + chunk_N * w)
        model.at[cla, 'N'] = N + chunk_N * w
    if type_DA == 'LDA':
        model.at[0, 'LDAcov'] = DA_Classifiers.LDA_Cov_weights(model)
    return model


def post_probabilities_Calculation(features, model, classes, type_DA):
    actualPredictor = np.zeros(classes)
    if type_DA == 'LDA':
        for i in range(np.size(features, axis=0)):
            actualPredictor[
                DA_Classifiers.predictedModelLDA(features[i, :], model, classes, model.loc[0, 'LDAcov']) - 1] += 1
    elif type_DA == 'QDA':
        for i in range(np.size(features, axis=0)):
            actualPredictor[DA_Classifiers.predictedModelQDA(features[i, :], model, classes) - 1] += 1

    return actualPredictor / actualPredictor.sum()


def entrophyVector(vector):
    entrophy = 0
    for i in range(len(vector)):
        if vector[i] == 0:
            entrophy += 0
        else:
            entrophy += vector[i] * math.log(vector[i], len(vector))
    entrophy *= -1
    return (1 - abs(entrophy))


def model_incre_proposedProbMSDA(currentModel, classes, trainFeatures, labeledGesturesFeatures, labeledGesturesLabels,
                                 type_DA):
    t = time.time()

    gesture_mean = np.mean(trainFeatures, axis=0)
    gesture_cov = np.cov(trainFeatures, rowvar=False)
    gesture_N = np.size(trainFeatures, axis=0)

    weightsMcc = np.array(weightMSDA_reduce(currentModel, gesture_mean, gesture_cov, classes, labeledGesturesFeatures,
                                            labeledGesturesLabels, type_DA))

    postProb_trainFeatures = post_probabilities_Calculation(trainFeatures, currentModel, classes, type_DA)

    if weightsMcc.sum() != 0:
        weightsMcc_norm = weightsMcc / weightsMcc.sum()
    else:
        weightsMcc_norm = weightsMcc.copy()

    weightsUnlabeledGesture = ((postProb_trainFeatures * entrophyVector(postProb_trainFeatures)) + (
            weightsMcc_norm * entrophyVector(weightsMcc_norm))) / 2

    return updateModel2(classes, weightsUnlabeledGesture, currentModel, gesture_mean, gesture_cov, gesture_N, type_DA), \
           time.time() - t, weightsUnlabeledGesture


def model_incre_proposedNigam(currentModel, classes, trainFeatures, type_DA, weight_Nigam):
    t = time.time()
    gesture_mean = np.mean(trainFeatures, axis=0)
    gesture_cov = np.cov(trainFeatures, rowvar=False)
    gesture_N = np.size(trainFeatures, axis=0)

    weightsUnlabeledGesture = post_probabilities_Calculation(trainFeatures, currentModel, classes,
                                                             type_DA) * weight_Nigam

    return updateModel2(classes, weightsUnlabeledGesture, currentModel, gesture_mean, gesture_cov, gesture_N, type_DA), \
           time.time() - t, weightsUnlabeledGesture


def model_incre_proposedLabel(currentModel, classes, trainFeatures, type_DA):
    t = time.time()
    gesture_mean = np.mean(trainFeatures, axis=0)
    gesture_cov = np.cov(trainFeatures, rowvar=False)
    gesture_N = np.size(trainFeatures, axis=0)

    weightsUnlabeledGesture = np.zeros(classes)
    weightsUnlabeledGesture[
        np.argmax(post_probabilities_Calculation(trainFeatures, currentModel, classes, type_DA))] = 1

    return updateModel2(classes, weightsUnlabeledGesture, currentModel, gesture_mean, gesture_cov, gesture_N, type_DA), \
           time.time() - t, weightsUnlabeledGesture


def model_incre_sequential_labels(currentModel, classes, trainFeatures, type_DA):
    t = time.time()

    for sample in range(len(trainFeatures[:, 0])):
        x = trainFeatures[sample, :]
        if type_DA == 'LDA':
            cla = DA_Classifiers.predictedModelLDA(x, currentModel, classes, currentModel.loc[0, 'LDAcov'])
        elif type_DA == 'QDA':
            cla = DA_Classifiers.predictedModelQDA(x, currentModel, classes)

        weightsUnlabeledGesture = np.zeros(classes)
        weightsUnlabeledGesture[cla - 1] = 1
        gesture_cov = 0
        gesture_N = 1
        currentModel = updateModel2(classes, weightsUnlabeledGesture, currentModel, x, gesture_cov, gesture_N, type_DA)

    return currentModel, time.time() - t, 0


def model_incre_supervised(currentModel, classes, trainFeatures, type_DA, cla):
    t = time.time()
    gesture_mean = np.mean(trainFeatures, axis=0)
    gesture_cov = np.cov(trainFeatures, rowvar=False)
    gesture_N = np.size(trainFeatures, axis=0)

    weightsUnlabeledGesture = np.zeros(classes)
    weightsUnlabeledGesture[cla - 1] = 1

    return updateModel2(classes, weightsUnlabeledGesture, currentModel, gesture_mean, gesture_cov, gesture_N, type_DA), \
           time.time() - t, weightsUnlabeledGesture


def RMSE(predictedWeight, observedWeight):
    aux = np.square(observedWeight - predictedWeight)
    return np.sqrt(aux.sum() / len(aux))


def MSE(predictedWeight, observedWeight):
    aux = np.square(observedWeight - predictedWeight)
    return aux.sum() / len(aux)


def MAE(predictedWeight, observedWeight):
    aux = np.absolute(observedWeight - predictedWeight)
    return aux.sum() / len(aux)


def errorWeights_type2(predictedWeight, observedWeight):
    aux = predictedWeight - observedWeight
    aux = aux[aux > 0]
    return aux.sum() / len(observedWeight)


def errorWeights_type1(predictedWeight, observedWeight):
    aux = observedWeight - predictedWeight
    aux = aux[aux > 0]
    return aux.sum() / len(observedWeight)


def calculationMcc(trueLabels, predeictedLabeles, currentClass):
    vectorCurrentClass = np.ones(len(predeictedLabeles)) * (currentClass + 1)
    TP = len(np.where((trueLabels == vectorCurrentClass) & (trueLabels == predeictedLabeles))[0])
    FN = len(np.where((trueLabels == vectorCurrentClass) & (trueLabels != predeictedLabeles))[0])
    TN = len(np.where((trueLabels != vectorCurrentClass) & (trueLabels == predeictedLabeles))[0])
    FP = len(np.where((trueLabels != vectorCurrentClass) & (trueLabels != predeictedLabeles))[0])
    return mcc(TP, TN, FP, FN)
    # return TP / (TP + 0.5 * (FP + FN))


def discriminantTab(trainFeatures, personMean, personCov, classes, currentValues):
    tabDiscriminantValues = []
    det = np.linalg.det(personCov)
    personDiscriminantValues = np.array([-.5 * np.log(det) - .5 * np.dot(
        np.dot((trainFeatures[a, :] - personMean), np.linalg.inv(personCov)),
        (trainFeatures[a, :] - personMean).T) for a in range(len(trainFeatures))])
    for cla in range(classes):
        covariance = currentValues['cov'].at[cla]
        mean = currentValues['mean'].at[cla]
        det = np.linalg.det(covariance)
        tabDiscriminantValues.append([-.5 * np.log(det) - .5 * np.dot(
            np.dot((trainFeatures[a, :] - mean), np.linalg.inv(covariance)), (trainFeatures[a, :] - mean).T)
                                      for a in range(len(trainFeatures))])
    tabDiscriminantValues = np.array(tabDiscriminantValues)
    return personDiscriminantValues, tabDiscriminantValues


def pseudoDiscriminantTab(trainFeatures, personMean, personCov, classes, currentValues):
    tabPseudoDiscriminantValues = []
    personPseudoDiscriminantValues = np.array([- .5 * np.dot(
        np.dot((trainFeatures[a, :] - personMean), np.linalg.pinv(personCov)),
        (trainFeatures[a, :] - personMean).T) for a in range(len(trainFeatures))])
    for cla in range(classes):
        covariance = currentValues['cov'].at[cla]
        mean = currentValues['mean'].at[cla]
        tabPseudoDiscriminantValues.append([- .5 * np.dot(
            np.dot((trainFeatures[a, :] - mean), np.linalg.pinv(covariance)), (trainFeatures[a, :] - mean).T)
                                            for a in range(len(trainFeatures))])
    tabPseudoDiscriminantValues = np.array(tabPseudoDiscriminantValues)
    return personPseudoDiscriminantValues, tabPseudoDiscriminantValues


def calculationWeight(personDiscriminantValues, tabDiscriminantValues, classes, trainLabels):
    weights = []
    for cla in range(classes):
        auxTab = tabDiscriminantValues.copy()
        auxTab[cla, :] = personDiscriminantValues.copy()
        weights.append(calculationMcc(trainLabels, np.argmax(auxTab, axis=0) + 1, cla))
    return weights


def calculationWeight2(determinantsCurrentModel, personPseudoDiscriminantValues, tabPseudoDiscriminantValues,
                       personDiscriminantValues, tabDiscriminantValues, classes, trainLabels):
    weights = []
    for cla in range(classes):
        if determinantsCurrentModel[cla] == float('NaN'):
            auxTab = tabPseudoDiscriminantValues.copy()
            auxTab[cla, :] = personPseudoDiscriminantValues.copy()
        else:
            auxTab = tabDiscriminantValues.copy()
            auxTab[cla, :] = personDiscriminantValues.copy()

        weights.append(calculationMcc(trainLabels, np.argmax(auxTab, axis=0) + 1, cla))
    return weights


def weightMSDA_reduce(currentValues, personMean, personCov, classes, trainFeatures, trainLabels, type_DA):
    if type_DA == 'LDA':
        weights = []

        for cla in range(classes):
            auxCurrentValues = currentValues.copy()
            auxCurrentValues['cov'].at[cla] = personCov
            LDACov = DA_Classifiers.LDA_Cov(auxCurrentValues, classes)

            tabDiscriminantValues = []
            if np.linalg.det(LDACov) > 0:
                invCov = np.linalg.inv(LDACov)
                for cla2 in range(classes):
                    if cla == cla2:
                        tabDiscriminantValues.append(list(
                            np.dot(np.dot(trainFeatures, invCov), personMean) - 0.5 * np.dot(np.dot(personMean, invCov),
                                                                                             personMean)))
                    else:
                        mean = currentValues['mean'].at[cla2]
                        tabDiscriminantValues.append(list(
                            np.dot(np.dot(trainFeatures, invCov), mean) - 0.5 * np.dot(np.dot(mean, invCov), mean)))
            else:
                invCov = np.linalg.pinv(LDACov)
                for cla2 in range(classes):
                    if cla == cla2:
                        tabDiscriminantValues.append(list(
                            np.dot(np.dot(trainFeatures, invCov), personMean) - 0.5 * np.dot(np.dot(personMean, invCov),
                                                                                             personMean)))
                    else:
                        mean = currentValues['mean'].at[cla2]
                        tabDiscriminantValues.append(list(
                            np.dot(np.dot(trainFeatures, invCov), mean) - 0.5 * np.dot(np.dot(mean, invCov), mean)))

            weights.append(calculationMcc(trainLabels, np.argmax(np.array(tabDiscriminantValues), axis=0) + 1, cla))
        return weights

    elif type_DA == 'QDA':

        if np.linalg.det(personCov) > 0:
            determinantsCurrentModel = []
            for cla in range(classes):
                det = np.linalg.det(currentValues['cov'].at[cla])
                if det > 0:
                    determinantsCurrentModel.append(det)
                else:
                    determinantsCurrentModel.append(float('NaN'))
            countNaN = np.count_nonzero(np.isnan(np.array(determinantsCurrentModel)))
            if countNaN == 0:
                personDiscriminantValues, tabDiscriminantValues = discriminantTab(
                    trainFeatures, personMean, personCov, classes, currentValues)
                return calculationWeight(personDiscriminantValues, tabDiscriminantValues, classes, trainLabels)

            elif countNaN == 1:
                personDiscriminantValues, tabDiscriminantValues = discriminantTab(
                    trainFeatures, personMean, personCov, classes, currentValues)
                personPseudoDiscriminantValues, tabPseudoDiscriminantValues = pseudoDiscriminantTab(
                    trainFeatures, personMean, personCov, classes, currentValues)
                return calculationWeight2(determinantsCurrentModel, personPseudoDiscriminantValues,
                                          tabPseudoDiscriminantValues, personDiscriminantValues, tabDiscriminantValues,
                                          classes, trainLabels)
            elif countNaN >= 2:
                personPseudoDiscriminantValues, tabPseudoDiscriminantValues = pseudoDiscriminantTab(
                    trainFeatures, personMean, personCov, classes, currentValues)
                return calculationWeight(personPseudoDiscriminantValues, tabPseudoDiscriminantValues, classes,
                                         trainLabels)
        else:
            personPseudoDiscriminantValues, tabPseudoDiscriminantValues = pseudoDiscriminantTab(
                trainFeatures, personMean, personCov, classes, currentValues)
            return calculationWeight(personPseudoDiscriminantValues, tabPseudoDiscriminantValues, classes, trainLabels)
