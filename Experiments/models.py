# %% Libraries
import time
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
# from scipy.spatial import distance

import DA_classifiers as DA_Classifiers


# %% Our soft-learning technique

def our_soft_labelling_technique(currentValues, personMean, personCov, classes, trainFeatures, trainLabels, type_DA):
    if type_DA == 'LDA':
        weights = []

        for cla in range(classes):
            LDAcov = currentValues.loc[0, 'LDAcov']
            tabDiscriminantValues = []
            if np.linalg.det(LDAcov) > 0:
                invCov = np.linalg.inv(LDAcov)
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
                invCov = np.linalg.pinv(LDAcov)
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

            elif countNaN >= 1:
                personDiscriminantValues, tabDiscriminantValues = discriminantTab(
                    trainFeatures, personMean, personCov, classes, currentValues)
                personPseudoDiscriminantValues, tabPseudoDiscriminantValues = pseudoDiscriminantTab(
                    trainFeatures, personMean, personCov, classes, currentValues)
                return calculationWeight2(determinantsCurrentModel, personPseudoDiscriminantValues,
                                          tabPseudoDiscriminantValues, personDiscriminantValues, tabDiscriminantValues,
                                          classes, trainLabels)
        else:
            personPseudoDiscriminantValues, tabPseudoDiscriminantValues = pseudoDiscriminantTab(
                trainFeatures, personMean, personCov, classes, currentValues)
            return calculationWeight(personPseudoDiscriminantValues, tabPseudoDiscriminantValues, classes, trainLabels)


def mcc(TP, TN, FP, FN):  # Matthews correlation coefficients
    mccValue = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    if np.isscalar(mccValue):
        if np.isnan(mccValue) or mccValue < 0:
            mccValue = 0
    else:
        mccValue[np.isnan(mccValue)] = 0
        mccValue[mccValue < 0] = 0

    return mccValue


def calculationMcc(trueLabels, predeictedLabeles, currentClass):
    vectorCurrentClass = np.ones(len(predeictedLabeles)) * (currentClass + 1)
    TP = len(np.where((trueLabels == vectorCurrentClass) & (trueLabels == predeictedLabeles))[0])
    FN = len(np.where((trueLabels == vectorCurrentClass) & (trueLabels != predeictedLabeles))[0])
    TN = len(np.where((trueLabels != vectorCurrentClass) & (trueLabels == predeictedLabeles))[0])
    FP = len(np.where((trueLabels != vectorCurrentClass) & (trueLabels != predeictedLabeles))[0])

    return mcc(TP, TN, FP, FN)


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


def entrophyVector(vector):
    entrophy = 0
    for i in range(len(vector)):
        if vector[i] == 0:
            entrophy += 0
        else:
            entrophy += vector[i] * math.log(vector[i], len(vector))
    entrophy *= -1
    return (1 - abs(entrophy))


def predicted_labels(features, model, classes, type_DA):
    actualPredictor = np.zeros(classes)
    if type_DA == 'LDA':
        for i in range(np.size(features, axis=0)):
            actualPredictor[
                DA_Classifiers.predictedModelLDA(features[i, :], model, classes, model.loc[0, 'LDAcov']) - 1] += 1
    elif type_DA == 'QDA':
        for i in range(np.size(features, axis=0)):
            actualPredictor[DA_Classifiers.predictedModelQDA(features[i, :], model, classes) - 1] += 1

    return actualPredictor


# %% Updating methods

def updating_state_art(classes, weights, model, chunk_mean, chunk_cov, chunk_N, type_DA):
    for cla in range(classes):
        w = weights[cla]
        if w != 0:
            mean = model.loc[cla, 'mean']
            cov = model.loc[cla, 'cov']
            N = model.loc[cla, 'N']

            model.at[cla, 'mean'] = (N * mean + chunk_N * w * chunk_mean) / \
                                    (N + chunk_N * w)
            model.at[cla, 'N'] = N + chunk_N * w

            aux = np.resize(chunk_mean - mean, (len(chunk_mean), 1))
            model.at[cla, 'cov'] = (1 / (N + chunk_N * w - 1)) * \
                                   (cov * (N - 1) + chunk_cov * w * (chunk_N - 1) +
                                    np.dot(aux, aux.T.conj()) * (N * chunk_N * w) / (N + chunk_N * w))

    if type_DA == 'LDA':
        model.at[0, 'LDAcov'] = DA_Classifiers.LDA_Cov_weights(model)
    return model


def updating_our_proposed(classes, weights, model, chunk_mean, chunk_cov, chunk_N, type_DA):
    for cla in range(classes):
        w = weights[cla]
        if w != 0:
            mean = model.loc[cla, 'mean']
            cov = model.loc[cla, 'cov']
            N = model.loc[cla, 'N']

            model.at[cla, 'mean'] = (N * mean + chunk_N * w * chunk_mean) / \
                                    (N + chunk_N * w)
            model.at[cla, 'N'] = N + chunk_N * w

            # if type_DA == 'QDA':
            model.at[cla, 'cov'] = ((N - 1) * cov + w * (chunk_N - 1) * chunk_cov) / \
                                       (N + chunk_N * w - 1)


    # if type_DA == 'LDA':
    #     LDAcov = model.loc[0, 'LDAcov']
    #     N_total = model.loc[0, 'N_total']
    #     model.at[0, 'LDAcov'] = (LDAcov * (N_total - 1) + (chunk_N - 1) * chunk_cov) / (N_total + chunk_N - 1)
    #     model.at[0, 'N_LDA'] = N_total + chunk_N

    if type_DA == 'LDA':
        model.at[0, 'LDAcov'] = DA_Classifiers.LDA_Cov_weights(model)

    return model


# %% Proposed Online Active learning

def model_ours_soft_labels(currentModel, classes, trainFeatures, labeledGesturesFeatures, labeledGesturesLabels,
                           type_DA, weakModel, gesture_N, gesture_mean, gesture_cov):
    time_weight = time.time()
    postProb_trainFeatures = predicted_labels(trainFeatures, weakModel, classes, type_DA) / len(trainFeatures)

    weightsMcc = np.array(
        our_soft_labelling_technique(weakModel, gesture_mean, gesture_cov, classes, labeledGesturesFeatures,
                                     labeledGesturesLabels, type_DA))

    if weightsMcc.sum() != 0:
        weightsMcc_norm = weightsMcc / weightsMcc.sum()
    else:
        weightsMcc_norm = weightsMcc.copy()

    weightsUnlabeledGesture = ((postProb_trainFeatures * entrophyVector(postProb_trainFeatures)) + (
            weightsMcc_norm * entrophyVector(weightsMcc_norm))) / 2
    time_weight = time.time() - time_weight

    time_update = time.time()
    model = updating_our_proposed(classes, weightsUnlabeledGesture, currentModel, gesture_mean, gesture_cov,
                                  gesture_N, type_DA)
    time_update = time.time() - time_update

    return model, time_update, time_weight, weightsUnlabeledGesture


def model_ours_labels(currentModel, classes, trainLabel, type_DA, gesture_N, gesture_mean, gesture_cov):
    weightsUnlabeledGesture = np.zeros(classes)
    weightsUnlabeledGesture[trainLabel - 1] = 1
    time_update = time.time()
    model = updating_our_proposed(classes, weightsUnlabeledGesture, currentModel, gesture_mean, gesture_cov,
                                  gesture_N, type_DA)
    time_update = time.time() - time_update
    return model, time_update, weightsUnlabeledGesture


def model_ours_probabilities(currentModel, classes, trainFeatures, type_DA, weakModel, gesture_N, gesture_mean,
                             gesture_cov, weight):
    time_weight = time.time()
    postProb_trainFeatures = predicted_labels(trainFeatures, weakModel, classes, type_DA) / len(trainFeatures)
    weightsUnlabeledGesture = postProb_trainFeatures * weight
    time_weight = time.time() - time_weight
    return updating_our_proposed(classes, weightsUnlabeledGesture, currentModel, gesture_mean, gesture_cov,
                                 gesture_N, type_DA), time_weight, weightsUnlabeledGesture


def model_ours_threshold(currentModel, classes, trainFeatures, type_DA, weakModel, gesture_N, gesture_mean, gesture_cov,
                         threshold):

    time_weight = time.time()
    postProb_trainFeatures = predicted_labels(trainFeatures, weakModel, classes, type_DA) / len(trainFeatures)
    time_weight = time.time() - time_weight
    if np.max(postProb_trainFeatures) > threshold:
        weightsUnlabeledGesture = np.zeros(classes)
        weightsUnlabeledGesture[np.argmax(postProb_trainFeatures)] = 1
        return updating_our_proposed(classes, weightsUnlabeledGesture, currentModel, gesture_mean, gesture_cov,
                                     gesture_N, type_DA), time_weight, weightsUnlabeledGesture
    else:
        return currentModel, time_weight, np.zeros(classes)


# %% online learning in the state of the art

def model_state_art_soft_labels(currentModel, classes, trainFeatures, labeledGesturesFeatures,
                                labeledGesturesLabels, type_DA, weakModel, gesture_N, gesture_mean, gesture_cov):
    time_weight = time.time()
    postProb_trainFeatures = predicted_labels(trainFeatures, weakModel, classes, type_DA) / len(trainFeatures)

    weightsMcc = np.array(
        our_soft_labelling_technique(weakModel, gesture_mean, gesture_cov, classes, labeledGesturesFeatures,
                                     labeledGesturesLabels, type_DA))

    if weightsMcc.sum() != 0:
        weightsMcc_norm = weightsMcc / weightsMcc.sum()
    else:
        weightsMcc_norm = weightsMcc.copy()

    weightsUnlabeledGesture = ((postProb_trainFeatures * entrophyVector(postProb_trainFeatures)) + (
            weightsMcc_norm * entrophyVector(weightsMcc_norm))) / 2
    time_weight = time.time() - time_weight

    time_update = time.time()
    model = updating_state_art(classes, weightsUnlabeledGesture, currentModel, gesture_mean, gesture_cov,
                               gesture_N, type_DA)
    time_update = time.time() - time_update

    return model, time_update, time_weight, weightsUnlabeledGesture


def model_state_art_labels(currentModel, classes, trainLabel, type_DA, gesture_N, gesture_mean, gesture_cov):
    weightsUnlabeledGesture = np.zeros(classes)
    weightsUnlabeledGesture[trainLabel - 1] = 1
    time_update = time.time()
    model = updating_state_art(classes, weightsUnlabeledGesture, currentModel, gesture_mean, gesture_cov,
                               gesture_N, type_DA)
    time_update = time.time() - time_update

    return model, time_update, weightsUnlabeledGesture


def model_state_art_probabilities(currentModel, classes, trainFeatures, type_DA, weakModel, gesture_N, gesture_mean,
                                  gesture_cov, weight):
    time_weight = time.time()
    postProb_trainFeatures = predicted_labels(trainFeatures, weakModel, classes, type_DA) / len(trainFeatures)
    weightsUnlabeledGesture = postProb_trainFeatures * weight
    time_weight = time.time() - time_weight

    return updating_state_art(classes, weightsUnlabeledGesture, currentModel, gesture_mean, gesture_cov,
                              gesture_N, type_DA), time_weight, weightsUnlabeledGesture


def model_state_art_threshold(currentModel, classes, trainFeatures, type_DA, weakModel, gesture_N, gesture_mean,
                              gesture_cov, threshold):

    time_weight = time.time()
    postProb_trainFeatures = predicted_labels(trainFeatures, weakModel, classes, type_DA) / len(trainFeatures)
    time_weight = time.time() - time_weight
    if np.max(postProb_trainFeatures) > threshold:
        weightsUnlabeledGesture = np.zeros(classes)
        weightsUnlabeledGesture[np.argmax(postProb_trainFeatures)] = 1
        return updating_state_art(classes, weightsUnlabeledGesture, currentModel, gesture_mean, gesture_cov,
                                  gesture_N, type_DA), time_weight, weightsUnlabeledGesture
    else:
        return currentModel, time_weight, np.zeros(classes)


# %% Errors
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


def errorWeights_type2_mse(predictedWeight, observedWeight):
    aux = np.square(predictedWeight - observedWeight)
    aux = aux[aux > 0]
    return aux.sum() / len(observedWeight)


def errorWeights_type1_mse(predictedWeight, observedWeight):
    aux = np.square(observedWeight - predictedWeight)
    aux = aux[aux > 0]
    return aux.sum() / len(observedWeight)
