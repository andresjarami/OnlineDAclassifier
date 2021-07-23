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


# %% Proposed Updating

def updating_proposed(classes, weights, model, chunk_mean, chunk_cov, chunk_N, type_DA):
    for cla in range(classes):
        w = weights[cla]
        if w != 0:
            mean = model.loc[cla, 'mean']
            cov = model.loc[cla, 'cov']
            N = model.loc[cla, 'N']

            model.at[cla, 'mean'] = (N * mean + chunk_N * w * chunk_mean) / \
                                    (N + chunk_N * w)

            #####1
            aux = np.resize(chunk_mean - mean, (len(chunk_mean), 1))
            model.at[cla, 'cov'] = (1 / (N + chunk_N * w - 1)) * \
                                   (cov * (N - 1) + chunk_cov * w * (chunk_N - 1) +
                                    np.dot(aux, aux.T.conj()) * (N * chunk_N * w) / (N + chunk_N * w))

            ######3
            # model.at[cla, 'cov'] = (N * cov + chunk_N * w * chunk_cov) / \
            #                        (N + chunk_N * w)

            model.at[cla, 'N'] = N + chunk_N * w
    if type_DA == 'LDA':
        model.at[0, 'LDAcov'] = DA_Classifiers.LDA_Cov_weights(model)
    return model


# %% predicted labels (posterior probability)
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


# %% posterior probability
# def post_probabilities2(features, model, classes, type_DA):
#     actualPredictor = np.zeros(classes)
#     if type_DA == 'LDA':
#         for i in range(np.size(features, axis=0)):
#             actualPredictor[
#                 DA_Classifiers.predictedModelLDA(features[i, :], model, classes, model.loc[0, 'LDAcov']) - 1] += 1
#     elif type_DA == 'QDA':
#         for i in range(np.size(features, axis=0)):
#             actualPredictor[DA_Classifiers.predictedModelQDA(features[i, :], model, classes) - 1] += 1
#
#     return actualPredictor
#
#
# %% Proposed model using our Weight
#
#
# def KLdivergence(mean0, mean1, k, cov0, cov1):
#     exp1 = np.trace(np.dot(np.linalg.inv(cov1), cov0))
#     exp2 = np.dot(np.dot((mean1 - mean0).T, np.linalg.inv(cov1)), (mean1 - mean0))
#     exp3 = np.log(np.linalg.det(cov1) / np.linalg.det(cov0))
#     return 0.5 * (exp1 + exp2 - k + exp3)
#
#
# def JSdivergence(mean0, cov0, mean1, cov1):
#     k = len(mean0)
#     meanM = (mean0 + mean1) / 2
#     covM = (cov0 + cov1) / 2
#     js = KLdivergence(mean0, meanM, k, cov0, covM) + KLdivergence(mean1, meanM, k, cov1, covM)
#     # js /= np.log(2)
#     return js / 2


# def model_incre_proposed(currentModel, classes, trainFeatures, labeledGesturesFeatures, labeledGesturesLabels,
#                          type_DA):
#     t = time.time()
#
#     gesture_mean = np.mean(trainFeatures, axis=0)
#     gesture_cov = np.cov(trainFeatures, rowvar=False)
#     gesture_N = np.size(trainFeatures, axis=0)
#
#     postProb_trainFeatures = post_probabilities(trainFeatures, currentModel, classes, type_DA)
#
#     weightsMcc4 = np.zeros(classes)
#     for cla in range(classes):
#         auxModel = currentModel.copy()
#         auxModel['cov'].at[cla] = gesture_cov
#         auxModel['mean'].at[cla] = gesture_mean
#         auxVect = np.zeros((classes, classes))
#         TP = 0
#         TN = 0
#         FP = 0
#         FN = 0
#         for cla2 in range(classes):
#             div = []
#             for cla3 in range(classes):
#                 div.append(1 / JSdivergence(
#                     auxModel['mean'].loc[cla3], auxModel['cov'].loc[cla3],
#                     np.mean(labeledGesturesFeatures[labeledGesturesLabels == cla2 + 1], axis=0),
#                     np.cov(labeledGesturesFeatures[labeledGesturesLabels == cla2 + 1], rowvar=False)))
#             div = np.array(div)
#             div /= np.sum(div)
#             if cla == cla2:
#                 TP += div[cla2]
#                 FN += 1 - div[cla2]
#             else:
#                 TN += div[cla2]
#                 FP += 1 - div[cla2]
#             # auxVect[cla2,:] = div
#         weightsMcc4[cla] = mcc(TP, TN, FP, FN)
#
#         #     if np.argmin(div) == cla2:
#         #         auxVect[cla2] = 1
#         # if auxVect[cla] == 1:
#         #     weightsMcc4[cla] = np.count_nonzero(auxVect) / classes
#
#     # weightsMcc2 = np.zeros(classes)
#     # weightsMcc3 = np.zeros(classes)
#     # for cla in range(classes):
#     #     if postProb_trainFeatures[cla] != 0:
#     #         x = labeledGesturesFeatures[labeledGesturesLabels == cla + 1, :]
#     #         x1= labeledGesturesFeatures[labeledGesturesLabels != cla + 1, :]
#     #         auxModel = currentModel.copy()
#     #         auxModel['cov'].at[cla] = gesture_cov
#     #         auxModel['mean'].at[cla] = gesture_mean
#     #         if type_DA == 'LDA':
#     #             auxModel.at[0, 'LDAcov'] = DA_Classifiers.LDA_Cov_weights(auxModel)
#     #         y = post_probabilities2(x, auxModel, classes, type_DA)
#     #         y1 = post_probabilities2(x1, auxModel, classes, type_DA)
#     #         weightsMcc2[cla] = y[cla]/len(x1)
#     #         TP = y[cla]
#     #         FN = np.sum(y) - TP
#     #         FP = y1[cla]
#     #         weightsMcc3[cla] = 2*TP/(2*TP+FP+FN)
#
#     # weightsMcc1 = np.array(weightMSDA_reduce(currentModel, gesture_mean, gesture_cov, classes, labeledGesturesFeatures,
#     #                                          labeledGesturesLabels, type_DA))
#
#     # print(type_DA)
#     # print(postProb_trainFeatures)
#     # print(weightsMcc1)
#     # print(weightsMcc2)
#     # print(weightsMcc3)
#     # print(weightsMcc4)
#
#     weightsMcc = weightsMcc4
#
#     if weightsMcc.sum() != 0:
#         weightsMcc_norm = weightsMcc / weightsMcc.sum()
#     else:
#         weightsMcc_norm = weightsMcc.copy()
#
#     weightsUnlabeledGesture = ((postProb_trainFeatures * entrophyVector(postProb_trainFeatures)) + (
#             weightsMcc_norm * entrophyVector(weightsMcc_norm))) / 2
#
#     return updating_proposed(classes, weightsUnlabeledGesture, currentModel.copy(), gesture_mean, gesture_cov,
#                              gesture_N, type_DA), time.time() - t, weightsUnlabeledGesture


def model_incre_proposed(currentModel, classes, trainFeatures, trainLabel, labeledGesturesFeatures,
                         labeledGesturesLabels, type_DA):
    predictedLabels = predicted_labels(trainFeatures, currentModel, classes, type_DA)
    for cla in range(classes):
        currentModel.at[trainLabel - 1, 'CL' + str(cla + 1)] += predictedLabels[cla]
    postProb_trainFeatures = predictedLabels / predictedLabels.sum()
    t = time.time()

    gesture_mean = np.mean(trainFeatures, axis=0)
    gesture_cov = np.cov(trainFeatures, rowvar=False)
    gesture_N = np.size(trainFeatures, axis=0)

    weightsMcc = np.array(weightMSDA_reduce(currentModel, gesture_mean, gesture_cov, classes, labeledGesturesFeatures,
                                            labeledGesturesLabels, type_DA))

    # print('\n', type_DA)
    # print(postProb_trainFeatures)
    # print(weightsMcc)

    if weightsMcc.sum() != 0:
        weightsMcc_norm = weightsMcc / weightsMcc.sum()
    else:
        weightsMcc_norm = weightsMcc.copy()

    weightsUnlabeledGesture = ((postProb_trainFeatures * entrophyVector(postProb_trainFeatures)) + (
            weightsMcc_norm * entrophyVector(weightsMcc_norm))) / 2

    return updating_proposed(classes, weightsUnlabeledGesture, currentModel.copy(), gesture_mean, gesture_cov,
                             gesture_N, type_DA), time.time() - t, weightsUnlabeledGesture


#

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


def weightMSDA_reduce(currentValues, personMean, personCov, classes, trainFeatures, trainLabels, type_DA):
    if type_DA == 'LDA':
        weights = []

        for cla in range(classes):
            auxCurrentValues = currentValues.copy()
            auxCurrentValues['cov'].at[cla] = personCov
            LDACov = DA_Classifiers.LDA_Cov_weights(auxCurrentValues)

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


def entrophyVector(vector):
    entrophy = 0
    for i in range(len(vector)):
        if vector[i] == 0:
            entrophy += 0
        else:
            entrophy += vector[i] * math.log(vector[i], len(vector))
    entrophy *= -1
    return (1 - abs(entrophy))


# %% model using Nigam weight
def model_incre_weight_Nigam(currentModel, classes, trainFeatures, trainLabel, type_DA, weight_Nigam):
    predictedLabels = predicted_labels(trainFeatures, currentModel, classes, type_DA)
    for cla in range(classes):
        currentModel.at[trainLabel - 1, 'CL' + str(cla + 1)] += predictedLabels[cla]
    postProb_trainFeatures = predictedLabels / predictedLabels.sum()
    t = time.time()
    gesture_mean = np.mean(trainFeatures, axis=0)
    gesture_cov = np.cov(trainFeatures, rowvar=False)
    gesture_N = np.size(trainFeatures, axis=0)

    weightsUnlabeledGesture = postProb_trainFeatures * weight_Nigam

    return updating_proposed(classes, weightsUnlabeledGesture, currentModel, gesture_mean, gesture_cov, gesture_N,
                             type_DA), time.time() - t, weightsUnlabeledGesture


# %% model using the pseudo labeled samples (from posterior probabilities) to update itself according a threshold
def model_incre_label_threshold(currentModel, classes, trainFeatures, trainLabel, type_DA, threshold):
    predictedLabels = predicted_labels(trainFeatures, currentModel, classes, type_DA)
    for cla in range(classes):
        currentModel.at[trainLabel - 1, 'CL' + str(cla + 1)] += predictedLabels[cla]
    postProb_trainFeatures = predictedLabels / predictedLabels.sum()

    t = time.time()
    if np.max(postProb_trainFeatures) > threshold:
        gesture_mean = np.mean(trainFeatures, axis=0)
        gesture_cov = np.cov(trainFeatures, rowvar=False)
        gesture_N = np.size(trainFeatures, axis=0)
        weightsUnlabeledGesture = np.zeros(classes)
        weightsUnlabeledGesture[np.argmax(postProb_trainFeatures)] = 1
        return updating_proposed(classes, weightsUnlabeledGesture, currentModel, gesture_mean, gesture_cov, gesture_N,
                                 type_DA), time.time() - t, weightsUnlabeledGesture
    else:
        return currentModel, time.time() - t, np.zeros(classes)


# %% model using the pseudo labeled samples (from posterior probabilities) to update itself

def model_incre_labels(currentModel, classes, trainFeatures, trainLabel, type_DA):
    predictedLabels = predicted_labels(trainFeatures, currentModel, classes, type_DA)
    for cla in range(classes):
        currentModel.at[trainLabel - 1, 'CL' + str(cla + 1)] += predictedLabels[cla]
    postProb_trainFeatures = predictedLabels / predictedLabels.sum()
    t = time.time()
    gesture_mean = np.mean(trainFeatures, axis=0)
    gesture_cov = np.cov(trainFeatures, rowvar=False)
    gesture_N = np.size(trainFeatures, axis=0)
    weightsUnlabeledGesture = np.zeros(classes)
    weightsUnlabeledGesture[np.argmax(postProb_trainFeatures)] = 1
    return updating_proposed(classes, weightsUnlabeledGesture, currentModel, gesture_mean, gesture_cov, gesture_N,
                             type_DA), time.time() - t, weightsUnlabeledGesture


# %% sequential model (unlike the other that are chunk models) using the pseudo labeled samples (from posterior probabilities) to update itself
def model_incre_sequential_labels(currentModel, classes, trainFeatures, trainLabel, type_DA):
    for sample in range(len(trainFeatures[:, 0])):
        x = trainFeatures[sample, :]
        if type_DA == 'LDA':
            cla = DA_Classifiers.predictedModelLDA(x, currentModel, classes, currentModel.loc[0, 'LDAcov'])
        elif type_DA == 'QDA':
            cla = DA_Classifiers.predictedModelQDA(x, currentModel, classes)

        currentModel.at[trainLabel - 1, 'CL' + str(cla)] += 1

        weightsUnlabeledGesture = np.zeros(classes)
        weightsUnlabeledGesture[cla - 1] = 1
        gesture_cov = 0
        gesture_N = 1
        currentModel = updating_proposed(classes, weightsUnlabeledGesture, currentModel, x, gesture_cov, gesture_N,
                                         type_DA)
    # there are a weight for each sample in the chunk
    # the time is the same as model_incre_supervised
    return currentModel, 0, 0


# %% model using the labeled samples to update itself (supervised incremental model)
def model_incre_supervised(currentModel, classes, trainFeatures, trainLabel, type_DA):
    predictedLabels = predicted_labels(trainFeatures, currentModel, classes, type_DA)
    for cla in range(classes):
        currentModel.at[trainLabel - 1, 'CL' + str(cla + 1)] += predictedLabels[cla]
    t = time.time()
    gesture_mean = np.mean(trainFeatures, axis=0)
    gesture_cov = np.cov(trainFeatures, rowvar=False)
    gesture_N = np.size(trainFeatures, axis=0)

    weightsUnlabeledGesture = np.zeros(classes)
    weightsUnlabeledGesture[trainLabel - 1] = 1

    return updating_proposed(classes, weightsUnlabeledGesture, currentModel, gesture_mean, gesture_cov, gesture_N,
                             type_DA), time.time() - t, weightsUnlabeledGesture


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
