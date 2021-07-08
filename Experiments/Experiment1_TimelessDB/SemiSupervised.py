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
# def updateModel(classes, weights, model, chunk_mean, chunk_cov, chunk_N, type_DA):
#     for cla in range(classes):
#         mean = model.loc[cla, 'mean']
#         cov = model.loc[cla, 'cov']
#         N = model.loc[cla, 'N']
#         w = weights[cla]
#
#         model.at[cla, 'mean'] = (N * mean + chunk_N * w * chunk_mean) / \
#                                 (N + chunk_N * w)
#         aux = np.resize(chunk_mean - mean, (len(chunk_mean), 1))
#         model.at[cla, 'cov'] = (1 / (N + chunk_N * w - 1)) * \
#                                (cov * (N - 1) + chunk_cov * w * (chunk_N - 1) +
#                                 np.dot(aux, aux.T.conj()) * (N * chunk_N * w) / (N + chunk_N * w))
#         model.at[cla, 'N'] = N + chunk_N * w
#     if type_DA == 'LDA':
#         model.at[0, 'LDAcov'] = DA_Classifiers.LDA_Cov_weights(model)
#     return model


def updateModel2(classes, weights, model, chunk_mean, chunk_cov, chunk_N, type_DA):
    for cla in range(classes):
        w = weights[cla]
        if w != 0:
            mean = model.loc[cla, 'mean']
            cov = model.loc[cla, 'cov']
            N = model.loc[cla, 'N']

            model.at[cla, 'mean'] = (N * mean + chunk_N * w * chunk_mean) / \
                                    (N + chunk_N * w)
            #####1
            # aux = np.resize(chunk_mean - mean, (len(chunk_mean), 1))
            # model.at[cla, 'cov'] = (1 / (N + chunk_N * w - 1)) * \
            #                        (cov * (N - 1) + chunk_cov * w * (chunk_N - 1) +
            #                         np.dot(aux, aux.T.conj()) * (N * chunk_N * w) / (N + chunk_N * w))
            ######2
            aux = np.resize(chunk_mean - mean, (len(chunk_mean), 1))
            model.at[cla, 'cov'] = (1 / (N + chunk_N * w - 1)) * \
                                   (cov * (N - 1) + chunk_cov * w * (chunk_N - 1) +
                                    np.dot(aux, aux.T.conj()) * N * chunk_N * (N * w + chunk_N) / (
                                            (N + chunk_N) ** 2) )
            ######3
            # model.at[cla, 'cov'] = (N * cov + chunk_N * w * chunk_cov) / \
            #                         (N + chunk_N * w)

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


def model_incre_threshold(currentModel, classes, trainFeatures, type_DA, threshold):
    t = time.time()


    p = post_probabilities_Calculation(trainFeatures, currentModel, classes, type_DA)
    if np.max(p) > threshold:

        gesture_mean = np.mean(trainFeatures, axis=0)
        gesture_cov = np.cov(trainFeatures, rowvar=False)
        gesture_N = np.size(trainFeatures, axis=0)
        weightsUnlabeledGesture = np.zeros(classes)
        weightsUnlabeledGesture[np.argmax(p)] = 1

        return updateModel2(classes, weightsUnlabeledGesture, currentModel, gesture_mean, gesture_cov, gesture_N,
                            type_DA), time.time() - t, weightsUnlabeledGesture
    else:
        return currentModel, time.time() - t, np.zeros(classes)


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

# %% old models2
# def model_incre_proposedProbMSDA(currentModel, classes, trainFeatures, weakModel, labeledGesturesFeatures,
#                                  labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
#     t = time.time()
#     updatedModel = pd.DataFrame(
#         columns=['cov', 'mean', 'class', 'mean_accumulated', 'w_mean_accumulated', 'cov_accumulated',
#                  'w_cov_accumulated', 'LDAcov'])
#
#     numSamples = 50
#     labeledGesturesFeatures, labeledGesturesLabels = DA_BasedAdaptiveModels.subsetTraining(
#         labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     weightsMSDA = weightMSDA_reduce(currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures,
#                                     labeledGesturesLabels, type_DA)
#
#     weightsMSDA = np.array(weightsMSDA)
#
#     postProb_trainFeatures = post_probabilities_Calculation(trainFeatures, currentModel, classes, type_DA)
#
#     if weightsMSDA.sum() != 0:
#         weightsMSDA_norm = weightsMSDA / weightsMSDA.sum()
#     else:
#         weightsMSDA_norm = weightsMSDA.copy()
#
#     weightsUnlabeledGesture = ((postProb_trainFeatures * entrophyVector(postProb_trainFeatures)) + (
#             weightsMSDA_norm * entrophyVector(weightsMSDA_norm))) / 2
#
#     w_mean_accumulated = np.zeros(classes)
#     mean_accumulated = np.zeros((classes, np.size(trainFeatures, axis=1)))
#     if type_DA == 'LDA':
#         w_cov_accumulated = np.zeros(classes)
#         weightsUnlabeledGesture_Cov = np.zeros(classes)
#         weightsUnlabeledGesture_Cov[np.argmax(weightsUnlabeledGesture)] = 1
#
#     cov_accumulated = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#
#     for cla in range(classes):
#         mean_accumulated[cla, :] = \
#             currentModel['mean_accumulated'].loc[cla] / currentModel['w_mean_accumulated'].loc[cla]
#         updatedModel.at[cla, 'mean_accumulated'] = \
#             currentModel['mean_accumulated'].loc[cla] + weightsUnlabeledGesture[cla] * gestureMean
#
#         w_mean_accumulated[cla] = currentModel['w_mean_accumulated'].loc[cla]
#         updatedModel.at[cla, 'w_mean_accumulated'] = \
#             currentModel['w_mean_accumulated'].loc[cla] + weightsUnlabeledGesture[cla]
#
#         if type_DA == 'QDA':
#             cov_accumulated[cla, :, :] = currentModel['cov_accumulated'].loc[cla] / w_mean_accumulated[cla]
#             updatedModel.at[cla, 'cov_accumulated'] = \
#                 currentModel['cov_accumulated'].loc[cla] + weightsUnlabeledGesture[cla] * gestureCov
#         elif type_DA == 'LDA':
#             cov_accumulated[cla, :, :] = \
#                 currentModel['cov_accumulated'].loc[cla] / currentModel['w_cov_accumulated'].loc[cla]
#             updatedModel.at[cla, 'cov_accumulated'] = \
#                 currentModel['cov_accumulated'].loc[cla] + weightsUnlabeledGesture_Cov[cla] * gestureCov
#
#             w_cov_accumulated[cla] = currentModel['w_cov_accumulated'].loc[cla]
#             updatedModel.at[cla, 'w_cov_accumulated'] = \
#                 currentModel['w_cov_accumulated'].loc[cla] + weightsUnlabeledGesture_Cov[cla]
#
#     sumMean = w_mean_accumulated + weightsUnlabeledGesture
#     weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#     w_mean_accumulated /= sumMean
#
#     means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#     if type_DA == 'QDA':
#         covs = np.resize(gestureCov,
#                          (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#     elif type_DA == 'LDA':
#         sumCov = w_cov_accumulated + weightsUnlabeledGesture_Cov
#         weightsUnlabeledGesture_Norm_Cov = weightsUnlabeledGesture_Cov / sumCov
#         w_cov_accumulated /= sumCov
#         covs = np.resize(gestureCov,
#                          (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm_Cov
#
#     for cla in range(classes):
#         updatedModel.at[cla, 'class'] = cla + 1
#         updatedModel.at[cla, 'mean'] = means[:, cla] + mean_accumulated[cla, :] * w_mean_accumulated[cla]
#
#         if type_DA == 'QDA':
#             updatedModel.at[cla, 'cov'] = covs[:, :, cla] + cov_accumulated[cla, :, :] * w_mean_accumulated[cla]
#         elif type_DA == 'LDA':
#             updatedModel.at[cla, 'cov'] = covs[:, :, cla] + cov_accumulated[cla, :, :] * w_cov_accumulated[cla]
#     updatedModel.at[0, 'LDAcov'] = (currentModel.loc[0, 'LDAcov'] * currentModel[
#         'w_cov_accumulated'].sum() + gestureCov) / (currentModel['w_cov_accumulated'].sum() + 1)
#     return updatedModel, time.time() - t, weightsUnlabeledGesture
#
# def post_probabilities_DA(features, model, classes, type_DA):
#     actualPredictor = []
#
#     if type_DA == 'LDA':
#         LDACov = DA_Classifiers.LDA_Cov(model, classes)
#         for i in range(np.size(features, axis=0)):
#             actualPredictor.append(DA_Classifiers.predictedModelLDA_Prob(features[i, :], model, classes, LDACov))
#     elif type_DA == 'QDA':
#         for i in range(np.size(features, axis=0)):
#             actualPredictor.append(DA_Classifiers.predictedModelQDA_Prob(features[i, :], model, classes))
#
#     return np.array(actualPredictor)
#

# def model_incre_proposedNigam(currentModel, classes, trainFeatures, weakModel, labeledGesturesFeatures,
#                               labeledGesturesLabels, type_DA, samplesInMemory, shotStart, weight_lambda):
#     t = time.time()
#     updatedModel = pd.DataFrame(
#         columns=['cov', 'mean', 'class', 'mean_accumulated', 'w_mean_accumulated', 'cov_accumulated',
#                  'w_cov_accumulated', 'LDAcov'])
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     weightsUnlabeledGesture = post_probabilities_Calculation(trainFeatures, currentModel, classes,
#                                                              type_DA) * weight_lambda
#
#     w_mean_accumulated = np.zeros(classes)
#     mean_accumulated = np.zeros((classes, np.size(trainFeatures, axis=1)))
#     if type_DA == 'LDA':
#         w_cov_accumulated = np.zeros(classes)
#         weightsUnlabeledGesture_Cov = np.zeros(classes)
#         weightsUnlabeledGesture_Cov[np.argmax(weightsUnlabeledGesture)] = 1
#
#     cov_accumulated = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#
#     for cla in range(classes):
#         mean_accumulated[cla, :] = \
#             currentModel['mean_accumulated'].loc[cla] / currentModel['w_mean_accumulated'].loc[cla]
#         updatedModel.at[cla, 'mean_accumulated'] = \
#             currentModel['mean_accumulated'].loc[cla] + weightsUnlabeledGesture[cla] * gestureMean
#
#         w_mean_accumulated[cla] = currentModel['w_mean_accumulated'].loc[cla]
#         updatedModel.at[cla, 'w_mean_accumulated'] = \
#             currentModel['w_mean_accumulated'].loc[cla] + weightsUnlabeledGesture[cla]
#
#         if type_DA == 'QDA':
#             cov_accumulated[cla, :, :] = currentModel['cov_accumulated'].loc[cla] / w_mean_accumulated[cla]
#             updatedModel.at[cla, 'cov_accumulated'] = \
#                 currentModel['cov_accumulated'].loc[cla] + weightsUnlabeledGesture[cla] * gestureCov
#         elif type_DA == 'LDA':
#             cov_accumulated[cla, :, :] = \
#                 currentModel['cov_accumulated'].loc[cla] / currentModel['w_cov_accumulated'].loc[cla]
#             updatedModel.at[cla, 'cov_accumulated'] = \
#                 currentModel['cov_accumulated'].loc[cla] + weightsUnlabeledGesture_Cov[cla] * gestureCov
#
#             w_cov_accumulated[cla] = currentModel['w_cov_accumulated'].loc[cla]
#             updatedModel.at[cla, 'w_cov_accumulated'] = \
#                 currentModel['w_cov_accumulated'].loc[cla] + weightsUnlabeledGesture_Cov[cla]
#
#     sumMean = w_mean_accumulated + weightsUnlabeledGesture
#     weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#     w_mean_accumulated /= sumMean
#
#     means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#     if type_DA == 'QDA':
#         covs = np.resize(gestureCov,
#                          (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#     elif type_DA == 'LDA':
#         sumCov = w_cov_accumulated + weightsUnlabeledGesture_Cov
#         weightsUnlabeledGesture_Norm_Cov = weightsUnlabeledGesture_Cov / sumCov
#         w_cov_accumulated /= sumCov
#         covs = np.resize(gestureCov,
#                          (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm_Cov
#
#     for cla in range(classes):
#         updatedModel.at[cla, 'class'] = cla + 1
#         updatedModel.at[cla, 'mean'] = means[:, cla] + mean_accumulated[cla, :] * w_mean_accumulated[cla]
#
#         if type_DA == 'QDA':
#             updatedModel.at[cla, 'cov'] = covs[:, :, cla] + cov_accumulated[cla, :, :] * w_mean_accumulated[cla]
#         elif type_DA == 'LDA':
#             updatedModel.at[cla, 'cov'] = covs[:, :, cla] + cov_accumulated[cla, :, :] * w_cov_accumulated[cla]
#     updatedModel.at[0, 'LDAcov'] = (currentModel.loc[0, 'LDAcov'] * currentModel[
#         'w_cov_accumulated'].sum() + gestureCov) / (currentModel['w_cov_accumulated'].sum() + 1)
#     return updatedModel, time.time() - t, weightsUnlabeledGesture
#

# def model_incre_proposedLabel(currentModel, classes, trainFeatures, weakModel, labeledGesturesFeatures,
#                               labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
#     t = time.time()
#     updatedModel = pd.DataFrame(
#         columns=['cov', 'mean', 'class', 'mean_accumulated', 'w_mean_accumulated', 'cov_accumulated',
#                  'w_cov_accumulated', 'LDAcov'])
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     aux = np.zeros(classes)
#     aux[np.argmax(post_probabilities_Calculation(trainFeatures, currentModel, classes, type_DA))] = 1
#     weightsUnlabeledGesture = aux
#
#     w_mean_accumulated = np.zeros(classes)
#     mean_accumulated = np.zeros((classes, np.size(trainFeatures, axis=1)))
#     if type_DA == 'LDA':
#         w_cov_accumulated = np.zeros(classes)
#         weightsUnlabeledGesture_Cov = np.zeros(classes)
#         weightsUnlabeledGesture_Cov[np.argmax(weightsUnlabeledGesture)] = 1
#
#     cov_accumulated = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#
#     for cla in range(classes):
#         mean_accumulated[cla, :] = \
#             currentModel['mean_accumulated'].loc[cla] / currentModel['w_mean_accumulated'].loc[cla]
#         updatedModel.at[cla, 'mean_accumulated'] = \
#             currentModel['mean_accumulated'].loc[cla] + weightsUnlabeledGesture[cla] * gestureMean
#
#         w_mean_accumulated[cla] = currentModel['w_mean_accumulated'].loc[cla]
#         updatedModel.at[cla, 'w_mean_accumulated'] = \
#             currentModel['w_mean_accumulated'].loc[cla] + weightsUnlabeledGesture[cla]
#
#         if type_DA == 'QDA':
#             cov_accumulated[cla, :, :] = currentModel['cov_accumulated'].loc[cla] / w_mean_accumulated[cla]
#             updatedModel.at[cla, 'cov_accumulated'] = \
#                 currentModel['cov_accumulated'].loc[cla] + weightsUnlabeledGesture[cla] * gestureCov
#         elif type_DA == 'LDA':
#             cov_accumulated[cla, :, :] = \
#                 currentModel['cov_accumulated'].loc[cla] / currentModel['w_cov_accumulated'].loc[cla]
#             updatedModel.at[cla, 'cov_accumulated'] = \
#                 currentModel['cov_accumulated'].loc[cla] + weightsUnlabeledGesture_Cov[cla] * gestureCov
#
#             w_cov_accumulated[cla] = currentModel['w_cov_accumulated'].loc[cla]
#             updatedModel.at[cla, 'w_cov_accumulated'] = \
#                 currentModel['w_cov_accumulated'].loc[cla] + weightsUnlabeledGesture_Cov[cla]
#
#     sumMean = w_mean_accumulated + weightsUnlabeledGesture
#     weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#     w_mean_accumulated /= sumMean
#
#     means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#     if type_DA == 'QDA':
#         covs = np.resize(gestureCov,
#                          (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#     elif type_DA == 'LDA':
#         sumCov = w_cov_accumulated + weightsUnlabeledGesture_Cov
#         weightsUnlabeledGesture_Norm_Cov = weightsUnlabeledGesture_Cov / sumCov
#         w_cov_accumulated /= sumCov
#         covs = np.resize(gestureCov,
#                          (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm_Cov
#
#     for cla in range(classes):
#         updatedModel.at[cla, 'class'] = cla + 1
#         updatedModel.at[cla, 'mean'] = means[:, cla] + mean_accumulated[cla, :] * w_mean_accumulated[cla]
#
#         if type_DA == 'QDA':
#             updatedModel.at[cla, 'cov'] = covs[:, :, cla] + cov_accumulated[cla, :, :] * w_mean_accumulated[cla]
#         elif type_DA == 'LDA':
#             updatedModel.at[cla, 'cov'] = covs[:, :, cla] + cov_accumulated[cla, :, :] * w_cov_accumulated[cla]
#     updatedModel.at[0, 'LDAcov'] = (currentModel.loc[0, 'LDAcov'] * currentModel[
#         'w_cov_accumulated'].sum() + gestureCov) / (currentModel['w_cov_accumulated'].sum() + 1)
#     return updatedModel, time.time() - t, weightsUnlabeledGesture
#
#
# def model_batch_proposedProbMSDA(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                                  labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart,
#                                  unlabeledGesturesTotal, dataTotal, kBest):
#     t = time.time()
#
#     unlabeledGesturesTotal = np.array(unlabeledGesturesTotal)
#     featuresUnlabeledGestures = unlabeledGesturesTotal[:, :np.size(labeledGesturesFeatures, axis=1)]
#     numberUnlabeledGestures = unlabeledGesturesTotal[:, -1]
#     probList = dataTotal[:, -classes:]
#     featuresList = labeledGesturesFeatures
#     model = weakModel.copy()
#     if kBest == None:
#         kBest = int(numberUnlabeledGestures[-1]) + 1
#     if int(numberUnlabeledGestures[-1]) <= kBest:
#         for gesture in range(1, int(numberUnlabeledGestures[-1]) + 1):
#             X = featuresUnlabeledGestures[numberUnlabeledGestures == gesture]
#
#             model, _, _ = model_incre_proposedProbMSDA(
#                 model, classes, X, weakModel,
#                 labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
#
#         #     prob_labels = post_probabilities_Calculation(X, model, classes, type_DA)
#         #     weightsMSDA = weightMSDA_reduce(
#         #         model, np.mean(X, axis=0), np.cov(X, rowvar=False), classes, labeledGesturesFeatures,
#         #         labeledGesturesLabels, type_DA)
#         #     weightsMSDA = np.array(weightsMSDA)
#         #
#         #     if weightsMSDA.sum() != 0:
#         #         weightsMSDA_norm = weightsMSDA / weightsMSDA.sum()
#         #     else:
#         #         weightsMSDA_norm = weightsMSDA.copy()
#         #
#         #     w = ((prob_labels * entrophyVector(prob_labels)) + (
#         #             weightsMSDA_norm * entrophyVector(weightsMSDA_norm))) / 2
#         #     numberSegments = len(numberUnlabeledGestures[numberUnlabeledGestures == gesture])
#         #     probList = np.vstack((probList, np.ones((numberSegments, classes)) * w))
#         #     featuresList = np.vstack((featuresList, X))
#         # return modelCalculation_prob(featuresList, probList, classes), time.time() - t, 0
#         return model, time.time() - t, 0
#     else:
#         controlVector = list(np.arange(int(numberUnlabeledGestures[-1])) + 1)
#         while len(controlVector) != 0:
#             prob_labels_max = []
#             for gesture in list(controlVector):
#                 X = featuresUnlabeledGestures[numberUnlabeledGestures == gesture]
#
#                 prob_labels = post_probabilities_Calculation(X, model, classes, type_DA)
#                 weightsMSDA = weightMSDA_reduce(
#                     model, np.mean(X, axis=0), np.cov(X, rowvar=False), classes, labeledGesturesFeatures,
#                     labeledGesturesLabels, type_DA)
#                 weightsMSDA = np.array(weightsMSDA)
#                 if weightsMSDA.sum() != 0:
#                     weightsMSDA_norm = weightsMSDA / weightsMSDA.sum()
#                 else:
#                     weightsMSDA_norm = weightsMSDA.copy()
#                 w = ((prob_labels * entrophyVector(prob_labels)) + (
#                         weightsMSDA_norm * entrophyVector(weightsMSDA_norm))) / 2
#                 prob_labels_max.append(np.max(w))
#             controlVector = [controlVector[i] for i in list(np.argsort(prob_labels_max))]
#             for gesture in list(controlVector[-kBest:]):
#                 controlVector = controlVector[:-1]
#                 X = featuresUnlabeledGestures[numberUnlabeledGestures == gesture]
#
#                 model, _, _ = model_incre_proposedProbMSDA(
#                     model, classes, X, weakModel,
#                     labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
#
#             #     prob_labels = post_probabilities_Calculation(X, model, classes, type_DA)
#             #     weightsMSDA = weightMSDA_reduce(
#             #         model, np.mean(X, axis=0), np.cov(X, rowvar=False), classes, labeledGesturesFeatures,
#             #         labeledGesturesLabels, type_DA)
#             #     weightsMSDA = np.array(weightsMSDA)
#             #     if weightsMSDA.sum() != 0:
#             #         weightsMSDA_norm = weightsMSDA / weightsMSDA.sum()
#             #     else:
#             #         weightsMSDA_norm = weightsMSDA.copy()
#             #     w = ((prob_labels * entrophyVector(prob_labels)) + (
#             #             weightsMSDA_norm * entrophyVector(weightsMSDA_norm))) / 2
#             #
#             #     numberSegments = len(numberUnlabeledGestures[numberUnlabeledGestures == gesture])
#             #     probList = np.vstack((probList, np.ones((numberSegments, classes)) * w))
#             #     featuresList = np.vstack((featuresList, X))
#             # model = modelCalculation_prob(featuresList, probList, classes)
#
#         return model, time.time() - t, 0


##############################################################################
# def model_seflfTraining(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                         labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart,
#                         unlabeledGesturesTotal, dataTotal, kBest):
#     t = time.time()
#
#     unlabeledGesturesTotal = np.array(unlabeledGesturesTotal)
#     numberFeatures = np.size(labeledGesturesFeatures, axis=1)
#     featuresUnlabeledGestures = unlabeledGesturesTotal[:, :numberFeatures]
#     numberUnlabeledGestures = unlabeledGesturesTotal[:, -1]
#     labelsList = labeledGesturesLabels
#     featuresList = labeledGesturesFeatures
#
#     if kBest == None:
#         kBest = int(numberUnlabeledGestures[-1]) + 1
#     if int(numberUnlabeledGestures[-1]) <= kBest:
#         for gesture in range(1, int(numberUnlabeledGestures[-1]) + 1):
#             X = featuresUnlabeledGestures[numberUnlabeledGestures == gesture]
#             prob_labels = post_probabilities_DA(X, weakModel, classes, type_DA)
#             cla = np.argmax(prob_labels.mean(axis=0)) + 1
#             numberSegments = len(numberUnlabeledGestures[numberUnlabeledGestures == gesture])
#             labelsList = np.hstack((labelsList, np.ones(numberSegments) * cla))
#             featuresList = np.vstack((featuresList, X))
#         return modelCalculation_labels(featuresList, labelsList, classes), time.time() - t, 0
#     else:
#         model = weakModel.copy()
#         controlVector = list(np.arange(int(numberUnlabeledGestures[-1])) + 1)
#         while len(controlVector) != 0:
#             prob_labels_max = []
#             for gesture in list(controlVector):
#                 X = featuresUnlabeledGestures[numberUnlabeledGestures == gesture]
#                 prob_labels = post_probabilities_DA(X, model, classes, type_DA)
#                 prob_labels_max.append(np.max(prob_labels.mean(axis=0)))
#             controlVector = [controlVector[i] for i in list(np.argsort(prob_labels_max))]
#             for gesture in list(controlVector[-kBest:]):
#                 controlVector = controlVector[:-1]
#                 X = featuresUnlabeledGestures[numberUnlabeledGestures == gesture]
#                 prob_labels = post_probabilities_DA(X, model, classes, type_DA)
#                 cla = np.argmax(prob_labels.mean(axis=0)) + 1
#                 numberSegments = len(numberUnlabeledGestures[numberUnlabeledGestures == gesture])
#                 labelsList = np.hstack((labelsList, np.ones(numberSegments) * cla))
#                 featuresList = np.vstack((featuresList, X))
#             model = modelCalculation_labels(featuresList, labelsList, classes)
#
#         return model, time.time() - t, 0
#
#
# def model_EM(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel, labeledGesturesFeatures,
#              labeledGesturesLabels, type_DA, samplesInMemory, shotStart, unlabeledGesturesTotal, dataTotal):
#     t = time.time()
#     model = weakModel.copy()
#     times = 10
#     unlabeledGesturesTotal = np.array(unlabeledGesturesTotal)
#     featuresUnlabeledGestures = unlabeledGesturesTotal[:, :np.size(labeledGesturesFeatures, axis=1)]
#     numberUnlabeledGestures = unlabeledGesturesTotal[:, -1]
#     for i in range(times):
#         probList = dataTotal[:, -classes:]
#         featuresList = labeledGesturesFeatures
#         for gesture in range(1, int(numberUnlabeledGestures[-1]) + 1):
#             X = featuresUnlabeledGestures[numberUnlabeledGestures == gesture]
#             prob_labels = post_probabilities_DA(X, model, classes, type_DA)
#             numberSegments = len(numberUnlabeledGestures[numberUnlabeledGestures == gesture])
#             probList = np.vstack((probList, np.ones((numberSegments, classes)) * prob_labels.mean(axis=0)))
#             featuresList = np.vstack((featuresList, X))
#         model = modelCalculation_prob(featuresList, probList, classes)
#     return model, time.time() - t, 0
#
#
# def model_Nigam(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel, labeledGesturesFeatures,
#                 labeledGesturesLabels, type_DA, samplesInMemory, shotStart, unlabeledGesturesTotal, dataTotal,
#                 weight_lambda, times):
#     t = time.time()
#     model = weakModel.copy()
#     unlabeledGesturesTotal = np.array(unlabeledGesturesTotal)
#     featuresUnlabeledGestures = unlabeledGesturesTotal[:, :np.size(labeledGesturesFeatures, axis=1)]
#     numberUnlabeledGestures = unlabeledGesturesTotal[:, -1]
#     for i in range(times):
#         probList = dataTotal[:, -classes:]
#         featuresList = labeledGesturesFeatures
#         for gesture in range(1, int(numberUnlabeledGestures[-1]) + 1):
#             X = featuresUnlabeledGestures[numberUnlabeledGestures == gesture]
#             prob_labels = post_probabilities_DA(X, model, classes, type_DA)
#             numberSegments = len(numberUnlabeledGestures[numberUnlabeledGestures == gesture])
#             probList = np.vstack(
#                 (probList, np.ones((numberSegments, classes)) * prob_labels.mean(axis=0) * weight_lambda))
#             featuresList = np.vstack((featuresList, X))
#         model = modelCalculation_prob(featuresList, probList, classes)
#     return model, time.time() - t, 0
#
#
# def model_Proposed(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                    labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart,
#                    unlabeledGesturesTotal, dataTotal, kBest):
#     t = time.time()
#
#     unlabeledGesturesTotal = np.array(unlabeledGesturesTotal)
#     featuresUnlabeledGestures = unlabeledGesturesTotal[:, :np.size(labeledGesturesFeatures, axis=1)]
#     numberUnlabeledGestures = unlabeledGesturesTotal[:, -1]
#     probList = dataTotal[:, -classes:]
#     featuresList = labeledGesturesFeatures
#     model = weakModel.copy()
#     if kBest == None:
#         kBest = int(numberUnlabeledGestures[-1]) + 1
#     if int(numberUnlabeledGestures[-1]) <= kBest:
#         for gesture in range(1, int(numberUnlabeledGestures[-1]) + 1):
#             X = featuresUnlabeledGestures[numberUnlabeledGestures == gesture]
#             # model, _, _ = model_incre_Proposed(
#             #     model, classes, X, weakModel,
#             #     labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
#
#             prob_labels = post_probabilities_Calculation(X, model, classes, type_DA)
#             weightsMSDA = weightMSDA_reduce(
#                 model, np.mean(X, axis=0), np.cov(X, rowvar=False), classes, labeledGesturesFeatures,
#                 labeledGesturesLabels, type_DA)
#             weightsMSDA = np.array(weightsMSDA)
#
#             if weightsMSDA.sum() != 0:
#                 weightsMSDA_norm = weightsMSDA / weightsMSDA.sum()
#             else:
#                 weightsMSDA_norm = weightsMSDA.copy()
#
#             w = ((prob_labels * entrophyVector(prob_labels)) + (
#                     weightsMSDA_norm * entrophyVector(weightsMSDA_norm))) / 2
#             numberSegments = len(numberUnlabeledGestures[numberUnlabeledGestures == gesture])
#             probList = np.vstack((probList, np.ones((numberSegments, classes)) * w))
#             featuresList = np.vstack((featuresList, X))
#         return modelCalculation_prob(featuresList, probList, classes), time.time() - t, 0
#         # return model, time.time() - t, 0
#     else:
#         controlVector = list(np.arange(int(numberUnlabeledGestures[-1])) + 1)
#         while len(controlVector) != 0:
#             prob_labels_max = []
#             for gesture in list(controlVector):
#                 X = featuresUnlabeledGestures[numberUnlabeledGestures == gesture]
#
#                 prob_labels = post_probabilities_Calculation(X, model, classes, type_DA)
#                 weightsMSDA = weightMSDA_reduce(
#                     model, np.mean(X, axis=0), np.cov(X, rowvar=False), classes, labeledGesturesFeatures,
#                     labeledGesturesLabels, type_DA)
#                 weightsMSDA = np.array(weightsMSDA)
#                 if weightsMSDA.sum() != 0:
#                     weightsMSDA_norm = weightsMSDA / weightsMSDA.sum()
#                 else:
#                     weightsMSDA_norm = weightsMSDA.copy()
#                 w = ((prob_labels * entrophyVector(prob_labels)) + (
#                         weightsMSDA_norm * entrophyVector(weightsMSDA_norm))) / 2
#                 prob_labels_max.append(np.max(w))
#             controlVector = [controlVector[i] for i in list(np.argsort(prob_labels_max))]
#             for gesture in list(controlVector[-kBest:]):
#                 controlVector = controlVector[:-1]
#                 X = featuresUnlabeledGestures[numberUnlabeledGestures == gesture]
#                 # model, _, _ = model_incre_Proposed(
#                 #     model, classes, X, weakModel,
#                 #     labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
#
#                 prob_labels = post_probabilities_Calculation(X, model, classes, type_DA)
#                 weightsMSDA = weightMSDA_reduce(
#                     model, np.mean(X, axis=0), np.cov(X, rowvar=False), classes, labeledGesturesFeatures,
#                     labeledGesturesLabels, type_DA)
#                 weightsMSDA = np.array(weightsMSDA)
#                 if weightsMSDA.sum() != 0:
#                     weightsMSDA_norm = weightsMSDA / weightsMSDA.sum()
#                 else:
#                     weightsMSDA_norm = weightsMSDA.copy()
#                 w = ((prob_labels * entrophyVector(prob_labels)) + (
#                         weightsMSDA_norm * entrophyVector(weightsMSDA_norm))) / 2
#
#                 numberSegments = len(numberUnlabeledGestures[numberUnlabeledGestures == gesture])
#                 probList = np.vstack((probList, np.ones((numberSegments, classes)) * w))
#                 featuresList = np.vstack((featuresList, X))
#             model = modelCalculation_prob(featuresList, probList, classes)
#
#         return model, time.time() - t, 0
#
#
# def model_weight(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                  labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart,
#                  unlabeledGesturesTotal, dataTotal, kBest):
#     t = time.time()
#
#     unlabeledGesturesTotal = np.array(unlabeledGesturesTotal)
#     featuresUnlabeledGestures = unlabeledGesturesTotal[:, :np.size(labeledGesturesFeatures, axis=1)]
#     numberUnlabeledGestures = unlabeledGesturesTotal[:, -1]
#     probList = dataTotal[:, -classes:]
#     featuresList = labeledGesturesFeatures
#     model = weakModel.copy()
#     if kBest == None:
#         kBest = int(numberUnlabeledGestures[-1]) + 1
#     if int(numberUnlabeledGestures[-1]) <= kBest:
#         for gesture in range(1, int(numberUnlabeledGestures[-1]) + 1):
#             X = featuresUnlabeledGestures[numberUnlabeledGestures == gesture]
#
#             prob_labels = post_probabilities_Calculation(X, model, classes, type_DA)
#
#             w = prob_labels
#             numberSegments = len(numberUnlabeledGestures[numberUnlabeledGestures == gesture])
#             probList = np.vstack((probList, np.ones((numberSegments, classes)) * w))
#             featuresList = np.vstack((featuresList, X))
#         return modelCalculation_prob(featuresList, probList, classes), time.time() - t, 0
#         # return model, time.time() - t, 0
#     else:
#         controlVector = list(np.arange(int(numberUnlabeledGestures[-1])) + 1)
#         while len(controlVector) != 0:
#             prob_labels_max = []
#             for gesture in list(controlVector):
#                 X = featuresUnlabeledGestures[numberUnlabeledGestures == gesture]
#
#                 prob_labels = post_probabilities_Calculation(X, model, classes, type_DA)
#
#                 w = prob_labels
#                 prob_labels_max.append(np.max(w))
#             controlVector = [controlVector[i] for i in list(np.argsort(prob_labels_max))]
#             for gesture in list(controlVector[-kBest:]):
#                 controlVector = controlVector[:-1]
#                 X = featuresUnlabeledGestures[numberUnlabeledGestures == gesture]
#                 # model, _, _ = model_incre_Proposed(
#                 #     model, classes, X, weakModel,
#                 #     labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
#
#                 prob_labels = post_probabilities_Calculation(X, model, classes, type_DA)
#
#                 w = prob_labels
#
#                 numberSegments = len(numberUnlabeledGestures[numberUnlabeledGestures == gesture])
#                 probList = np.vstack((probList, np.ones((numberSegments, classes)) * w))
#                 featuresList = np.vstack((featuresList, X))
#             model = modelCalculation_prob(featuresList, probList, classes)
#
#         return model, time.time() - t, 0
#
#
# def model_incre_weight(currentModel, classes, trainFeatures, weakModel, labeledGesturesFeatures, labeledGesturesLabels,
#                        type_DA, samplesInMemory, shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     weightsMSDA = post_probabilities_Calculation(trainFeatures, currentModel, classes, type_DA)
#
#     if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures = np.ones(classes) * shotStart
#
#         weightsUnlabeledGesture = weightsMSDA
#         if type_DA == 'LDA':
#             weightsUnlabeledGesture_Cov = np.zeros(classes)
#             weightsUnlabeledGesture_Cov[np.argmax(weightsUnlabeledGesture)] = 1
#             w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures[cla] + \
#                                               weightsUnlabeledGesture[cla] * gestureMean
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]
#
#             if type_DA == 'QDA':
#                 adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures[cla] + \
#                                                  weightsUnlabeledGesture[cla] * gestureCov
#             elif type_DA == 'LDA':
#                 adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture_Cov[cla]
#
#                 adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures[cla] + \
#                                                  weightsUnlabeledGesture_Cov[cla] * gestureCov
#
#         sumMean = w_labeledGestures + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         w_labeledGestures /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         if type_DA == 'LDA':
#             sumCov = w_labeledGestures_Cov + weightsUnlabeledGesture_Cov
#             weightsUnlabeledGesture_Norm_Cov = weightsUnlabeledGesture_Cov / sumCov
#             w_labeledGestures_Cov /= sumCov
#             covs = np.resize(gestureCov,
#                              (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm_Cov
#         elif type_DA == 'QDA':
#             covs = np.resize(gestureCov,
#                              (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + weakModel['mean'].loc[cla] * w_labeledGestures[cla]
#             if type_DA == 'QDA':
#                 adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + weakModel['cov'].loc[cla] * w_labeledGestures[cla]
#             elif type_DA == 'LDA':
#                 adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#         weightsUnlabeledGesture = weightsMSDA
#         if type_DA == 'LDA':
#             wJCov = np.zeros(classes)
#             weightsUnlabeledGesture_Cov = np.zeros(classes)
#             weightsUnlabeledGesture_Cov[np.argmax(weightsUnlabeledGesture)] = 1
#
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                               gestureMean
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]
#
#             if type_DA == 'QDA':
#                 JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#                 adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                                  gestureCov
#             elif type_DA == 'LDA':
#                 wJCov[cla] = currentModel['wCov_J'].loc[cla]
#                 adaptiveModel.at[cla, 'wCov_J'] = currentModel['wCov_J'].loc[cla] + weightsUnlabeledGesture_Cov[cla]
#
#                 JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wCov_J'].loc[cla]
#                 adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture_Cov[cla] * \
#                                                  gestureCov
#
#         sumMean = wJMean + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         wJMean /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         if type_DA == 'QDA':
#             covs = np.resize(gestureCov,
#                              (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#         elif type_DA == 'LDA':
#             sumCov = wJCov + weightsUnlabeledGesture_Cov
#             weightsUnlabeledGesture_Norm_Cov = weightsUnlabeledGesture_Cov / sumCov
#             wJCov /= sumCov
#             covs = np.resize(gestureCov,
#                              (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm_Cov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]
#
#             if type_DA == 'QDA':
#                 adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]
#             elif type_DA == 'LDA':
#                 adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJCov[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, weightsUnlabeledGesture
#
#
# def model_incre_Proposed(currentModel, classes, trainFeatures, weakModel, labeledGesturesFeatures,
#                          labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     numSamples = 50
#     labeledGesturesFeatures, labeledGesturesLabels = DA_BasedAdaptiveModels.subsetTraining(
#         labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     weightsMSDA = weightMSDA_reduce(currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures,
#                                     labeledGesturesLabels, type_DA)
#
#     weightsMSDA = np.array(weightsMSDA)
#
#     postProb_trainFeatures = post_probabilities_Calculation(trainFeatures, currentModel, classes, type_DA)
#
#     a = postProb_trainFeatures.copy()
#     b = weightsMSDA.copy()
#     if b.sum() != 0:
#         b_norm = b / b.sum()
#     else:
#         b_norm = b.copy()
#
#     w = ((a * entrophyVector(a)) + (b_norm * entrophyVector(b_norm))) / 2
#     # print('sum COMPARE RANDOM_ b_norm', w)
#     weightsMSDA = w.copy()
#
#     if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures = np.ones(classes) * shotStart
#
#         weightsUnlabeledGesture = weightsMSDA
#         if type_DA == 'LDA':
#             weightsUnlabeledGesture_Cov = np.zeros(classes)
#             weightsUnlabeledGesture_Cov[np.argmax(weightsMSDA)] = 1
#             w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures[cla] + \
#                                               weightsUnlabeledGesture[cla] * gestureMean
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]
#
#             if type_DA == 'QDA':
#                 adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures[cla] + \
#                                                  weightsUnlabeledGesture[cla] * gestureCov
#             elif type_DA == 'LDA':
#                 adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture_Cov[cla]
#
#                 adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures[cla] + \
#                                                  weightsUnlabeledGesture_Cov[cla] * gestureCov
#
#         sumMean = w_labeledGestures + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         w_labeledGestures /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         if type_DA == 'LDA':
#             sumCov = w_labeledGestures_Cov + weightsUnlabeledGesture_Cov
#             weightsUnlabeledGesture_Norm_Cov = weightsUnlabeledGesture_Cov / sumCov
#             w_labeledGestures_Cov /= sumCov
#             covs = np.resize(gestureCov,
#                              (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm_Cov
#         elif type_DA == 'QDA':
#             covs = np.resize(gestureCov,
#                              (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + weakModel['mean'].loc[cla] * w_labeledGestures[cla]
#             if type_DA == 'QDA':
#                 adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + weakModel['cov'].loc[cla] * w_labeledGestures[cla]
#             elif type_DA == 'LDA':
#                 adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#         weightsUnlabeledGesture = weightsMSDA
#         if type_DA == 'LDA':
#             wJCov = np.zeros(classes)
#             weightsUnlabeledGesture_Cov = np.zeros(classes)
#             weightsUnlabeledGesture_Cov[np.argmax(weightsMSDA)] = 1
#
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                               gestureMean
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]
#
#             if type_DA == 'QDA':
#                 JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#                 adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                                  gestureCov
#             elif type_DA == 'LDA':
#                 wJCov[cla] = currentModel['wCov_J'].loc[cla]
#                 adaptiveModel.at[cla, 'wCov_J'] = currentModel['wCov_J'].loc[cla] + weightsUnlabeledGesture_Cov[cla]
#
#                 JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wCov_J'].loc[cla]
#                 adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture_Cov[cla] * \
#                                                  gestureCov
#
#         sumMean = wJMean + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         wJMean /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         if type_DA == 'QDA':
#             covs = np.resize(gestureCov,
#                              (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#         elif type_DA == 'LDA':
#             sumCov = wJCov + weightsUnlabeledGesture_Cov
#             weightsUnlabeledGesture_Norm_Cov = weightsUnlabeledGesture_Cov / sumCov
#             wJCov /= sumCov
#             covs = np.resize(gestureCov,
#                              (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm_Cov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]
#
#             if type_DA == 'QDA':
#                 adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]
#             elif type_DA == 'LDA':
#                 adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJCov[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, weightsUnlabeledGesture
#
#
# def model_incre_selfTraining(currentModel, classes, trainFeatures, weakModel, labeledGesturesFeatures,
#                              labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#     postProb_trainFeatures = post_probabilities_DA(trainFeatures, currentModel, classes, type_DA)
#     aux = np.zeros(classes)
#     aux[np.argmax(postProb_trainFeatures.mean(axis=0))] = 1
#     postProb_trainFeatures = aux
#
#     if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures = np.ones(classes) * shotStart
#
#         weightsUnlabeledGesture = postProb_trainFeatures
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures[cla] + \
#                                               weightsUnlabeledGesture[cla] * gestureMean
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]
#
#             adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures[cla] + \
#                                              weightsUnlabeledGesture[cla] * gestureCov
#
#         sumMean = w_labeledGestures + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         w_labeledGestures /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + weakModel['mean'].loc[cla] * w_labeledGestures[cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + weakModel['cov'].loc[cla] * w_labeledGestures[cla]
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#         weightsUnlabeledGesture = postProb_trainFeatures
#
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                               gestureMean
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]
#
#             JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                              gestureCov
#
#         sumMean = wJMean + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         wJMean /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]
#
#             adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, weightsUnlabeledGesture
#
#
# def model_incre_Nigam(currentModel, classes, trainFeatures, weakModel, labeledGesturesFeatures, labeledGesturesLabels,
#                       type_DA, samplesInMemory, shotStart, weight_lambda):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     # postProb_trainFeatures = post_probabilities_DA(trainFeatures, currentModel, classes, type_DA)
#     # postProb_trainFeatures = postProb_trainFeatures.mean(axis=0) * weight_lambda
#
#     postProb_trainFeatures = post_probabilities_Calculation(trainFeatures, currentModel, classes,
#                                                             type_DA) * weight_lambda
#
#     if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures = np.ones(classes) * shotStart
#
#         weightsUnlabeledGesture = postProb_trainFeatures
#
#         if type_DA == 'LDA':
#             weightsUnlabeledGesture_Cov = np.zeros(classes)
#             weightsUnlabeledGesture_Cov[np.argmax(weightsUnlabeledGesture)] = 1
#             w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures[cla] + \
#                                               weightsUnlabeledGesture[cla] * gestureMean
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]
#
#             if type_DA == 'QDA':
#                 adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures[cla] + \
#                                                  weightsUnlabeledGesture[cla] * gestureCov
#             elif type_DA == 'LDA':
#                 adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture_Cov[cla]
#
#                 adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures[cla] + \
#                                                  weightsUnlabeledGesture_Cov[cla] * gestureCov
#
#             # adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures[cla] + \
#             #                                  weightsUnlabeledGesture[cla] * gestureCov
#
#         sumMean = w_labeledGestures + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         w_labeledGestures /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         if type_DA == 'LDA':
#             sumCov = w_labeledGestures_Cov + weightsUnlabeledGesture_Cov
#             weightsUnlabeledGesture_Norm_Cov = weightsUnlabeledGesture_Cov / sumCov
#             w_labeledGestures_Cov /= sumCov
#             covs = np.resize(gestureCov,
#                              (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm_Cov
#         elif type_DA == 'QDA':
#             covs = np.resize(gestureCov,
#                              (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         # covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + weakModel['mean'].loc[cla] * w_labeledGestures[cla]
#             if type_DA == 'QDA':
#                 adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + weakModel['cov'].loc[cla] * w_labeledGestures[cla]
#             elif type_DA == 'LDA':
#                 adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#             # adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + weakModel['cov'].loc[cla] * w_labeledGestures[cla]
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#         weightsUnlabeledGesture = postProb_trainFeatures
#
#         if type_DA == 'LDA':
#             wJCov = np.zeros(classes)
#             weightsUnlabeledGesture_Cov = np.zeros(classes)
#             weightsUnlabeledGesture_Cov[np.argmax(weightsUnlabeledGesture)] = 1
#
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                               gestureMean
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]
#
#             if type_DA == 'QDA':
#                 JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#                 adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                                  gestureCov
#             elif type_DA == 'LDA':
#                 wJCov[cla] = currentModel['wCov_J'].loc[cla]
#                 adaptiveModel.at[cla, 'wCov_J'] = currentModel['wCov_J'].loc[cla] + weightsUnlabeledGesture_Cov[cla]
#
#                 JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wCov_J'].loc[cla]
#                 adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture_Cov[cla] * \
#                                                  gestureCov
#
#             # JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             # adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#             #                                  gestureCov
#
#         sumMean = wJMean + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         wJMean /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         if type_DA == 'QDA':
#             covs = np.resize(gestureCov,
#                              (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#         elif type_DA == 'LDA':
#             sumCov = wJCov + weightsUnlabeledGesture_Cov
#             weightsUnlabeledGesture_Norm_Cov = weightsUnlabeledGesture_Cov / sumCov
#             wJCov /= sumCov
#             covs = np.resize(gestureCov,
#                              (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm_Cov
#
#         # covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]
#
#             if type_DA == 'QDA':
#                 adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]
#             elif type_DA == 'LDA':
#                 adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJCov[cla]
#
#             # adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, weightsUnlabeledGesture
#
#
# ################################################################################
#
# def model_incre_gestures_labels(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                                 labeledGesturesFeatures,
#                                 labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     aux = np.zeros(classes)
#     aux[np.argmax(postProb_trainFeatures)] = 1
#     postProb_trainFeatures = aux
#
#     if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures = np.ones(classes) * shotStart
#
#         weightsUnlabeledGesture = postProb_trainFeatures
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures[cla] + \
#                                               weightsUnlabeledGesture[cla] * gestureMean
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]
#
#             adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures[cla] + \
#                                              weightsUnlabeledGesture[cla] * gestureCov
#
#         sumMean = w_labeledGestures + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         w_labeledGestures /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + weakModel['mean'].loc[cla] * w_labeledGestures[cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + weakModel['cov'].loc[cla] * w_labeledGestures[cla]
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#         weightsUnlabeledGesture = postProb_trainFeatures
#
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                               gestureMean
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]
#
#             JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                              gestureCov
#
#         sumMean = wJMean + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         wJMean /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]
#
#             adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, weightsUnlabeledGesture
#
#
# def model_incre_gestures_weight(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                                 labeledGesturesFeatures,
#                                 labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures = np.ones(classes) * shotStart
#
#         weightsUnlabeledGesture = postProb_trainFeatures
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures[cla] + \
#                                               weightsUnlabeledGesture[cla] * gestureMean
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]
#
#             adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures[cla] + \
#                                              weightsUnlabeledGesture[cla] * gestureCov
#
#         sumMean = w_labeledGestures + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         w_labeledGestures /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + weakModel['mean'].loc[cla] * w_labeledGestures[cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + weakModel['cov'].loc[cla] * w_labeledGestures[cla]
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#         weightsUnlabeledGesture = postProb_trainFeatures
#
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                               gestureMean
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]
#
#             JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                              gestureCov
#
#         sumMean = wJMean + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         wJMean /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]
#
#             adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, weightsUnlabeledGesture
#
#
# #
# # def model_incre_gestures_weight_MSDA_2(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
# #                                      labeledGesturesFeatures,
# #                                      labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
# #     t = time.time()
# #     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
# #
# #     numSamples = 50
# #     labeledGesturesFeatures, labeledGesturesLabels = DA_BasedAdaptiveModels.subsetTraining(
# #         labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)
# #
# #     gestureMean = np.mean(trainFeatures, axis=0)
# #     gestureCov = np.cov(trainFeatures, rowvar=False)
# #
# #     weightsMSDA = weightMSDA_reduce(currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures,
# #                                     labeledGesturesLabels, type_DA)
# #
# #     weightsMSDA = np.array(weightsMSDA)
# #
# #     # weightsMSDA =postProb_trainFeatures
# #
# #     # weightsMSDA = weight_MSDA_JS(currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures,
# #     #                              labeledGesturesLabels, type_DA)
# #
# #     # print('weightsMSDA', type_DA)
# #     # print(np.round(weightsMSDA, 2))
# #     # print('postProb_trainFeatures', type_DA)
# #     # print(np.round(postProb_trainFeatures, 2))
# #
# #     # if weightsMSDA.sum() != 0:
# #     #     entMSDA = entrophyVector(weightsMSDA / weightsMSDA.sum())
# #     # else:
# #     #     entMSDA = 0
# #
# #     # weightsMSDA = postProb_trainFeatures * weightsMSDA
# #
# #     # weightsMSDA = postProb_trainFeatures * weightsMSDA
# #
# #     # if weightsMSDA.sum() != 0:
# #
# #     # else:
# #     #     weightsMSDA = postProb_trainFeatures
# #
# #     # # ## Entrophy
# #     # if weightsMSDA.sum() != 0:
# #     #     entMSDA = entrophyVector(weightsMSDA / weightsMSDA.sum())
# #     # else:
# #     #     entMSDA = 0
# #     #
# #     # entPost = entrophyVector(postProb_trainFeatures)
# #     #
# #     # sumEnt = entMSDA + entPost
# #     # entMSDA /= sumEnt
# #     # entPost /= sumEnt
# #     # print('entMSDA', entMSDA)
# #     # weightsMSDA = postProb_trainFeatures * entPost + weightsMSDA * entMSDA
# #
# #     # weightsMSDA/=weightsMSDA.sum()
# #
# #     # if weightsMSDA.sum() != 0:
# #     #     entMSDA = entrophyVector(weightsMSDA / weightsMSDA.sum())
# #     # else:
# #     #     entMSDA = 0
# #
# #     # # print('entrophy')
# #     # # print('weightsMSDA', type_DA)
# #     # # print(weightsMSDA * entMSDA)
# #     # # print('postProb_trainFeatures', type_DA)
# #     # # print(postProb_trainFeatures * entPost)
# #     # print('entrophyTotal', type_DA)
# #
# #     # #####1
# #     # weightsMSDA *= entMSDA
# #     # weightsMSDA += postProb_trainFeatures
# #     # weightsMSDA /= weightsMSDA.sum()
# #     # weightsMSDA = weightsMSDA * entrophyVector(weightsMSDA)
# #
# #     ######2
# #     # weightsMSDA = (postProb_trainFeatures * entPost + weightsMSDA * entMSDA) / 2
# #
# #     ############METHOD2
# #     # if weightsMSDA.sum() != 0:
# #     #     entMSDA = entrophyVector(weightsMSDA / weightsMSDA.sum())
# #     # else:
# #     #     entMSDA = 0
# #     # entPost = entrophyVector(postProb_trainFeatures)
# #     a = postProb_trainFeatures.copy()
# #     b = weightsMSDA.copy()
# #     if b.sum() != 0:
# #         b_norm = b / b.sum()
# #     else:
# #         b_norm = b.copy()
# #
# #     # w = a * b
# #     # print('multi', w)
# #     #
# #     # w = a * b
# #     # if w.sum() != 0:
# #     #     w /= w.sum()
# #     # print('multi norm', w)
# #
# #     # w = (a * entPost) * (b * entMSDA)
# #     # print('multi entrophy', w)
# #     #
# #     # w = (a * entPost) * (b * entMSDA)
# #     # if w.sum() != 0:
# #     #     w /= w.sum()
# #     # print('multi norm entrophy', w)
# #     #
# #     # aa = a.copy()
# #     # bb = b.copy()
# #     # aa[b == 0] = 0
# #     # bb[a == 0] = 0
# #     # w = (aa + bb) / 2
# #     # print('sum no zeros ', w)
# #
# #     # aa = a.copy()
# #     # bb = b.copy()
# #     # aa[b == 0] = 0
# #     # bb[a == 0] = 0
# #     # w = aa + bb
# #     # if w.sum() != 0:
# #     #     w /= w.sum()
# #     # print('sum no zeros norm', w)
# #
# #     # aa = a.copy()
# #     # bb = b.copy()
# #     # aa[b == 0] = 0
# #     # bb[a == 0] = 0
# #     # w = ((aa * entPost) + (bb * entMSDA)) / 2
# #     # print('sum no zeros entrophy ', w)
# #
# #     # aa = a.copy()
# #     # bb = b.copy()
# #     # aa[b == 0] = 0
# #     # bb[a == 0] = 0
# #     # w = (aa * entPost) + (bb * entMSDA)
# #     # if w.sum() != 0:
# #     #     w /= w.sum()
# #     # print('sum no zeros norm entrophy', w)
# #
# #     # w = (a * entPost) + (b * entMSDA)
# #     # if w.sum() != 0:
# #     #     w /= w.sum()
# #     # print('sum norm entrophy', w)
# #
# #     # # the best
# #     # w = ((a * np.max(a)) + (b * np.max(b_norm))) / 2
# #     # print('sum  max values', w)
# #
# #     # w = (a * np.max(a)) + (b * np.max(b_norm))
# #     # if w.sum() != 0:
# #     #     w /= w.sum()
# #     # print('sum norm max values', w)
# #
# #     # # the best
# #     # AUX_a = (np.max(a) - (1 / len(a))) / (1 - (1 / len(a)))
# #     # AUX_b = (np.max(b_norm) - (1 / len(b_norm))) / (1 - (1 / len(b_norm)))
# #     # w = ((a * AUX_a) + (b * AUX_b)) / 2
# #     # print('sum COMPARE RANDOM', w)
# #
# #     AUX_a = (np.max(a) - (1 / len(a))) / (1 - (1 / len(a)))
# #     AUX_b = (np.max(b_norm) - (1 / len(b_norm))) / (1 - (1 / len(b_norm)))
# #     w = ((a * AUX_a) + (b_norm * AUX_b)) / 2
# #     # print('sum COMPARE RANDOM_ b_norm', w)
# #     weightsMSDA = w.copy()
# #
# #     #
# #     #
# #     # a = postProb_trainFeatures.copy()
# #     # b = weightsMSDA.copy()
# #     # b[a == 0] = 0
# #     # print('sum no zeros_2', (a + b) / 2)
# #     #
# #     #
# #     # postProb_trainFeatures /= 2
# #     # if weightsMSDA.sum() != 0:
# #     #     weightsMSDA /= 2
# #     # else:
# #     #     weightsMSDA = 0
# #     # weightsMSDA = weightsMSDA + postProb_trainFeatures
# #     # ############METHOD1
# #     # postProb_trainFeatures /= 2
# #     # weightsMSDA /= 2
# #     # aux = np.zeros(classes)
# #     # T = 0.8
# #     # weightsMSDA_Norm = weightsMSDA / weightsMSDA.sum()
# #     #
# #     # if weightsMSDA.sum() != 0:
# #     #
# #     #     if np.max(weightsMSDA) == weightsMSDA[np.argmax(postProb_trainFeatures)]:
# #     #         if np.max(weightsMSDA_Norm) >= T and np.max(postProb_trainFeatures) >= T:
# #     #             aux[np.argmax(postProb_trainFeatures)] = 1
# #     #         else:
# #     #             aux[np.argmax(postProb_trainFeatures)] = weightsMSDA[np.argmax(weightsMSDA)] + postProb_trainFeatures[
# #     #                 np.argmax(postProb_trainFeatures)]
# #     #     else:
# #     #         aux[np.argmax(weightsMSDA)] = weightsMSDA[np.argmax(weightsMSDA)] + \
# #     #                                        postProb_trainFeatures[np.argmax(weightsMSDA)]
# #     #         aux[np.argmax(postProb_trainFeatures)] = weightsMSDA[np.argmax(postProb_trainFeatures)] + \
# #     #                                                   postProb_trainFeatures[np.argmax(postProb_trainFeatures)]
# #     # else:
# #     #     aux = postProb_trainFeatures
# #     #
# #     # weightsMSDA = aux
# #
# #     # ########################### THE LAST EVALUATED
# #     # postProb_trainFeatures /= 2
# #     # aux = np.zeros(classes)
# #     # if weightsMSDA.sum() != 0:
# #     #     weightsMSDA /= 2
# #     #     if np.max(weightsMSDA) == weightsMSDA[np.argmax(postProb_trainFeatures)]:
# #     #
# #     #         aux[np.argmax(postProb_trainFeatures)] = weightsMSDA[np.argmax(weightsMSDA)] + postProb_trainFeatures[
# #     #             np.argmax(postProb_trainFeatures)]
# #     #     else:
# #     #         aux[np.argmax(weightsMSDA)] = weightsMSDA[np.argmax(weightsMSDA)] + \
# #     #                                       postProb_trainFeatures[np.argmax(weightsMSDA)]
# #     #         aux[np.argmax(postProb_trainFeatures)] = weightsMSDA[np.argmax(postProb_trainFeatures)] + \
# #     #                                                  postProb_trainFeatures[np.argmax(postProb_trainFeatures)]
# #     # else:
# #     #     aux[np.argmax(postProb_trainFeatures)] = postProb_trainFeatures[np.argmax(postProb_trainFeatures)]
# #     #
# #     # weightsMSDA = aux
# #
# #     # print('Final')
# #     # print(np.round(weightsMSDA, 2))
# #     # print('\n')
# #     if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
# #         w_labeledGestures = np.ones(classes) * shotStart
# #
# #         weightsUnlabeledGesture = weightsMSDA
# #         if type_DA == 'LDA':
# #             weightsUnlabeledGesture_Cov = np.zeros(classes)
# #             weightsUnlabeledGesture_Cov[np.argmax(weightsMSDA)] = 1
# #             w_labeledGestures_Cov = np.ones(classes) * shotStart
# #
# #         for cla in range(classes):
# #             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures[cla] + \
# #                                               weightsUnlabeledGesture[cla] * gestureMean
# #             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]
# #
# #             if type_DA == 'QDA':
# #                 adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures[cla] + \
# #                                                  weightsUnlabeledGesture[cla] * gestureCov
# #             elif type_DA == 'LDA':
# #                 adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture_Cov[cla]
# #
# #                 adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures[cla] + \
# #                                                  weightsUnlabeledGesture_Cov[cla] * gestureCov
# #
# #         sumMean = w_labeledGestures + weightsUnlabeledGesture
# #         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
# #         w_labeledGestures /= sumMean
# #
# #         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
# #
# #         if type_DA == 'LDA':
# #             sumCov = w_labeledGestures_Cov + weightsUnlabeledGesture_Cov
# #             weightsUnlabeledGesture_Norm_Cov = weightsUnlabeledGesture_Cov / sumCov
# #             w_labeledGestures_Cov /= sumCov
# #             covs = np.resize(gestureCov,
# #                              (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm_Cov
# #         elif type_DA == 'QDA':
# #             covs = np.resize(gestureCov,
# #                              (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
# #
# #         for cla in range(classes):
# #             adaptiveModel.at[cla, 'class'] = cla + 1
# #             adaptiveModel.at[cla, 'mean'] = means[:, cla] + weakModel['mean'].loc[cla] * w_labeledGestures[cla]
# #             if type_DA == 'QDA':
# #                 adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + weakModel['cov'].loc[cla] * w_labeledGestures[cla]
# #             elif type_DA == 'LDA':
# #                 adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
# #
# #     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
# #         wJMean = np.zeros(classes)
# #         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
# #         weightsUnlabeledGesture = weightsMSDA
# #         if type_DA == 'LDA':
# #             wJCov = np.zeros(classes)
# #             weightsUnlabeledGesture_Cov = np.zeros(classes)
# #             weightsUnlabeledGesture_Cov[np.argmax(weightsMSDA)] = 1
# #
# #         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
# #
# #         for cla in range(classes):
# #             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
# #             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
# #                                               gestureMean
# #             wJMean[cla] = currentModel['wMean_J'].loc[cla]
# #             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]
# #
# #             if type_DA == 'QDA':
# #                 JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
# #                 adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
# #                                                  gestureCov
# #             elif type_DA == 'LDA':
# #                 wJCov[cla] = currentModel['wCov_J'].loc[cla]
# #                 adaptiveModel.at[cla, 'wCov_J'] = currentModel['wCov_J'].loc[cla] + weightsUnlabeledGesture_Cov[cla]
# #
# #                 JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wCov_J'].loc[cla]
# #                 adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture_Cov[cla] * \
# #                                                  gestureCov
# #
# #         sumMean = wJMean + weightsUnlabeledGesture
# #         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
# #         wJMean /= sumMean
# #
# #         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
# #
# #         if type_DA == 'QDA':
# #             covs = np.resize(gestureCov,
# #                              (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
# #         elif type_DA == 'LDA':
# #             sumCov = wJCov + weightsUnlabeledGesture_Cov
# #             weightsUnlabeledGesture_Norm_Cov = weightsUnlabeledGesture_Cov / sumCov
# #             wJCov /= sumCov
# #             covs = np.resize(gestureCov,
# #                              (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm_Cov
# #
# #         for cla in range(classes):
# #             adaptiveModel.at[cla, 'class'] = cla + 1
# #             adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]
# #
# #             if type_DA == 'QDA':
# #                 adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]
# #             elif type_DA == 'LDA':
# #                 adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJCov[cla]
# #
# #     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
# #     return adaptiveModel, time.time() - t, weightsUnlabeledGesture
#
#
# def model_incre_gestures_weight_MSDA(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                                      labeledGesturesFeatures,
#                                      labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     numSamples = 50
#     labeledGesturesFeatures, labeledGesturesLabels = DA_BasedAdaptiveModels.subsetTraining(
#         labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     weightsMSDA = weightMSDA_reduce(currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures,
#                                     labeledGesturesLabels, type_DA)
#
#     weightsMSDA = np.array(weightsMSDA)
#
#     a = postProb_trainFeatures.copy()
#     b = weightsMSDA.copy()
#     if b.sum() != 0:
#         b_norm = b / b.sum()
#     else:
#         b_norm = b.copy()
#
#     w = ((a * entrophyVector(a)) + (b_norm * entrophyVector(b_norm))) / 2
#     # print('sum COMPARE RANDOM_ b_norm', w)
#     weightsMSDA = w.copy()
#
#     if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures = np.ones(classes) * shotStart
#
#         weightsUnlabeledGesture = weightsMSDA
#         if type_DA == 'LDA':
#             weightsUnlabeledGesture_Cov = np.zeros(classes)
#             weightsUnlabeledGesture_Cov[np.argmax(weightsMSDA)] = 1
#             w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures[cla] + \
#                                               weightsUnlabeledGesture[cla] * gestureMean
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]
#
#             if type_DA == 'QDA':
#                 adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures[cla] + \
#                                                  weightsUnlabeledGesture[cla] * gestureCov
#             elif type_DA == 'LDA':
#                 adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture_Cov[cla]
#
#                 adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures[cla] + \
#                                                  weightsUnlabeledGesture_Cov[cla] * gestureCov
#
#         sumMean = w_labeledGestures + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         w_labeledGestures /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         if type_DA == 'LDA':
#             sumCov = w_labeledGestures_Cov + weightsUnlabeledGesture_Cov
#             weightsUnlabeledGesture_Norm_Cov = weightsUnlabeledGesture_Cov / sumCov
#             w_labeledGestures_Cov /= sumCov
#             covs = np.resize(gestureCov,
#                              (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm_Cov
#         elif type_DA == 'QDA':
#             covs = np.resize(gestureCov,
#                              (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + weakModel['mean'].loc[cla] * w_labeledGestures[cla]
#             if type_DA == 'QDA':
#                 adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + weakModel['cov'].loc[cla] * w_labeledGestures[cla]
#             elif type_DA == 'LDA':
#                 adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#         weightsUnlabeledGesture = weightsMSDA
#         if type_DA == 'LDA':
#             wJCov = np.zeros(classes)
#             weightsUnlabeledGesture_Cov = np.zeros(classes)
#             weightsUnlabeledGesture_Cov[np.argmax(weightsMSDA)] = 1
#
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                               gestureMean
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]
#
#             if type_DA == 'QDA':
#                 JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#                 adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                                  gestureCov
#             elif type_DA == 'LDA':
#                 wJCov[cla] = currentModel['wCov_J'].loc[cla]
#                 adaptiveModel.at[cla, 'wCov_J'] = currentModel['wCov_J'].loc[cla] + weightsUnlabeledGesture_Cov[cla]
#
#                 JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wCov_J'].loc[cla]
#                 adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture_Cov[cla] * \
#                                                  gestureCov
#
#         sumMean = wJMean + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         wJMean /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         if type_DA == 'QDA':
#             covs = np.resize(gestureCov,
#                              (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#         elif type_DA == 'LDA':
#             sumCov = wJCov + weightsUnlabeledGesture_Cov
#             weightsUnlabeledGesture_Norm_Cov = weightsUnlabeledGesture_Cov / sumCov
#             wJCov /= sumCov
#             covs = np.resize(gestureCov,
#                              (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm_Cov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]
#
#             if type_DA == 'QDA':
#                 adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]
#             elif type_DA == 'LDA':
#                 adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJCov[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, weightsUnlabeledGesture
#
#
# def model_incre_samples_labels(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                                labeledGesturesFeatures,
#                                labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
#     t = time.time()
#
#     adaptiveModel = currentModel.copy()
#     if currentModel['# gestures'].loc[0] == shotStart:
#         for cla in range(classes):
#             adaptiveModel.at[cla, '# gestures'] = len(labeledGesturesLabels[labeledGesturesLabels == cla + 1])
#
#     for sample in range(len(trainFeatures[:, 0])):
#         x = trainFeatures[sample, :]
#         if type_DA == 'LDA':
#             cla = DA_Classifiers.predictedModelLDA(x, currentModel, classes,
#                                                    DA_Classifiers.LDA_Cov(currentModel, classes))
#         elif type_DA == 'QDA':
#             cla = DA_Classifiers.predictedModelQDA(x, currentModel, classes)
#         cla -= 1
#         mean = adaptiveModel['mean'].loc[cla]
#         cov = adaptiveModel['cov'].loc[cla]
#         N = adaptiveModel['# gestures'].loc[cla]
#
#         adaptiveModel.at[cla, 'mean'] = (mean * N + x) / (N + 1)
#
#         x_mean = np.resize(x - mean, (len(x), 1))
#         adaptiveModel.at[cla, 'cov'] = ((N - 1) / N) * cov + (1 / N) * np.dot(x_mean, x_mean.T.conj()) * (N / (N + 1))
#
#         adaptiveModel.at[cla, '# gestures'] = N + 1
#
#     return adaptiveModel, time.time() - t, 0
#
#
# def model_incre_samples_prob(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                              labeledGesturesFeatures,
#                              labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
#     t = time.time()
#
#     adaptiveModel = currentModel.copy()
#     if currentModel['# gestures'].loc[0] == shotStart:
#         for cla in range(classes):
#             adaptiveModel.at[cla, '# gestures'] = len(labeledGesturesLabels[labeledGesturesLabels == cla + 1])
#
#     for sample in range(len(trainFeatures[:, 0])):
#         x = trainFeatures[sample, :]
#         if type_DA == 'LDA':
#             probVector = DA_Classifiers.predictedModelLDA_Prob(x, currentModel, classes,
#                                                                DA_Classifiers.LDA_Cov(currentModel, classes))
#         elif type_DA == 'QDA':
#             probVector = DA_Classifiers.predictedModelQDA_Prob(x, currentModel, classes)
#         for cla in range(classes):
#             mean = adaptiveModel['mean'].loc[cla]
#             cov = adaptiveModel['cov'].loc[cla]
#             N = adaptiveModel['# gestures'].loc[cla]
#             p = probVector[cla]
#
#             adaptiveModel.at[cla, 'mean'] = (mean * N + p * x) / (N + p)
#
#             x_mean = np.resize(x - mean, (len(x), 1))
#             adaptiveModel.at[cla, 'cov'] = ((N - 1) / (N + p - 1)) * cov + (1 / (N + p - 1)) * \
#                                            np.dot(x_mean, x_mean.T.conj()) * (N * p / (N + p))
#
#             adaptiveModel.at[cla, '# gestures'] = N + p
#
#     return adaptiveModel, time.time() - t, 0
#
#
# def model_semi_gestures_labels(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                                labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart,
#                                unlabeledGesturesTotal, dataTotal):
#     t = time.time()
#
#     adaptiveModel = weakModel.copy()
#     for cla in range(classes):
#         adaptiveModel.at[cla, '# gestures'] = adaptiveModel.loc[0, '# gestures']
#
#     unlabeledGesturesTotal = np.array(unlabeledGesturesTotal)
#     numberFeatures = np.size(labeledGesturesFeatures, axis=1)
#     featuresUnlabeledGestures = unlabeledGesturesTotal[:, :numberFeatures]
#     numberUnlabeledGestures = unlabeledGesturesTotal[:, -1]
#     labelsList = labeledGesturesLabels
#     featuresList = labeledGesturesFeatures
#     for gesture in range(1, int(numberUnlabeledGestures[-1]) + 1):
#         X = featuresUnlabeledGestures[numberUnlabeledGestures == gesture]
#         prob_labels = post_probabilities_Calculation(X, weakModel, classes, type_DA)
#         cla = np.argmax(prob_labels) + 1
#         numberSegments = len(numberUnlabeledGestures[numberUnlabeledGestures == gesture])
#         labelsList = np.hstack((labelsList, np.ones(numberSegments) * cla))
#         featuresList = np.vstack((featuresList, X))
#     return modelCalculation_labels(featuresList, labelsList, classes), time.time() - t, 0
#
#
# def model_semi_gestures_weight(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                                labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart,
#                                unlabeledGesturesTotal, dataTotal):
#     t = time.time()
#
#     adaptiveModel = weakModel.copy()
#     for cla in range(classes):
#         adaptiveModel.at[cla, '# gestures'] = adaptiveModel.loc[0, '# gestures']
#
#     unlabeledGesturesTotal = np.array(unlabeledGesturesTotal)
#     numberFeatures = np.size(labeledGesturesFeatures, axis=1)
#     featuresUnlabeledGestures = unlabeledGesturesTotal[:, :numberFeatures]
#     numberUnlabeledGestures = unlabeledGesturesTotal[:, -1]
#     probList = dataTotal[:, -classes:]
#     featuresList = labeledGesturesFeatures
#     for gesture in range(1, int(numberUnlabeledGestures[-1]) + 1):
#         X = featuresUnlabeledGestures[numberUnlabeledGestures == gesture]
#         prob_labels = post_probabilities_Calculation(X, weakModel, classes, type_DA)
#         numberSegments = len(numberUnlabeledGestures[numberUnlabeledGestures == gesture])
#         probList = np.vstack((probList, np.ones((numberSegments, classes)) * prob_labels))
#         featuresList = np.vstack((featuresList, X))
#     return modelCalculation_prob(featuresList, probList, classes), time.time() - t, 0
#
#
# def model_semi_gestures_weight_MSDA(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                                     labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart,
#                                     unlabeledGesturesTotal, dataTotal):
#     t = time.time()
#
#     adaptiveModel = weakModel.copy()
#     for cla in range(classes):
#         adaptiveModel.at[cla, '# gestures'] = adaptiveModel.loc[0, '# gestures']
#
#     unlabeledGesturesTotal = np.array(unlabeledGesturesTotal)
#     numberFeatures = np.size(labeledGesturesFeatures, axis=1)
#     featuresUnlabeledGestures = unlabeledGesturesTotal[:, :numberFeatures]
#     numberUnlabeledGestures = unlabeledGesturesTotal[:, -1]
#     probList = dataTotal[:, -classes:]
#     featuresList = labeledGesturesFeatures
#     for gesture in range(1, int(numberUnlabeledGestures[-1]) + 1):
#         X = featuresUnlabeledGestures[numberUnlabeledGestures == gesture]
#         prob_labels = post_probabilities_Calculation(X, weakModel, classes, type_DA)
#         weightsMSDA = weightMSDA_reduce(weakModel, np.mean(X, axis=0), np.cov(X, rowvar=False), classes,
#                                         labeledGesturesFeatures,
#                                         labeledGesturesLabels, type_DA)
#         weightsMSDA = np.array(weightsMSDA)
#
#         if weightsMSDA.sum() != 0:
#             weightsMSDA_norm = weightsMSDA / weightsMSDA.sum()
#         else:
#             weightsMSDA_norm = weightsMSDA.copy()
#
#         w = ((prob_labels * entrophyVector(prob_labels)) + (weightsMSDA_norm * entrophyVector(weightsMSDA_norm))) / 2
#
#         numberSegments = len(numberUnlabeledGestures[numberUnlabeledGestures == gesture])
#         probList = np.vstack((probList, np.ones((numberSegments, classes)) * w))
#         featuresList = np.vstack((featuresList, X))
#     return modelCalculation_prob(featuresList, probList, classes), time.time() - t, 0
#
#
# def model_semi_samples_labels(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                               labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart,
#                               unlabeledGesturesTotal, dataTotal):
#     t = time.time()
#
#     unlabeledGesturesTotal = np.array(unlabeledGesturesTotal)
#     numberFeatures = np.size(labeledGesturesFeatures, axis=1)
#     labelsList = list(labeledGesturesLabels)
#     featuresList = list(labeledGesturesFeatures)
#     for sample in range(len(unlabeledGesturesTotal[:, 0])):
#         x = unlabeledGesturesTotal[sample, :numberFeatures]
#         if type_DA == 'LDA':
#             cla = DA_Classifiers.predictedModelLDA(x, weakModel, classes,
#                                                    DA_Classifiers.LDA_Cov(weakModel, classes))
#         elif type_DA == 'QDA':
#             cla = DA_Classifiers.predictedModelQDA(x, weakModel, classes)
#         labelsList.append(cla)
#         featuresList.append(x)
#     return modelCalculation_labels(np.array(featuresList), np.array(labelsList), classes), time.time() - t, 0
#
#
# def model_semi_samples_prob(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                             labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart,
#                             unlabeledGesturesTotal, dataTotal):
#     t = time.time()
#
#     unlabeledGesturesTotal = np.array(unlabeledGesturesTotal)
#     numberFeatures = np.size(labeledGesturesFeatures, axis=1)
#     probList = list(dataTotal[:, -classes:])
#     featuresList = list(labeledGesturesFeatures)
#     for sample in range(len(unlabeledGesturesTotal[:, 0])):
#
#         x = unlabeledGesturesTotal[sample, :numberFeatures]
#         if type_DA == 'LDA':
#             probVector = DA_Classifiers.predictedModelLDA_Prob(x, weakModel, classes,
#                                                                DA_Classifiers.LDA_Cov(weakModel, classes))
#         elif type_DA == 'QDA':
#             probVector = DA_Classifiers.predictedModelQDA_Prob(x, weakModel, classes)
#         probList.append(probVector)
#         featuresList.append(x)
#     return modelCalculation_prob(np.array(featuresList), np.array(probList), classes), time.time() - t, 0
#
#
# def model_semi_gestures_weight_MSDA_2(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                                       labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory,
#                                       shotStart,
#                                       unlabeledGesturesTotal, dataTotal):
#     t = time.time()
#
#     adaptiveModel = weakModel.copy()
#     for cla in range(classes):
#         adaptiveModel.at[cla, '# gestures'] = adaptiveModel.loc[0, '# gestures']
#
#     unlabeledGesturesTotal = np.array(unlabeledGesturesTotal)
#     numberFeatures = np.size(labeledGesturesFeatures, axis=1)
#     featuresUnlabeledGestures = unlabeledGesturesTotal[:, :numberFeatures]
#     numberUnlabeledGestures = unlabeledGesturesTotal[:, -1]
#     probList = dataTotal[:, -classes:]
#     featuresList = labeledGesturesFeatures
#     for gesture in range(1, int(numberUnlabeledGestures[-1]) + 1):
#         X = featuresUnlabeledGestures[numberUnlabeledGestures == gesture]
#         prob_labels = post_probabilities_Calculation(X, weakModel, classes, type_DA)
#         weightsMSDA = weightMSDA_reduce(weakModel, np.mean(X, axis=0), np.cov(X, rowvar=False), classes,
#                                         labeledGesturesFeatures,
#                                         labeledGesturesLabels, type_DA)
#         weightsMSDA = np.array(weightsMSDA)
#
#         if weightsMSDA.sum() != 0:
#             weightsMSDA_norm = weightsMSDA / weightsMSDA.sum()
#         else:
#             weightsMSDA_norm = weightsMSDA.copy()
#
#         # w = ((prob_labels * entrophyVector(prob_labels)) + (weightsMSDA_norm * entrophyVector(weightsMSDA_norm))) / 2
#         w = prob_labels * weightsMSDA_norm
#
#         numberSegments = len(numberUnlabeledGestures[numberUnlabeledGestures == gesture])
#         probList = np.vstack((probList, np.ones((numberSegments, classes)) * w))
#         featuresList = np.vstack((featuresList, X))
#     return modelCalculation_prob(featuresList, probList, classes), time.time() - t, 0
#
#
# def model_semi_gestures_weight_MSDA_3(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                                       labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory,
#                                       shotStart,
#                                       unlabeledGesturesTotal, dataTotal):
#     t = time.time()
#
#     adaptiveModel = weakModel.copy()
#     for cla in range(classes):
#         adaptiveModel.at[cla, '# gestures'] = adaptiveModel.loc[0, '# gestures']
#
#     unlabeledGesturesTotal = np.array(unlabeledGesturesTotal)
#     numberFeatures = np.size(labeledGesturesFeatures, axis=1)
#     featuresUnlabeledGestures = unlabeledGesturesTotal[:, :numberFeatures]
#     numberUnlabeledGestures = unlabeledGesturesTotal[:, -1]
#     probList = dataTotal[:, -classes:]
#     featuresList = labeledGesturesFeatures
#     for gesture in range(1, int(numberUnlabeledGestures[-1]) + 1):
#         X = featuresUnlabeledGestures[numberUnlabeledGestures == gesture]
#         prob_labels = post_probabilities_Calculation(X, weakModel, classes, type_DA)
#         weightsMSDA = weightMSDA_reduce(weakModel, np.mean(X, axis=0), np.cov(X, rowvar=False), classes,
#                                         labeledGesturesFeatures,
#                                         labeledGesturesLabels, type_DA)
#         weightsMSDA = np.array(weightsMSDA)
#
#         # if weightsMSDA.sum() != 0:
#         #     weightsMSDA_norm = weightsMSDA / weightsMSDA.sum()
#         # else:
#         #     weightsMSDA_norm = weightsMSDA.copy()
#
#         # w = ((prob_labels * entrophyVector(prob_labels)) + (weightsMSDA_norm * entrophyVector(weightsMSDA_norm))) / 2
#         w = prob_labels * weightsMSDA
#
#         numberSegments = len(numberUnlabeledGestures[numberUnlabeledGestures == gesture])
#         probList = np.vstack((probList, np.ones((numberSegments, classes)) * w))
#         featuresList = np.vstack((featuresList, X))
#     return modelCalculation_prob(featuresList, probList, classes), time.time() - t, 0


# def model_PostProb_MSDA(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel, labeledGesturesFeatures,
#                         labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     numSamples = 50
#     labeledGesturesFeatures, labeledGesturesLabels = DA_BasedAdaptiveModels.subsetTraining(
#         labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     weightsMSDA = weightMSDA_reduce(currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures,
#                                     labeledGesturesLabels, type_DA)
#
#     weightsMSDA = np.array(weightsMSDA) / 2
#     postProb_trainFeatures /= 2
#
#     print(type_DA)
#     print(weightsMSDA)
#     print(postProb_trainFeatures)
#
#     if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures = np.ones(classes) * shotStart
#
#         weightsUnlabeledGesture = postProb_trainFeatures + weightsMSDA
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures[cla] + \
#                                               weightsUnlabeledGesture[cla] * gestureMean
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]
#
#             adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures[cla] + \
#                                              weightsUnlabeledGesture[cla] * gestureCov
#
#         sumMean = w_labeledGestures + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         w_labeledGestures /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + weakModel['mean'].loc[cla] * w_labeledGestures[cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + weakModel['cov'].loc[cla] * w_labeledGestures[cla]
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#         weightsUnlabeledGesture = postProb_trainFeatures + weightsMSDA
#
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                               gestureMean
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]
#
#             JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                              gestureCov
#
#         sumMean = wJMean + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         wJMean /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]
#
#             adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, weightsUnlabeledGesture
#
#
#
# def model_PostProb_MSDA_multiplication(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                                        labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory,
#                                        shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     numSamples = 50
#     labeledGesturesFeatures, labeledGesturesLabels = DA_BasedAdaptiveModels.subsetTraining(
#         labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     weightsMSDA = weightMSDA_reduce(currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures,
#                                     labeledGesturesLabels, type_DA)
#
#     weightsMSDA = np.array(weightsMSDA)
#
#     if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures = np.ones(classes) * shotStart
#
#         weightsUnlabeledGesture = postProb_trainFeatures * weightsMSDA
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures[cla] + \
#                                               weightsUnlabeledGesture[cla] * gestureMean
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]
#
#             adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures[cla] + \
#                                              weightsUnlabeledGesture[cla] * gestureCov
#
#         sumMean = w_labeledGestures + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         w_labeledGestures /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + weakModel['mean'].loc[cla] * w_labeledGestures[cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + weakModel['cov'].loc[cla] * w_labeledGestures[cla]
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#         weightsUnlabeledGesture = postProb_trainFeatures * weightsMSDA
#
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                               gestureMean
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]
#
#             JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                              gestureCov
#
#         sumMean = wJMean + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         wJMean /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]
#
#             adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, weightsUnlabeledGesture
#
#
# def model_PostProb_JS_multiplication(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                                      labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory,
#                                      shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     numSamples = 50
#     labeledGesturesFeatures, labeledGesturesLabels = DA_BasedAdaptiveModels.subsetTraining(
#         labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     weightsMSDA = weight_MSDA_JS(currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures,
#                                  labeledGesturesLabels, type_DA)
#
#     # weightsMSDA = np.array(weightsMSDA)
#     print('weightsJS', type_DA)
#     print(np.round(weightsMSDA, 2))
#     print('postProb_trainFeatures')
#     print(np.round(postProb_trainFeatures, 2))
#
#     if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures = np.ones(classes) * shotStart
#
#         weightsUnlabeledGesture = weightsMSDA
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures[cla] + \
#                                               weightsUnlabeledGesture[cla] * gestureMean
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]
#
#             adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures[cla] + \
#                                              weightsUnlabeledGesture[cla] * gestureCov
#
#         sumMean = w_labeledGestures + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         w_labeledGestures /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + weakModel['mean'].loc[cla] * w_labeledGestures[cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + weakModel['cov'].loc[cla] * w_labeledGestures[cla]
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#         weightsUnlabeledGesture = weightsMSDA
#
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                               gestureMean
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]
#
#             JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                              gestureCov
#
#         sumMean = wJMean + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         wJMean /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]
#
#             adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, weightsUnlabeledGesture
#
#
# def model_PostProbOld(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel, labeledGesturesFeatures,
#                       labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#     postProb_trainFeatures = post_probabilities_Calculation(trainFeatures, weakModel, classes, type_DA)
#
#     if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures = np.ones(classes) * shotStart
#
#         weightsUnlabeledGesture = postProb_trainFeatures
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures[cla] + \
#                                               weightsUnlabeledGesture[cla] * gestureMean
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]
#
#             adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures[cla] + \
#                                              weightsUnlabeledGesture[cla] * gestureCov
#
#         sumMean = w_labeledGestures + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         w_labeledGestures /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + weakModel['mean'].loc[cla] * w_labeledGestures[cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + weakModel['cov'].loc[cla] * w_labeledGestures[cla]
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#         weightsUnlabeledGesture = postProb_trainFeatures
#
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                               gestureMean
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]
#
#             JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                              gestureCov
#
#         sumMean = wJMean + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         wJMean /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]
#
#             adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, weightsUnlabeledGesture
#
#
# def model_PostProb_PostProbOld_multiplication(currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                                               labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory,
#                                               shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     numSamples = 50
#     labeledGesturesFeatures, labeledGesturesLabels = DA_BasedAdaptiveModels.subsetTraining(
#         labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     # weightsMSDA = weightMSDA_reduce(currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures,
#     #                                 labeledGesturesLabels, type_DA)
#     #
#     # weightsMSDA = np.array(weightsMSDA)
#
#     weightsMSDA = post_probabilities_Calculation(trainFeatures, weakModel, classes, type_DA)
#
#     if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures = np.ones(classes) * shotStart
#
#         weightsUnlabeledGesture = postProb_trainFeatures * weightsMSDA
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures[cla] + \
#                                               weightsUnlabeledGesture[cla] * gestureMean
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]
#
#             adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures[cla] + \
#                                              weightsUnlabeledGesture[cla] * gestureCov
#
#         sumMean = w_labeledGestures + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         w_labeledGestures /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + weakModel['mean'].loc[cla] * w_labeledGestures[cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + weakModel['cov'].loc[cla] * w_labeledGestures[cla]
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#         weightsUnlabeledGesture = postProb_trainFeatures * weightsMSDA
#
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                               gestureMean
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]
#
#             JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
#                                              gestureCov
#
#         sumMean = wJMean + weightsUnlabeledGesture
#         weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
#         wJMean /= sumMean
#
#         means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]
#
#             adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, weightsUnlabeledGesture


# def model_PostProb_MSDA_JS(currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                            labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     numberUnlabeledGestures = len(unlabeledGestures.index)
#     if numberUnlabeledGestures == 0:
#         currentModel.at[0, '# gestures'] = shotStart
#     elif samplesInMemory >= numberUnlabeledGestures > 0:
#         unlabeledGestures = postProbabilities_weights_Calculation_JS(unlabeledGestures, currentModel, classes, type_DA,
#                                                                      numberUnlabeledGestures, labeledGesturesFeatures,
#                                                                      labeledGesturesLabels, weights=True, post=True)
#     else:
#         unlabeledGestures = unlabeledGestures.tail(samplesInMemory).reset_index(drop=True)
#         unlabeledGestures = postProbabilities_weights_Calculation_JS(unlabeledGestures, currentModel, classes, type_DA,
#                                                                      samplesInMemory, labeledGesturesFeatures,
#                                                                      labeledGesturesLabels, weights=True, post=True)
#     # print(type_DA)
#     # print(unlabeledGestures)
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     wGestureMean, wGestureCov = weight_MSDA_JS(currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures,
#                                                labeledGesturesLabels, type_DA)
#
#     # wGestureMean = []
#     # wGestureCov = []
#     #
#     # for cla in range(classes):
#     #     wGestureMean.append(
#     #         weightPerPersonMean_KL(currentModel, gestureMean, cla, classes, labeledGesturesFeatures,
#     #                                labeledGesturesLabels,
#     #                                type_DA))
#     #
#     #     wGestureCov.append(
#     #         weightPerPersonCov_KL(currentModel, gestureCov, cla, classes, labeledGesturesFeatures,
#     #                               labeledGesturesLabels,
#     #                               type_DA))
#     numSamples = 50
#     labeledGesturesFeatures, labeledGesturesLabels = DA_BasedAdaptiveModels.subsetTraining(
#         labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)
#
#     new_row = {'mean': gestureMean, 'cov': gestureCov, 'postProb': postProb_trainFeatures, 'wMean': wGestureMean,
#                'wCov': wGestureCov, 'features': trainFeatures}
#
#     unlabeledGestures = unlabeledGestures.append(new_row, ignore_index=True)
#
#     print('\nJS ', type_DA)
#     print('wGestureMean', np.around(wGestureMean, 2))
#     print('wGestureCov', np.around(wGestureCov, 2))
#
#     if samplesInMemory > currentModel['# gestures'].loc[0] - shotStart:
#         # print('less than memmory')
#         w_labeledGestures_Mean = np.ones(classes) * shotStart
#         w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         weightsMean = unlabeledGestures['postProb'].values * unlabeledGestures['wMean'].values
#         sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         w_labeledGestures_Mean /= sumMean
#         # print('mean weights')
#         # print(np.around(weightsMean, 2))
#         # print('mean few')
#         # print(np.around(w_labeledGestures_Mean, 2))
#         means = np.resize(unlabeledGestures['mean'].values, (classes, numberUnlabeledGestures + 1)).T * weightsMean
#
#         # if type_DA == 'QDA':
#         weightsCov = unlabeledGestures['postProb'].values * unlabeledGestures['wCov'].values
#         sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         w_labeledGestures_Cov /= sumCov
#         # weightsCov = np.nan_to_num(weightsCov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, numberUnlabeledGestures + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + weakModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#
#
#     elif samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         # print('equal than memmory')
#         w_labeledGestures_Mean = np.ones(classes) * shotStart
#         w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         weightsMean = unlabeledGestures['postProb'].values * unlabeledGestures['wMean'].values
#         # if type_DA == 'QDA':
#         weightsCov = unlabeledGestures['postProb'].values * unlabeledGestures['wCov'].values
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * unlabeledGestures['mean'].loc[0]
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#             # if type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
#                 cla] * unlabeledGestures['cov'].loc[0]
#             adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures_Cov[cla] + weightsCov[0][cla]
#
#         sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         w_labeledGestures_Mean /= sumMean
#         # print('mean weights')
#         # print(np.around(weightsMean, 2))
#         # print('mean equal')
#         # print(np.around(w_labeledGestures_Mean, 2))
#         means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         # if type_DA == 'QDA':
#         sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         w_labeledGestures_Cov /= sumCov
#         # weightsCov = np.nan_to_num(weightsCov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + weakModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             #
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         # print('greater than memmory')
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#         weightsMean = unlabeledGestures['postProb'].values * unlabeledGestures['wMean'].values
#
#         # if type_DA == 'QDA':
#         wJCov = np.zeros(classes)
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#         weightsCov = unlabeledGestures['postProb'].values * unlabeledGestures['wCov'].values
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsMean[0][cla] * \
#                                               unlabeledGestures['mean'].loc[0]
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsMean[0][cla]
#             # if type_DA == 'QDA':
#             JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wCov_J'].loc[cla]
#             adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsCov[0][cla] * \
#                                              unlabeledGestures['cov'].loc[0]
#             wJCov[cla] = currentModel['wCov_J'].loc[cla]
#             adaptiveModel.at[cla, 'wCov_J'] = currentModel['wCov_J'].loc[cla] + weightsCov[0][cla]
#
#         sumMean = wJMean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         wJMean /= sumMean
#         # print('mean weights')
#         # print(np.around(weightsMean, 2))
#         # print('mean wJ')
#         # print(np.around(wJMean, 2))
#         means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         # if type_DA == 'QDA':
#         sumCov = wJCov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         wJCov /= sumCov
#         # weightsCov = np.nan_to_num(weightsCov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + JMean[cla, :] * wJMean[cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + JCov[cla, :, :] * wJCov[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, unlabeledGestures

# def postProbabilities_weights_Calculation_JS(unlabeledGestures, model, classes, type_DA, numberGestures,
#                                              labeledGesturesFeatures, labeledGesturesLabels, weights=True, post=True):
#     if post:
#         post_probabilities = []
#     if weights:
#         weightsMean = []
#         weightsCov = []
#
#     for i in range(numberGestures):
#         if post:
#             post_probabilities.append(
#                 post_probabilities_Calculation(unlabeledGestures['features'].loc[i], model, classes, type_DA))
#         if weights:
#             wGestureMean, wGestureCov = weight_MSDA_JS(model, unlabeledGestures['mean'].loc[i],
#                                                        unlabeledGestures['cov'].loc[i], classes,
#                                                        labeledGesturesFeatures, labeledGesturesLabels, type_DA)
#
#             weightsMean.append(np.array(wGestureMean))
#             weightsCov.append(np.array(wGestureCov))
#
#     if post:
#         unlabeledGestures['postProb'] = post_probabilities
#     if weights:
#         unlabeledGestures['wMean'] = weightsMean
#         unlabeledGestures['wCov'] = weightsCov
#     return unlabeledGestures
#
#
# def distributions_js(mean1, cov1, mean2, cov2, n_samples=10 ** 5):
#     # jensen shannon divergence. (Jensen shannon distance is the square root of the divergence)
#     # all the logarithms are defined as log2 (because of information entrophy)
#     distribution_p = stats.multivariate_normal(mean1, cov1)
#     distribution_q = stats.multivariate_normal(mean2, cov2)
#
#     X = distribution_p.rvs(n_samples)
#     p_X = distribution_p.pdf(X)
#     q_X = distribution_q.pdf(X)
#     log_mix_X = np.log2(p_X + q_X)
#
#     Y = distribution_q.rvs(n_samples)
#     p_Y = distribution_p.pdf(Y)
#     q_Y = distribution_q.pdf(Y)
#     log_mix_Y = np.log2(p_Y + q_Y)
#
#     return 1 - (abs(np.log2(p_X).mean() - (log_mix_X.mean() - np.log2(2)) + np.log2(q_Y).mean() - (
#             log_mix_Y.mean() - np.log2(2))) / 2)


# def weight_MSDA_JS(currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures, labeledGesturesLabels,
#                    type_DA):
#     if type_DA == 'QDA':
#         JS_Unlabeled2Labeled = []
#         for cla in range(classes):
#             mean1 = np.mean(labeledGesturesFeatures[labeledGesturesLabels == cla + 1, :], axis=0)
#             cov1 = np.cov(labeledGesturesFeatures[labeledGesturesLabels == cla + 1, :], rowvar=False)
#             JS_Unlabeled2Labeled.append(distributions_js(mean1, cov1, gestureMean, gestureCov))
#         JS_Unlabeled2Labeled = np.array(JS_Unlabeled2Labeled)
#         JS_Unlabeled2Labeled /= JS_Unlabeled2Labeled.sum()
#
#         JS_Unlabeled2Model = []
#         for cla in range(classes):
#             covModel = currentModel.loc[cla, 'cov']
#             meanModel = currentModel.loc[cla, 'mean']
#             JS_Unlabeled2Model.append(distributions_js(meanModel, covModel, gestureMean, gestureCov))
#         JS_Unlabeled2Model = np.array(JS_Unlabeled2Model)
#         JS_Unlabeled2Model /= JS_Unlabeled2Model.sum()
#
#         matrix_D = matrixDivergences(labeledGesturesFeatures, labeledGesturesLabels, currentModel, classes, type_DA)
#
#         JS_Labeled2Model_Unlabeled = np.zeros(classes)
#
#         for cla in range(classes):
#             matrix_D_aux = matrix_D.copy()
#
#             matrix_D_aux[cla, :] = rowDivergences(labeledGesturesFeatures, labeledGesturesLabels, gestureMean,
#                                                   gestureCov, classes)
#             TP = matrix_D_aux[cla, cla]
#             TN = np.trace(matrix_D_aux) - TP
#             FP = np.sum(matrix_D_aux[cla, :]) - TP
#             FN = matrix_D_aux.sum() - np.trace(matrix_D_aux) - FP
#             JS_Labeled2Model_Unlabeled[cla] = mcc(TP, TN, FP, FN)
#         if JS_Labeled2Model_Unlabeled.sum() != 0:
#             JS_Labeled2Model_Unlabeled /= JS_Labeled2Model_Unlabeled.sum()
#     elif type_DA == 'LDA':
#         JS_Unlabeled2Labeled = []
#         LDA_Cov = np.cov(labeledGesturesFeatures[labeledGesturesLabels == 1, :], rowvar=False)
#         for cla in range(1, classes):
#             LDA_Cov += np.cov(labeledGesturesFeatures[labeledGesturesLabels == cla + 1, :], rowvar=False)
#         LDA_Cov /= classes
#         for cla in range(classes):
#             mean1 = np.mean(labeledGesturesFeatures[labeledGesturesLabels == cla + 1, :], axis=0)
#             JS_Unlabeled2Labeled.append(distributions_js(mean1, LDA_Cov, gestureMean, gestureCov))
#         JS_Unlabeled2Labeled = np.array(JS_Unlabeled2Labeled)
#         JS_Unlabeled2Labeled /= JS_Unlabeled2Labeled.sum()
#
#         JS_Unlabeled2Model = []
#         LDA_Cov = DA_Classifiers.LDA_Cov(currentModel, classes)
#         for cla in range(classes):
#             meanModel = currentModel.loc[cla, 'mean']
#             JS_Unlabeled2Model.append(distributions_js(meanModel, LDA_Cov, gestureMean, gestureCov))
#         JS_Unlabeled2Model = np.array(JS_Unlabeled2Model)
#         JS_Unlabeled2Model /= JS_Unlabeled2Model.sum()
#
#         JS_Labeled2Model_Unlabeled = np.zeros(classes)
#         for cla in range(classes):
#             currentModel_aux = currentModel.copy()
#             currentModel_aux.loc[cla, 'mean'] = gestureMean
#             currentModel_aux.loc[cla, 'cov'] = gestureCov
#             matrix_D = matrixDivergences(labeledGesturesFeatures, labeledGesturesLabels, currentModel_aux, classes,
#                                          type_DA)
#             TP = matrix_D[cla, cla]
#             TN = np.trace(matrix_D) - TP
#             FP = np.sum(matrix_D[cla, :]) - TP
#             FN = matrix_D.sum() - np.trace(matrix_D) - FP
#             JS_Labeled2Model_Unlabeled[cla] = mcc(TP, TN, FP, FN)
#         if JS_Labeled2Model_Unlabeled.sum() != 0:
#             JS_Labeled2Model_Unlabeled /= JS_Labeled2Model_Unlabeled.sum()
#     # print('JS_Unlabeled2Labeled')
#     # print(np.round(JS_Unlabeled2Labeled, 2))
#     # print('JS_Unlabeled2Model')
#     # print(np.round(JS_Unlabeled2Model, 2))
#     # print('JS_Labeled2Model_Unlabeled')
#     # print(np.round(JS_Labeled2Model_Unlabeled, 2))
#
#     # JS_Unlabeled2Labeled=entrophyVector(JS_Unlabeled2Labeled)
#     # JS_Unlabeled2Model = entrophyVector(JS_Unlabeled2Model)
#     # JS_Labeled2Model_Unlabeled = entrophyVector(JS_Labeled2Model_Unlabeled)
#
#     print('JS_Unlabeled2Labeled')
#     print(np.round(JS_Unlabeled2Labeled, 2))
#     print('JS_Unlabeled2Model')
#     print(np.round(JS_Unlabeled2Model, 2))
#     print('JS_Labeled2Model_Unlabeled')
#     print(np.round(JS_Labeled2Model_Unlabeled, 2))
#     weightTotal = JS_Unlabeled2Model + JS_Unlabeled2Labeled
#     return weightTotal / weightTotal.sum()


# def matrixDivergences(Features, Labels, model, classes, type_DA):
#     matrix = np.zeros((classes, classes))
#     if type_DA == 'QDA':
#         for i in range(classes):
#             meanSamples = np.mean(Features[Labels == i + 1, :], axis=0)
#             covSamples = np.cov(Features[Labels == i + 1, :], rowvar=False)
#             for j in range(classes):
#                 covModel = model.loc[j, 'cov']
#                 meanModel = model.loc[j, 'mean']
#                 matrix[j, i] = distributions_js(meanSamples, covSamples, meanModel, covModel)
#     elif type_DA == 'LDA':
#         LDA_Cov = DA_Classifiers.LDA_Cov(model, classes)
#         for i in range(classes):
#             meanSamples = np.mean(Features[Labels == i + 1, :], axis=0)
#             covSamples = np.cov(Features[Labels == i + 1, :], rowvar=False)
#             for j in range(classes):
#                 meanModel = model.loc[j, 'mean']
#                 matrix[j, i] = distributions_js(meanSamples, covSamples, meanModel, LDA_Cov)
#     for cla in range(classes):
#         matrix[cla, :] /= np.sum(matrix[cla, :])
#
#     return matrix
#
#
# def rowDivergences(Features, Labels, meanModel, covModel, classes):
#     row = np.zeros((1, classes))
#     for i in range(classes):
#         meanSamples = np.mean(Features[Labels == i + 1, :], axis=0)
#         covSamples = np.cov(Features[Labels == i + 1, :], rowvar=False)
#         row[0, i] = distributions_js(meanSamples, covSamples, meanModel, covModel)
#     row[0, :] /= np.sum(row[0, :])
#     return row
#
#
# def mcc_from_matrixDivergences(matrix, classes, currentClass):
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#     for cla in range(classes):
#         if np.argmin(matrix[:, cla]) == cla:
#             if cla == currentClass:
#                 TP += 1
#             else:
#                 TN += 1
#         else:
#             if cla == currentClass:
#                 FN += 1
#             else:
#                 FP += 1
#     return mcc(TP, TN, FP, FN)
#
#
# def modelCalculation_labels(features, labels, classes):
#     model = pd.DataFrame(columns=['cov', 'mean', 'class'])
#     for cla in range(classes):
#         model.at[cla, 'mean'] = np.mean(features[labels == cla + 1], axis=0)
#         model.at[cla, 'cov'] = np.cov(features[labels == cla + 1], rowvar=False)
#         model.at[cla, 'class'] = cla + 1
#     return model
#
#
# def modelCalculation_prob(features, prob, classes):
#     model = pd.DataFrame(columns=['cov', 'mean', 'class', 'w_cov_accumulated', 'LDAcov'])
#     for cla in range(classes):
#         X = features * np.resize(prob[:, cla], (len(prob[:, cla]), 1))
#         model.at[cla, 'mean'] = np.sum(X, axis=0) / np.sum(prob[:, cla])
#         X_mean = features - model.loc[cla, 'mean']
#         cov = np.zeros((np.size(features, axis=1), np.size(features, axis=1)))
#         for i in range(len(prob[:, cla])):
#             x = np.resize(X_mean[i, :], (np.size(X_mean, axis=1), 1))
#             cov += prob[i, cla] * np.dot(x, x.T)
#         model.at[cla, 'cov'] = cov / (np.sum(prob[:, cla]) - 1)
#         model.at[cla, 'class'] = cla + 1
#         model.at[cla, 'w_cov_accumulated'] = 1
#     model.at[0, 'LDAcov'] = DA_Classifiers.LDA_Cov_weights(model)
#     return model
#
#
# # %% Divergences
# def KLdivergence(mean0, mean1, k, cov0, cov1):
#     exp1 = np.trace(np.dot(np.linalg.inv(cov1), cov0))
#     exp2 = np.dot(np.dot((mean1 - mean0).T, np.linalg.inv(cov1)), (mean1 - mean0))
#     exp3 = np.log(np.linalg.det(cov1) / np.linalg.det(cov0))
#     return 0.5 * (exp1 + exp2 - k + exp3)
#
#
# def JSdivergence(mean0, mean1, k, cov0, cov1):
#     meanM = (mean0 + mean1) / 2
#     covM = (cov0 + cov1) / 2
#     js = KLdivergence(mean0, meanM, k, cov0, covM) + KLdivergence(mean1, meanM, k, cov1, covM)
#     # js /= np.log(2)
#     return js / 2


######
#
# def post_probabilities_Calculation_Prob(features, model, classes, type_DA):
#     actualPredictor = np.zeros(classes)
#
#     if type_DA == 'LDA':
#         LDACov = DA_Classifiers.LDA_Cov(model, classes)
#         for i in range(np.size(features, axis=0)):
#             actualPredictor += DA_Classifiers.predictedModelLDA_Prob(features[i, :], model, classes, LDACov)
#     elif type_DA == 'QDA':
#         for i in range(np.size(features, axis=0)):
#             actualPredictor += DA_Classifiers.predictedModelQDA_Prob(features[i, :], model, classes)
#
#     return actualPredictor / np.size(features, axis=0)
#
#
# def post_probabilities_Calculation_Prob2(features, model, classes, type_DA):
#     actualPredictor = np.zeros(classes)
#
#     if type_DA == 'LDA':
#         LDACov = DA_Classifiers.LDA_Cov(model, classes)
#         for i in range(np.size(features, axis=0)):
#             aux = DA_Classifiers.predictedModelLDA_Prob(features[i, :], model, classes, LDACov)
#             actualPredictor[np.argmax(aux)] += np.max(aux)
#     elif type_DA == 'QDA':
#         for i in range(np.size(features, axis=0)):
#             aux = DA_Classifiers.predictedModelQDA_Prob(features[i, :], model, classes)
#             actualPredictor[np.argmax(aux)] += np.max(aux)
#
#     return actualPredictor / np.size(features, axis=0)


# %% Models Old
# %% reduce random the datasets
# def subsetTraining_One(trainFeatures, numSamples):
#     return trainFeatures[np.random.choice(len(trainFeatures), size=numSamples)]
# def model_PostProb_MSDA_reduce(currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures,
#                                weakModel, labeledGesturesFeatures, labeledGesturesLabels, type_DA,
#                                samplesInMemory, shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     numSamples = 50
#     labeledGesturesFeatures, labeledGesturesLabels = DA_BasedAdaptiveModels.subsetTraining(
#         labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)
#
#     numberUnlabeledGestures = len(unlabeledGestures.index)
#     if numberUnlabeledGestures == 0:
#         currentModel.at[0, '# gestures'] = shotStart
#     elif samplesInMemory >= numberUnlabeledGestures > 0:
#         unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
#                                                                   numberUnlabeledGestures, labeledGesturesFeatures,
#                                                                   labeledGesturesLabels, weights=True, post=True)
#     else:
#         unlabeledGestures = unlabeledGestures.tail(samplesInMemory).reset_index(drop=True)
#         unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
#                                                                   samplesInMemory, labeledGesturesFeatures,
#                                                                   labeledGesturesLabels, weights=True, post=True)
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     # wGestureMean = []
#     # wGestureCov = []
#
#     # currentModelMatrix, gestureMatrix = discriminantMatrix(currentModel, gestureMean, gestureCov, classes,
#     #                                                        labeledGesturesFeatures, labeledGesturesLabels, type_DA)
#
#     wGestureMean = weightMSDA_reduce(currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures,
#                                      labeledGesturesLabels, type_DA)
#
#     # w = []
#     # for cla in range(classes):
#     #     # wGestureMean.append(
#     #     #     weightPerPersonMean(currentModel, gestureMean, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,
#     #     #                         type_DA))
#     #     #
#     #     # wGestureCov.append(
#     #     #     weightPerPersonCov(currentModel, gestureCov, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,
#     #     #                        type_DA))
#     #
#     #     w.append(weightMSDA(currentModel, gestureMean, gestureCov, cla, classes, labeledGesturesFeatures,
#     #                         labeledGesturesLabels, type_DA))
#
#     wGestureMean[:] = [x / 2 for x in wGestureMean]
#     postProb_trainFeatures /= 2
#     wGestureCov = wGestureMean.copy()
#     # postProb_trainFeatures2 = post_probabilities_Calculation(trainFeatures, weakModel, classes, type_DA)
#
#     # wGestureMean = np.array(wGestureMean)
#     # if wGestureMean.sum() != 0:
#     #     wGestureMean /= wGestureMean.sum()
#     # wGestureCov = np.array(wGestureCov)
#     # if wGestureCov.sum() != 0:
#     #     wGestureCov /= wGestureCov.sum()
#     #
#     #
#     # # weightTotal = wGestureMean * postProb_trainFeatures
#     # # if weightTotal.sum() != 0:
#     # #     weightTotal /= weightTotal.sum()
#     #
#     # wGestureMean = list(wGestureMean)
#     # wGestureCov = list(wGestureCov)
#     # # wGestureCov = wGestureMean.copy()
#
#     # print('\nmodel_PostProb_MSDA_1w_sum_Nonorm_Reduce ', type_DA)
#     # print('postProb_2', np.around(postProb_trainFeatures2, 2))
#     # print('postProb', np.around(np.array(postProb_trainFeatures), 2))
#     # print('wGestureMean', np.around(wGestureMean, 2))
#     # print('wGestureCov', np.around(wGestureCov, 2))
#     # # print('weightTotal', np.around(weightTotal, 2))
#
#     trainFeatures = subsetTraining_One(trainFeatures, numSamples)
#     new_row = {'mean': gestureMean, 'cov': gestureCov, 'postProb': postProb_trainFeatures, 'wMean': wGestureMean,
#                'wCov': wGestureCov, 'features': trainFeatures}
#     # new_row = {'mean': gestureMean, 'cov': gestureCov, 'postProb': postProb_trainFeatures, 'wMean': wGestureMean,
#     #            'wCov': wGestureCov, 'features': trainFeatures, 'weightTotal': weightTotal}
#
#     unlabeledGestures = unlabeledGestures.append(new_row, ignore_index=True)
#
#     if samplesInMemory > currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures_Mean = np.ones(classes) * shotStart
#         w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         weightsMean = unlabeledGestures['postProb'].values + unlabeledGestures['wMean'].values
#         sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         w_labeledGestures_Mean /= sumMean
#         # print('mean weights')
#         # print(weightsMean)
#         # print('mean few')
#         # print(w_labeledGestures_Mean)
#
#         means = np.resize(unlabeledGestures['mean'].values, (classes, numberUnlabeledGestures + 1)).T * weightsMean
#
#         weightsCov = unlabeledGestures['postProb'].values + unlabeledGestures['wCov'].values
#         sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         w_labeledGestures_Cov /= sumCov
#
#         # print('cov weights')
#         # print(weightsCov)
#         # print('cov few')
#         # print(w_labeledGestures_Cov)
#
#         covs = np.resize(unlabeledGestures['cov'], (classes, numberUnlabeledGestures + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + weakModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#
#
#     elif samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures_Mean = np.ones(classes) * shotStart
#         w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         weightsMean = unlabeledGestures['postProb'].values + unlabeledGestures['wMean'].values
#
#         weightsCov = unlabeledGestures['postProb'].values + unlabeledGestures['wCov'].values
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * unlabeledGestures['mean'].loc[0]
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#
#             adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
#                 cla] * unlabeledGestures['cov'].loc[0]
#             adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures_Cov[cla] + weightsCov[0][cla]
#
#         sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         w_labeledGestures_Mean /= sumMean
#         # print('mean weights')
#         # print(weightsMean)
#         # print('mean equal')
#         # print(w_labeledGestures_Mean)
#         means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         w_labeledGestures_Cov /= sumCov
#         # print('cov weights')
#         # print(weightsCov)
#         # print('cov equal')
#         # print(w_labeledGestures_Cov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + weakModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#         weightsMean = unlabeledGestures['postProb'].values + unlabeledGestures['wMean'].values
#
#         wJCov = np.zeros(classes)
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#         weightsCov = unlabeledGestures['postProb'].values + unlabeledGestures['wCov'].values
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsMean[0][cla] * \
#                                               unlabeledGestures['mean'].loc[0]
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsMean[0][cla]
#
#             JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wCov_J'].loc[cla]
#             adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsCov[0][cla] * \
#                                              unlabeledGestures['cov'].loc[0]
#             wJCov[cla] = currentModel['wCov_J'].loc[cla]
#             adaptiveModel.at[cla, 'wCov_J'] = currentModel['wCov_J'].loc[cla] + weightsCov[0][cla]
#
#         sumMean = wJMean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         wJMean /= sumMean
#         # print('mean weights')
#         # print(weightsMean)
#         # print('mean wJ')
#         # print(wJMean)
#         means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         sumCov = wJCov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         wJCov /= sumCov
#         # print('cov weights')
#         # print(weightsCov)
#         # print('cov wJ')
#         # print(wJCov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + JMean[cla, :] * wJMean[cla]
#
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + JCov[cla, :, :] * wJCov[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, unlabeledGestures
#
#
# def model_PostProb_MSDA_2w_multiplica_norm(currentModel, unlabeledGestures, classes, trainFeatures,
#                                            postProb_trainFeatures, weakModel,
#                                            labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory,
#                                            shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     numSamples = 50
#     labeledGesturesFeatures, labeledGesturesLabels = DA_BasedAdaptiveModels.subsetTraining(
#         labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)
#
#     numberUnlabeledGestures = len(unlabeledGestures.index)
#     if numberUnlabeledGestures == 0:
#         currentModel.at[0, '# gestures'] = shotStart
#     elif samplesInMemory >= numberUnlabeledGestures > 0:
#         unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
#                                                                   numberUnlabeledGestures, labeledGesturesFeatures,
#                                                                   labeledGesturesLabels, weights=True, post=True)
#     else:
#         unlabeledGestures = unlabeledGestures.tail(samplesInMemory).reset_index(drop=True)
#         unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
#                                                                   samplesInMemory, labeledGesturesFeatures,
#                                                                   labeledGesturesLabels, weights=True, post=True)
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     wGestureMean = []
#     wGestureCov = []
#
#     # currentModelMatrix, gestureMatrix = discriminantMatrix(currentModel, gestureMean, gestureCov, classes,
#     #                                                        labeledGesturesFeatures, labeledGesturesLabels, type_DA)
#
#     for cla in range(classes):
#         wGestureMean.append(
#             weightPerPersonMean(currentModel, gestureMean, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,
#                                 type_DA))
#
#         wGestureCov.append(
#             weightPerPersonCov(currentModel, gestureCov, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,
#                                type_DA))
#
#         # wGestureMean.append(weightMSDA(currentModel, gestureMean, gestureCov, cla, classes, labeledGesturesFeatures,
#         #                                labeledGesturesLabels, type_DA))
#
#     postProb_trainFeatures2 = post_probabilities_Calculation(trainFeatures, weakModel, classes, type_DA)
#
#     wGestureMean = np.array(wGestureMean)
#     if wGestureMean.sum() != 0:
#         wGestureMean /= wGestureMean.sum()
#     wGestureCov = np.array(wGestureCov)
#     if wGestureCov.sum() != 0:
#         wGestureCov /= wGestureCov.sum()
#
#     # weightTotal = wGestureMean * postProb_trainFeatures
#     # if weightTotal.sum() != 0:
#     #     weightTotal /= weightTotal.sum()
#
#     wGestureMean = list(wGestureMean)
#     wGestureCov = list(wGestureCov)
#     # wGestureCov = wGestureMean.copy()
#
#     print('\nmodel_PostProb_MSDA_2w_multiplica_norm ', type_DA)
#     print('postProb_2', np.around(postProb_trainFeatures2, 2))
#     print('postProb', np.around(np.array(postProb_trainFeatures), 2))
#     print('wGestureMean', np.around(wGestureMean, 2))
#     print('wGestureCov', np.around(wGestureCov, 2))
#     # print('weightTotal', np.around(weightTotal, 2))
#
#     trainFeatures = subsetTraining_One(trainFeatures, numSamples)
#     new_row = {'mean': gestureMean, 'cov': gestureCov, 'postProb': postProb_trainFeatures, 'wMean': wGestureMean,
#                'wCov': wGestureCov, 'features': trainFeatures}
#     # new_row = {'mean': gestureMean, 'cov': gestureCov, 'postProb': postProb_trainFeatures, 'wMean': wGestureMean,
#     #            'wCov': wGestureCov, 'features': trainFeatures, 'weightTotal': weightTotal}
#
#     unlabeledGestures = unlabeledGestures.append(new_row, ignore_index=True)
#
#     if samplesInMemory > currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures_Mean = np.ones(classes) * shotStart
#         w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         weightsMean = unlabeledGestures['postProb'].values * unlabeledGestures['wMean'].values
#         sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         w_labeledGestures_Mean /= sumMean
#         print('mean weights')
#         print(weightsMean)
#         print('mean few')
#         print(w_labeledGestures_Mean)
#
#         means = np.resize(unlabeledGestures['mean'].values, (classes, numberUnlabeledGestures + 1)).T * weightsMean
#
#         weightsCov = unlabeledGestures['postProb'].values * unlabeledGestures['wCov'].values
#         sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         w_labeledGestures_Cov /= sumCov
#
#         print('cov weights')
#         print(weightsCov)
#         print('cov few')
#         print(w_labeledGestures_Cov)
#
#         covs = np.resize(unlabeledGestures['cov'], (classes, numberUnlabeledGestures + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + weakModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#
#
#     elif samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures_Mean = np.ones(classes) * shotStart
#         w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         weightsMean = unlabeledGestures['postProb'].values * unlabeledGestures['wMean'].values
#
#         weightsCov = unlabeledGestures['postProb'].values * unlabeledGestures['wCov'].values
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * unlabeledGestures['mean'].loc[0]
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#
#             adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
#                 cla] * unlabeledGestures['cov'].loc[0]
#             adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures_Cov[cla] + weightsCov[0][cla]
#
#         sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         w_labeledGestures_Mean /= sumMean
#         print('mean weights')
#         print(weightsMean)
#         print('mean equal')
#         print(w_labeledGestures_Mean)
#         means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         w_labeledGestures_Cov /= sumCov
#         print('cov weights')
#         print(weightsCov)
#         print('cov equal')
#         print(w_labeledGestures_Cov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + weakModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#         weightsMean = unlabeledGestures['postProb'].values * unlabeledGestures['wMean'].values
#
#         wJCov = np.zeros(classes)
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#         weightsCov = unlabeledGestures['postProb'].values * unlabeledGestures['wCov'].values
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsMean[0][cla] * \
#                                               unlabeledGestures['mean'].loc[0]
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsMean[0][cla]
#
#             JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wCov_J'].loc[cla]
#             adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsCov[0][cla] * \
#                                              unlabeledGestures['cov'].loc[0]
#             wJCov[cla] = currentModel['wCov_J'].loc[cla]
#             adaptiveModel.at[cla, 'wCov_J'] = currentModel['wCov_J'].loc[cla] + weightsCov[0][cla]
#
#         sumMean = wJMean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         wJMean /= sumMean
#         print('mean weights')
#         print(weightsMean)
#         print('mean wJ')
#         print(wJMean)
#         means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         sumCov = wJCov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         wJCov /= sumCov
#         print('cov weights')
#         print(weightsCov)
#         print('cov wJ')
#         print(wJCov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + JMean[cla, :] * wJMean[cla]
#
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + JCov[cla, :, :] * wJCov[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, unlabeledGestures
#
#
# def model_PostProb_MSDA_2w_sum_Nonorm(currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures,
#                                       weakModel,
#                                       labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory,
#                                       shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     numSamples = 50
#     labeledGesturesFeatures, labeledGesturesLabels = DA_BasedAdaptiveModels.subsetTraining(
#         labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)
#
#     numberUnlabeledGestures = len(unlabeledGestures.index)
#     if numberUnlabeledGestures == 0:
#         currentModel.at[0, '# gestures'] = shotStart
#     elif samplesInMemory >= numberUnlabeledGestures > 0:
#         unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
#                                                                   numberUnlabeledGestures, labeledGesturesFeatures,
#                                                                   labeledGesturesLabels, weights=True, post=True)
#     else:
#         unlabeledGestures = unlabeledGestures.tail(samplesInMemory).reset_index(drop=True)
#         unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
#                                                                   samplesInMemory, labeledGesturesFeatures,
#                                                                   labeledGesturesLabels, weights=True, post=True)
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     wGestureMean = []
#     wGestureCov = []
#
#     # currentModelMatrix, gestureMatrix = discriminantMatrix(currentModel, gestureMean, gestureCov, classes,
#     #                                                        labeledGesturesFeatures, labeledGesturesLabels, type_DA)
#
#     for cla in range(classes):
#         wGestureMean.append(
#             weightPerPersonMean(currentModel, gestureMean, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,
#                                 type_DA))
#
#         wGestureCov.append(
#             weightPerPersonCov(currentModel, gestureCov, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,
#                                type_DA))
#
#         # wGestureMean.append(weightMSDA(currentModel, gestureMean, gestureCov, cla, classes, labeledGesturesFeatures,
#         #                                labeledGesturesLabels, type_DA))
#
#     wGestureMean[:] = [x / 2 for x in wGestureMean]
#     wGestureCov[:] = [x / 2 for x in wGestureCov]
#     postProb_trainFeatures /= 2
#     postProb_trainFeatures2 = post_probabilities_Calculation(trainFeatures, weakModel, classes, type_DA)
#
#     # wGestureMean = np.array(wGestureMean)
#     # if wGestureMean.sum() != 0:
#     #     wGestureMean /= wGestureMean.sum()
#     # wGestureCov = np.array(wGestureCov)
#     # if wGestureCov.sum() != 0:
#     #     wGestureCov /= wGestureCov.sum()
#     #
#     #
#     # # weightTotal = wGestureMean * postProb_trainFeatures
#     # # if weightTotal.sum() != 0:
#     # #     weightTotal /= weightTotal.sum()
#     #
#     # wGestureMean = list(wGestureMean)
#     # wGestureCov = list(wGestureCov)
#     # # wGestureCov = wGestureMean.copy()
#
#     print('\nmodel_PostProb_MSDA_2w_sum_Nonorm ', type_DA)
#     print('postProb_2', np.around(postProb_trainFeatures2, 2))
#     print('postProb', np.around(np.array(postProb_trainFeatures), 2))
#     print('wGestureMean', np.around(wGestureMean, 2))
#     print('wGestureCov', np.around(wGestureCov, 2))
#     # print('weightTotal', np.around(weightTotal, 2))
#
#     trainFeatures = subsetTraining_One(trainFeatures, numSamples)
#     new_row = {'mean': gestureMean, 'cov': gestureCov, 'postProb': postProb_trainFeatures, 'wMean': wGestureMean,
#                'wCov': wGestureCov, 'features': trainFeatures}
#     # new_row = {'mean': gestureMean, 'cov': gestureCov, 'postProb': postProb_trainFeatures, 'wMean': wGestureMean,
#     #            'wCov': wGestureCov, 'features': trainFeatures, 'weightTotal': weightTotal}
#
#     unlabeledGestures = unlabeledGestures.append(new_row, ignore_index=True)
#
#     if samplesInMemory > currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures_Mean = np.ones(classes) * shotStart
#         w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         weightsMean = unlabeledGestures['postProb'].values + unlabeledGestures['wMean'].values
#         sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         w_labeledGestures_Mean /= sumMean
#         print('mean weights')
#         print(weightsMean)
#         print('mean few')
#         print(w_labeledGestures_Mean)
#
#         means = np.resize(unlabeledGestures['mean'].values, (classes, numberUnlabeledGestures + 1)).T * weightsMean
#
#         weightsCov = unlabeledGestures['postProb'].values + unlabeledGestures['wCov'].values
#         sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         w_labeledGestures_Cov /= sumCov
#
#         print('cov weights')
#         print(weightsCov)
#         print('cov few')
#         print(w_labeledGestures_Cov)
#
#         covs = np.resize(unlabeledGestures['cov'], (classes, numberUnlabeledGestures + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + weakModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#
#
#     elif samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures_Mean = np.ones(classes) * shotStart
#         w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         weightsMean = unlabeledGestures['postProb'].values + unlabeledGestures['wMean'].values
#
#         weightsCov = unlabeledGestures['postProb'].values + unlabeledGestures['wCov'].values
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * unlabeledGestures['mean'].loc[0]
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#
#             adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
#                 cla] * unlabeledGestures['cov'].loc[0]
#             adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures_Cov[cla] + weightsCov[0][cla]
#
#         sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         w_labeledGestures_Mean /= sumMean
#         print('mean weights')
#         print(weightsMean)
#         print('mean equal')
#         print(w_labeledGestures_Mean)
#         means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         w_labeledGestures_Cov /= sumCov
#         print('cov weights')
#         print(weightsCov)
#         print('cov equal')
#         print(w_labeledGestures_Cov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + weakModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#         weightsMean = unlabeledGestures['postProb'].values + unlabeledGestures['wMean'].values
#
#         wJCov = np.zeros(classes)
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#         weightsCov = unlabeledGestures['postProb'].values + unlabeledGestures['wCov'].values
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsMean[0][cla] * \
#                                               unlabeledGestures['mean'].loc[0]
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsMean[0][cla]
#
#             JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wCov_J'].loc[cla]
#             adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsCov[0][cla] * \
#                                              unlabeledGestures['cov'].loc[0]
#             wJCov[cla] = currentModel['wCov_J'].loc[cla]
#             adaptiveModel.at[cla, 'wCov_J'] = currentModel['wCov_J'].loc[cla] + weightsCov[0][cla]
#
#         sumMean = wJMean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         wJMean /= sumMean
#         print('mean weights')
#         print(weightsMean)
#         print('mean wJ')
#         print(wJMean)
#         means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         sumCov = wJCov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         wJCov /= sumCov
#         print('cov weights')
#         print(weightsCov)
#         print('cov wJ')
#         print(wJCov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + JMean[cla, :] * wJMean[cla]
#
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + JCov[cla, :, :] * wJCov[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, unlabeledGestures
#
#
# def model_PostProb_MSDA_1w_sum_Nonorm(currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures,
#                                       weakModel, labeledGesturesFeatures, labeledGesturesLabels, type_DA,
#                                       samplesInMemory, shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     numSamples = 50
#     labeledGesturesFeatures, labeledGesturesLabels = DA_BasedAdaptiveModels.subsetTraining(
#         labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)
#
#     numberUnlabeledGestures = len(unlabeledGestures.index)
#     if numberUnlabeledGestures == 0:
#         currentModel.at[0, '# gestures'] = shotStart
#     elif samplesInMemory >= numberUnlabeledGestures > 0:
#         unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
#                                                                   numberUnlabeledGestures, labeledGesturesFeatures,
#                                                                   labeledGesturesLabels, weights=True, post=True)
#     else:
#         unlabeledGestures = unlabeledGestures.tail(samplesInMemory).reset_index(drop=True)
#         unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
#                                                                   samplesInMemory, labeledGesturesFeatures,
#                                                                   labeledGesturesLabels, weights=True, post=True)
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     wGestureMean = []
#     wGestureCov = []
#
#     # currentModelMatrix, gestureMatrix = discriminantMatrix(currentModel, gestureMean, gestureCov, classes,
#     #                                                        labeledGesturesFeatures, labeledGesturesLabels, type_DA)
#
#     for cla in range(classes):
#         # wGestureMean.append(
#         #     weightPerPersonMean(currentModel, gestureMean, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,
#         #                         type_DA))
#         #
#         # wGestureCov.append(
#         #     weightPerPersonCov(currentModel, gestureCov, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,
#         #                        type_DA))
#
#         wGestureMean.append(weightMSDA(currentModel, gestureMean, gestureCov, cla, classes, labeledGesturesFeatures,
#                                        labeledGesturesLabels, type_DA))
#
#     wGestureMean[:] = [x / 2 for x in wGestureMean]
#     postProb_trainFeatures /= 2
#     wGestureCov = wGestureMean.copy()
#     postProb_trainFeatures2 = post_probabilities_Calculation(trainFeatures, weakModel, classes, type_DA)
#
#     # wGestureMean = np.array(wGestureMean)
#     # if wGestureMean.sum() != 0:
#     #     wGestureMean /= wGestureMean.sum()
#     # wGestureCov = np.array(wGestureCov)
#     # if wGestureCov.sum() != 0:
#     #     wGestureCov /= wGestureCov.sum()
#     #
#     #
#     # # weightTotal = wGestureMean * postProb_trainFeatures
#     # # if weightTotal.sum() != 0:
#     # #     weightTotal /= weightTotal.sum()
#     #
#     # wGestureMean = list(wGestureMean)
#     # wGestureCov = list(wGestureCov)
#     # # wGestureCov = wGestureMean.copy()
#
#     print('\nmodel_PostProb_MSDA_1w_sum_Nonorm ', type_DA)
#     print('postProb_2', np.around(postProb_trainFeatures2, 2))
#     print('postProb', np.around(np.array(postProb_trainFeatures), 2))
#     print('wGestureMean', np.around(wGestureMean, 2))
#     print('wGestureCov', np.around(wGestureCov, 2))
#     # print('weightTotal', np.around(weightTotal, 2))
#
#     trainFeatures = subsetTraining_One(trainFeatures, numSamples)
#     new_row = {'mean': gestureMean, 'cov': gestureCov, 'postProb': postProb_trainFeatures, 'wMean': wGestureMean,
#                'wCov': wGestureCov, 'features': trainFeatures}
#     # new_row = {'mean': gestureMean, 'cov': gestureCov, 'postProb': postProb_trainFeatures, 'wMean': wGestureMean,
#     #            'wCov': wGestureCov, 'features': trainFeatures, 'weightTotal': weightTotal}
#
#     unlabeledGestures = unlabeledGestures.append(new_row, ignore_index=True)
#
#     if samplesInMemory > currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures_Mean = np.ones(classes) * shotStart
#         w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         weightsMean = unlabeledGestures['postProb'].values + unlabeledGestures['wMean'].values
#         sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         w_labeledGestures_Mean /= sumMean
#         print('mean weights')
#         print(weightsMean)
#         print('mean few')
#         print(w_labeledGestures_Mean)
#
#         means = np.resize(unlabeledGestures['mean'].values, (classes, numberUnlabeledGestures + 1)).T * weightsMean
#
#         weightsCov = unlabeledGestures['postProb'].values + unlabeledGestures['wCov'].values
#         sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         w_labeledGestures_Cov /= sumCov
#
#         print('cov weights')
#         print(weightsCov)
#         print('cov few')
#         print(w_labeledGestures_Cov)
#
#         covs = np.resize(unlabeledGestures['cov'], (classes, numberUnlabeledGestures + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + weakModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#
#
#     elif samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures_Mean = np.ones(classes) * shotStart
#         w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         weightsMean = unlabeledGestures['postProb'].values + unlabeledGestures['wMean'].values
#
#         weightsCov = unlabeledGestures['postProb'].values + unlabeledGestures['wCov'].values
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * unlabeledGestures['mean'].loc[0]
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#
#             adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
#                 cla] * unlabeledGestures['cov'].loc[0]
#             adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures_Cov[cla] + weightsCov[0][cla]
#
#         sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         w_labeledGestures_Mean /= sumMean
#         print('mean weights')
#         print(weightsMean)
#         print('mean equal')
#         print(w_labeledGestures_Mean)
#         means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         w_labeledGestures_Cov /= sumCov
#         print('cov weights')
#         print(weightsCov)
#         print('cov equal')
#         print(w_labeledGestures_Cov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + weakModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#         weightsMean = unlabeledGestures['postProb'].values + unlabeledGestures['wMean'].values
#
#         wJCov = np.zeros(classes)
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#         weightsCov = unlabeledGestures['postProb'].values + unlabeledGestures['wCov'].values
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsMean[0][cla] * \
#                                               unlabeledGestures['mean'].loc[0]
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsMean[0][cla]
#
#             JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wCov_J'].loc[cla]
#             adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsCov[0][cla] * \
#                                              unlabeledGestures['cov'].loc[0]
#             wJCov[cla] = currentModel['wCov_J'].loc[cla]
#             adaptiveModel.at[cla, 'wCov_J'] = currentModel['wCov_J'].loc[cla] + weightsCov[0][cla]
#
#         sumMean = wJMean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         wJMean /= sumMean
#         print('mean weights')
#         print(weightsMean)
#         print('mean wJ')
#         print(wJMean)
#         means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         sumCov = wJCov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         wJCov /= sumCov
#         print('cov weights')
#         print(weightsCov)
#         print('cov wJ')
#         print(wJCov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + JMean[cla, :] * wJMean[cla]
#
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + JCov[cla, :, :] * wJCov[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, unlabeledGestures
#
#
# def model_PostProb_MSDA_1w_old(currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures,
#                                weakModel,
#                                labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     numSamples = 50
#     labeledGesturesFeatures, labeledGesturesLabels = DA_BasedAdaptiveModels.subsetTraining(
#         labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)
#
#     numberUnlabeledGestures = len(unlabeledGestures.index)
#     if numberUnlabeledGestures == 0:
#         currentModel.at[0, '# gestures'] = shotStart
#     elif samplesInMemory >= numberUnlabeledGestures > 0:
#         unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
#                                                                   numberUnlabeledGestures, labeledGesturesFeatures,
#                                                                   labeledGesturesLabels, weights=True, post=True)
#     else:
#         unlabeledGestures = unlabeledGestures.tail(samplesInMemory).reset_index(drop=True)
#         unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
#                                                                   samplesInMemory, labeledGesturesFeatures,
#                                                                   labeledGesturesLabels, weights=True, post=True)
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     wGestureMean = []
#     # wGestureCov = []
#
#     for cla in range(classes):
#         # wGestureMean.append(
#         #     weightPerPersonMean(currentModel, gestureMean, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,
#         #                         type_DA))
#         #
#         # wGestureCov.append(
#         #     weightPerPersonCov(currentModel, gestureCov, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,
#         #                        type_DA))
#
#         wGestureMean.append(weightMSDA(currentModel, gestureMean, gestureCov, cla, classes, labeledGesturesFeatures,
#                                        labeledGesturesLabels, type_DA))
#
#     wGestureMean = np.array(wGestureMean)
#     if wGestureMean.sum() != 0:
#         wGestureMean = wGestureMean / (2 * wGestureMean.sum())
#     wGestureMean = list(wGestureMean)
#     wGestureCov = wGestureMean.copy()
#     postProb_trainFeatures /= 2
#
#     trainFeatures = subsetTraining_One(trainFeatures, numSamples)
#     new_row = {'mean': gestureMean, 'cov': gestureCov, 'postProb': postProb_trainFeatures, 'wMean': wGestureMean,
#                'wCov': wGestureCov, 'features': trainFeatures}
#
#     unlabeledGestures = unlabeledGestures.append(new_row, ignore_index=True)
#
#     print('after RQ1: MSDA....post and wgesture')
#     print(np.around(np.array(postProb_trainFeatures), 2))
#     print(np.around(np.array(wGestureMean), 2))
#
#     if samplesInMemory > currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures_Mean = np.ones(classes) * shotStart
#         # w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         weightsMean = unlabeledGestures['postProb'].values + unlabeledGestures['wMean'].values
#         sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         w_labeledGestures_Mean /= sumMean
#         print('mean weights')
#         print(weightsMean)
#         print('mean few')
#         print(w_labeledGestures_Mean)
#
#         means = np.resize(unlabeledGestures['mean'].values, (classes, numberUnlabeledGestures + 1)).T * weightsMean
#
#         # weightsCov = unlabeledGestures['postProb'].values + unlabeledGestures['wCov'].values
#         # sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
#         # weightsCov = list(weightsCov) / sumCov
#         # w_labeledGestures_Cov /= sumCov
#
#         covs = np.resize(unlabeledGestures['cov'], (classes, numberUnlabeledGestures + 1)).T * weightsMean
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + weakModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + weakModel['cov'].loc[cla] * w_labeledGestures_Mean[cla]
#
#
#
#     elif samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures_Mean = np.ones(classes) * shotStart
#         # w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         weightsMean = unlabeledGestures['postProb'].values + unlabeledGestures['wMean'].values
#
#         # weightsCov = unlabeledGestures['postProb'].values + unlabeledGestures['wCov'].values
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * unlabeledGestures['mean'].loc[0]
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#
#             adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * unlabeledGestures['cov'].loc[0]
#             adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#
#         sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         w_labeledGestures_Mean /= sumMean
#         print('mean weights')
#         print(weightsMean)
#         print('mean equal')
#         print(w_labeledGestures_Mean)
#         means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         # sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
#         # weightsCov = list(weightsCov) / sumCov
#         # w_labeledGestures_Cov /= sumCov
#
#         covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + weakModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + weakModel['cov'].loc[cla] * w_labeledGestures_Mean[cla]
#
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#         weightsMean = unlabeledGestures['postProb'].values + unlabeledGestures['wMean'].values
#
#         wJCov = np.zeros(classes)
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#         # weightsCov = unlabeledGestures['postProb'].values + unlabeledGestures['wCov'].values
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsMean[0][cla] * \
#                                               unlabeledGestures['mean'].loc[0]
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsMean[0][cla]
#
#             JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wCov_J'].loc[cla]
#             adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsMean[0][cla] * \
#                                              unlabeledGestures['cov'].loc[0]
#             wJCov[cla] = currentModel['wCov_J'].loc[cla]
#             adaptiveModel.at[cla, 'wCov_J'] = currentModel['wCov_J'].loc[cla] + weightsMean[0][cla]
#
#         sumMean = wJMean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         wJMean /= sumMean
#         print('mean weights')
#         print(weightsMean)
#         print('mean wJ')
#         print(wJMean)
#         means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         # sumCov = wJCov + np.sum(weightsCov, axis=0)
#         # weightsCov = list(weightsCov) / sumCov
#         # wJCov /= sumCov
#
#         covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + JMean[cla, :] * wJMean[cla]
#
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + JCov[cla, :, :] * wJCov[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, unlabeledGestures
#
#
# def model_MSDA(currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     numSamples = 50
#     labeledGesturesFeatures, labeledGesturesLabels = DA_BasedAdaptiveModels.subsetTraining(
#         labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)
#
#     numberUnlabeledGestures = len(unlabeledGestures.index)
#     if numberUnlabeledGestures == 0:
#         currentModel.at[0, '# gestures'] = shotStart
#     elif samplesInMemory >= numberUnlabeledGestures > 0:
#         unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
#                                                                   numberUnlabeledGestures, labeledGesturesFeatures,
#                                                                   labeledGesturesLabels, weights=True, post=False)
#     else:
#         unlabeledGestures = unlabeledGestures.tail(samplesInMemory).reset_index(drop=True)
#         unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
#                                                                   samplesInMemory, labeledGesturesFeatures,
#                                                                   labeledGesturesLabels, weights=True, post=False)
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     wGestureMean = []
#     # wGestureCov = []
#
#     for cla in range(classes):
#         # wGestureMean.append(
#         #     weightPerPersonMean(currentModel, gestureMean, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,
#         #                         type_DA))
#         #
#         # wGestureCov.append(
#         #     weightPerPersonCov(currentModel, gestureCov, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,
#         #                        type_DA))
#
#         wGestureMean.append(weightMSDA(currentModel, gestureMean, gestureCov, cla, classes, labeledGesturesFeatures,
#                                        labeledGesturesLabels, type_DA))
#
#     wGestureMean = np.array(wGestureMean)
#     if wGestureMean.sum() != 0:
#         wGestureMean = wGestureMean / (2 * wGestureMean.sum())
#     wGestureMean = list(wGestureMean)
#     wGestureCov = wGestureMean.copy()
#     postProb_trainFeatures /= 2
#
#     trainFeatures = subsetTraining_One(trainFeatures, 50)
#     new_row = {'mean': gestureMean, 'cov': gestureCov, 'wMean': wGestureMean, 'wCov': wGestureCov,
#                'features': trainFeatures}
#
#     unlabeledGestures = unlabeledGestures.append(new_row, ignore_index=True)
#
#     # print('after RQ1')
#     # print(unlabeledGestures[['postProb', 'wMean', 'wCov']])
#
#     if samplesInMemory > currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures_Mean = np.ones(classes) * shotStart
#         w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         weightsMean = unlabeledGestures['wMean'].values
#         sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         w_labeledGestures_Mean /= sumMean
#         # print('mean weights')
#         # print(weightsMean)
#         # print('mean few')
#         # print(w_labeledGestures_Mean)
#         means = np.resize(unlabeledGestures['mean'].values, (classes, numberUnlabeledGestures + 1)).T * weightsMean
#
#         # if type_DA == 'QDA':
#         weightsCov = unlabeledGestures['wCov'].values
#         sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         w_labeledGestures_Cov /= sumCov
#         # weightsCov = np.nan_to_num(weightsCov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, numberUnlabeledGestures + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + weakModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#
#
#     elif samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures_Mean = np.ones(classes) * shotStart
#         w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         weightsMean = unlabeledGestures['wMean'].values
#         # if type_DA == 'QDA':
#         weightsCov = unlabeledGestures['wCov'].values
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * \
#                                               unlabeledGestures['mean'].loc[0]
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#             # if type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
#                 cla] * \
#                                              unlabeledGestures['cov'].loc[0]
#             adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures_Cov[cla] + weightsCov[0][cla]
#
#         sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         w_labeledGestures_Mean /= sumMean
#         # print('mean weights')
#         # print(weightsMean)
#         # print('mean equal')
#         # print(w_labeledGestures_Mean)
#         means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         # if type_DA == 'QDA':
#         sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         w_labeledGestures_Cov /= sumCov
#         # weightsCov = np.nan_to_num(weightsCov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + weakModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             #
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#
#         weightsMean = unlabeledGestures['wMean'].values
#         # if type_DA == 'QDA':
#         wJCov = np.zeros(classes)
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#         weightsCov = unlabeledGestures['wCov'].values
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsMean[0][cla] * \
#                                               unlabeledGestures['mean'].loc[0]
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsMean[0][cla]
#             # if type_DA == 'QDA':
#             JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wCov_J'].loc[cla]
#             adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsCov[0][cla] * \
#                                              unlabeledGestures['cov'].loc[0]
#             wJCov[cla] = currentModel['wCov_J'].loc[cla]
#             adaptiveModel.at[cla, 'wCov_J'] = currentModel['wCov_J'].loc[cla] + weightsCov[0][cla]
#
#         sumMean = wJMean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         wJMean /= sumMean
#         # print('mean weights')
#         # print(weightsMean)
#         # print('mean wJ')
#         # print(wJMean)
#         means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         # if type_DA == 'QDA':
#         sumCov = wJCov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         wJCov /= sumCov
#         # weightsCov = np.nan_to_num(weightsCov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + JMean[cla, :] * wJMean[cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + JCov[cla, :, :] * wJCov[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, unlabeledGestures
#
#
# def model_PostProb(currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                    labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     numberUnlabeledGestures = len(unlabeledGestures.index)
#     if numberUnlabeledGestures == 0:
#         currentModel.at[0, '# gestures'] = shotStart
#     elif samplesInMemory >= numberUnlabeledGestures > 0:
#         unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
#                                                                   numberUnlabeledGestures, labeledGesturesFeatures,
#                                                                   labeledGesturesLabels, weights=False, post=True)
#     else:
#         unlabeledGestures = unlabeledGestures.tail(samplesInMemory).reset_index(drop=True)
#         unlabeledGestures = postProbabilities_weights_Calculation(unlabeledGestures, currentModel, classes, type_DA,
#                                                                   samplesInMemory, labeledGesturesFeatures,
#                                                                   labeledGesturesLabels, weights=False, post=True)
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     # wGestureMean = []
#     # wGestureCov = []
#
#     # for cla in range(classes):
#     #     wGestureMean.append(
#     #         weightPerPersonMean(currentModel, gestureMean, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,type_DA))
#     #     # if type_DA == 'QDA':
#     #     wGestureCov.append(
#     #         weightPerPersonCov(currentModel, gestureCov, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,type_DA))
#
#     trainFeatures = subsetTraining_One(trainFeatures, 50)
#     new_row = {'mean': gestureMean, 'cov': gestureCov, 'postProb': postProb_trainFeatures, 'features': trainFeatures}
#
#     unlabeledGestures = unlabeledGestures.append(new_row, ignore_index=True)
#
#     # print('after RQ1')
#     # print(unlabeledGestures[['postProb', 'wMean', 'wCov']])
#
#     if samplesInMemory > currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures_Mean = np.ones(classes) * shotStart
#         w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         weightsMean = unlabeledGestures['postProb'].values
#         sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         w_labeledGestures_Mean /= sumMean
#         # print('mean weights')
#         # print(weightsMean)
#         # print('mean few')
#         # print(w_labeledGestures_Mean)
#         means = np.resize(unlabeledGestures['mean'].values, (classes, numberUnlabeledGestures + 1)).T * weightsMean
#
#         # if type_DA == 'QDA':
#         weightsCov = unlabeledGestures['postProb'].values
#         sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         w_labeledGestures_Cov /= sumCov
#         # weightsCov = np.nan_to_num(weightsCov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, numberUnlabeledGestures + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + weakModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#
#
#     elif samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures_Mean = np.ones(classes) * shotStart
#         w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         weightsMean = unlabeledGestures['postProb'].values
#         # if type_DA == 'QDA':
#         weightsCov = unlabeledGestures['postProb'].values
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * \
#                                               unlabeledGestures['mean'].loc[0]
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#             # if type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
#                 cla] * \
#                                              unlabeledGestures['cov'].loc[0]
#             adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures_Cov[cla] + weightsCov[0][cla]
#
#         sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         w_labeledGestures_Mean /= sumMean
#         # print('mean weights')
#         # print(weightsMean)
#         # print('mean equal')
#         # print(w_labeledGestures_Mean)
#         means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         # if type_DA == 'QDA':
#         sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         w_labeledGestures_Cov /= sumCov
#         # weightsCov = np.nan_to_num(weightsCov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + weakModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             #
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#
#         weightsMean = unlabeledGestures['postProb'].values
#         # if type_DA == 'QDA':
#         wJCov = np.zeros(classes)
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#         weightsCov = unlabeledGestures['postProb'].values
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsMean[0][cla] * \
#                                               unlabeledGestures['mean'].loc[0]
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsMean[0][cla]
#             # if type_DA == 'QDA':
#             JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wCov_J'].loc[cla]
#             adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsCov[0][cla] * \
#                                              unlabeledGestures['cov'].loc[0]
#             wJCov[cla] = currentModel['wCov_J'].loc[cla]
#             adaptiveModel.at[cla, 'wCov_J'] = currentModel['wCov_J'].loc[cla] + weightsCov[0][cla]
#
#         sumMean = wJMean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         wJMean /= sumMean
#         # print('mean weights')
#         # print(weightsMean)
#         # print('mean wJ')
#         # print(wJMean)
#         means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         # if type_DA == 'QDA':
#         sumCov = wJCov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         wJCov /= sumCov
#         # weightsCov = np.nan_to_num(weightsCov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + JMean[cla, :] * wJMean[cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + JCov[cla, :, :] * wJCov[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, unlabeledGestures
#
#
# def model_Baseline(currentModel, unlabeledGestures, classes, trainFeatures, trainLabels, postProb_trainFeatures,
#                    weakModel, labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     numberUnlabeledGestures = len(unlabeledGestures.index)
#
#     postProb_trainFeatures = np.zeros(classes)
#     postProb_trainFeatures[int(np.mean(trainLabels)) - 1] = 1
#     if numberUnlabeledGestures == 0:
#         currentModel.at[0, '# gestures'] = shotStart
#     else:
#         unlabeledGestures = unlabeledGestures.tail(samplesInMemory).reset_index(drop=True)
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     # wGestureMean = []
#     # wGestureCov = []
#
#     # for cla in range(classes):
#     #     wGestureMean.append(
#     #         weightPerPersonMean(currentModel, gestureMean, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,type_DA))
#     #     # if type_DA == 'QDA':
#     #     wGestureCov.append(
#     #         weightPerPersonCov(currentModel, gestureCov, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,type_DA))
#
#     trainFeatures = subsetTraining_One(trainFeatures, 50)
#     new_row = {'mean': gestureMean, 'cov': gestureCov, 'postProb': postProb_trainFeatures, 'features': trainFeatures}
#
#     unlabeledGestures = unlabeledGestures.append(new_row, ignore_index=True)
#
#     # print('after RQ1')
#     # print(unlabeledGestures[['postProb', 'wMean', 'wCov']])
#
#     if samplesInMemory > currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures_Mean = np.ones(classes) * shotStart
#         w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         weightsMean = unlabeledGestures['postProb'].values
#         sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         w_labeledGestures_Mean /= sumMean
#         # print('mean weights')
#         # print(weightsMean)
#         # print('mean few')
#         # print(w_labeledGestures_Mean)
#         means = np.resize(unlabeledGestures['mean'].values, (classes, numberUnlabeledGestures + 1)).T * weightsMean
#
#         # if type_DA == 'QDA':
#         weightsCov = unlabeledGestures['postProb'].values
#         sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         w_labeledGestures_Cov /= sumCov
#         # weightsCov = np.nan_to_num(weightsCov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, numberUnlabeledGestures + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + weakModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#
#
#     elif samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures_Mean = np.ones(classes) * shotStart
#         w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         weightsMean = unlabeledGestures['postProb'].values
#         # if type_DA == 'QDA':
#         weightsCov = unlabeledGestures['postProb'].values
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * \
#                                               unlabeledGestures['mean'].loc[0]
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#             # if type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
#                 cla] * \
#                                              unlabeledGestures['cov'].loc[0]
#             adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures_Cov[cla] + weightsCov[0][cla]
#
#         sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         w_labeledGestures_Mean /= sumMean
#         # print('mean weights')
#         # print(weightsMean)
#         # print('mean equal')
#         # print(w_labeledGestures_Mean)
#         means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         # if type_DA == 'QDA':
#         sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         w_labeledGestures_Cov /= sumCov
#         # weightsCov = np.nan_to_num(weightsCov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + weakModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             #
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#
#         weightsMean = unlabeledGestures['postProb'].values
#         # if type_DA == 'QDA':
#         wJCov = np.zeros(classes)
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#         weightsCov = unlabeledGestures['postProb'].values
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsMean[0][cla] * \
#                                               unlabeledGestures['mean'].loc[0]
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsMean[0][cla]
#             # if type_DA == 'QDA':
#             JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wCov_J'].loc[cla]
#             adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsCov[0][cla] * \
#                                              unlabeledGestures['cov'].loc[0]
#             wJCov[cla] = currentModel['wCov_J'].loc[cla]
#             adaptiveModel.at[cla, 'wCov_J'] = currentModel['wCov_J'].loc[cla] + weightsCov[0][cla]
#
#         sumMean = wJMean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         wJMean /= sumMean
#         # print('mean weights')
#         # print(weightsMean)
#         # print('mean wJ')
#         # print(wJMean)
#         means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         # if type_DA == 'QDA':
#         sumCov = wJCov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         wJCov /= sumCov
#         # weightsCov = np.nan_to_num(weightsCov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + JMean[cla, :] * wJMean[cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + JCov[cla, :, :] * wJCov[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, unlabeledGestures
#
#
# def model_MSDA_JS(currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, weakModel,
#                   labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
#     t = time.time()
#     adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])
#
#     numberUnlabeledGestures = len(unlabeledGestures.index)
#     if numberUnlabeledGestures == 0:
#         currentModel.at[0, '# gestures'] = shotStart
#     elif samplesInMemory >= numberUnlabeledGestures > 0:
#         unlabeledGestures = postProbabilities_weights_Calculation_JS(unlabeledGestures, currentModel, classes, type_DA,
#                                                                      numberUnlabeledGestures, labeledGesturesFeatures,
#                                                                      labeledGesturesLabels, weights=True, post=False)
#     else:
#         unlabeledGestures = unlabeledGestures.tail(samplesInMemory).reset_index(drop=True)
#         unlabeledGestures = postProbabilities_weights_Calculation_JS(unlabeledGestures, currentModel, classes, type_DA,
#                                                                      samplesInMemory, labeledGesturesFeatures,
#                                                                      labeledGesturesLabels, weights=True, post=False)
#
#     gestureMean = np.mean(trainFeatures, axis=0)
#     gestureCov = np.cov(trainFeatures, rowvar=False)
#
#     wGestureMean, wGestureCov = weight_MSDA_JS(currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures,
#                                                labeledGesturesLabels, type_DA)
#
#     trainFeatures = subsetTraining_One(trainFeatures, 50)
#     new_row = {'mean': gestureMean, 'cov': gestureCov, 'wMean': wGestureMean, 'wCov': wGestureCov,
#                'features': trainFeatures}
#
#     unlabeledGestures = unlabeledGestures.append(new_row, ignore_index=True)
#
#     # print('after RQ1')
#     # print(unlabeledGestures[['postProb', 'wMean', 'wCov']])
#
#     if samplesInMemory > currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures_Mean = np.ones(classes) * shotStart
#         w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         weightsMean = unlabeledGestures['wMean'].values
#         sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         w_labeledGestures_Mean /= sumMean
#         # print('mean weights')
#         # print(weightsMean)
#         # print('mean few')
#         # print(w_labeledGestures_Mean)
#         means = np.resize(unlabeledGestures['mean'].values, (classes, numberUnlabeledGestures + 1)).T * weightsMean
#
#         # if type_DA == 'QDA':
#         weightsCov = unlabeledGestures['wCov'].values
#         sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         w_labeledGestures_Cov /= sumCov
#         # weightsCov = np.nan_to_num(weightsCov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, numberUnlabeledGestures + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + weakModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#
#
#     elif samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
#         w_labeledGestures_Mean = np.ones(classes) * shotStart
#         w_labeledGestures_Cov = np.ones(classes) * shotStart
#
#         weightsMean = unlabeledGestures['wMean'].values
#         # if type_DA == 'QDA':
#         weightsCov = unlabeledGestures['wCov'].values
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'mean_J'] = weakModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * \
#                                               unlabeledGestures['mean'].loc[0]
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#             # if type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov_J'] = weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
#                 cla] * \
#                                              unlabeledGestures['cov'].loc[0]
#             adaptiveModel.at[cla, 'wCov_J'] = w_labeledGestures_Cov[cla] + weightsCov[0][cla]
#
#         sumMean = w_labeledGestures_Mean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         w_labeledGestures_Mean /= sumMean
#         # print('mean weights')
#         # print(weightsMean)
#         # print('mean equal')
#         # print(w_labeledGestures_Mean)
#         means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         # if type_DA == 'QDA':
#         sumCov = w_labeledGestures_Cov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         w_labeledGestures_Cov /= sumCov
#         # weightsCov = np.nan_to_num(weightsCov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + weakModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             #
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + weakModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
#
#
#     elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
#         wJMean = np.zeros(classes)
#         JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
#
#         weightsMean = unlabeledGestures['wMean'].values
#         # if type_DA == 'QDA':
#         wJCov = np.zeros(classes)
#         JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))
#         weightsCov = unlabeledGestures['wCov'].values
#
#         for cla in range(classes):
#             JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsMean[0][cla] * \
#                                               unlabeledGestures['mean'].loc[0]
#             wJMean[cla] = currentModel['wMean_J'].loc[cla]
#             adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsMean[0][cla]
#             # if type_DA == 'QDA':
#             JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wCov_J'].loc[cla]
#             adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsCov[0][cla] * \
#                                              unlabeledGestures['cov'].loc[0]
#             wJCov[cla] = currentModel['wCov_J'].loc[cla]
#             adaptiveModel.at[cla, 'wCov_J'] = currentModel['wCov_J'].loc[cla] + weightsCov[0][cla]
#
#         sumMean = wJMean + np.sum(weightsMean, axis=0)
#         weightsMean = list(weightsMean) / sumMean
#         wJMean /= sumMean
#         # print('mean weights')
#         # print(weightsMean)
#         # print('mean wJ')
#         # print(wJMean)
#         means = np.resize(unlabeledGestures['mean'], (classes, samplesInMemory + 1)).T * weightsMean
#
#         # if type_DA == 'QDA':
#         sumCov = wJCov + np.sum(weightsCov, axis=0)
#         weightsCov = list(weightsCov) / sumCov
#         wJCov /= sumCov
#         # weightsCov = np.nan_to_num(weightsCov)
#         covs = np.resize(unlabeledGestures['cov'], (classes, samplesInMemory + 1)).T * weightsCov
#
#         for cla in range(classes):
#             adaptiveModel.at[cla, 'class'] = cla + 1
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + JMean[cla, :] * wJMean[cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + JCov[cla, :, :] * wJCov[cla]
#
#     adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
#     return adaptiveModel, time.time() - t, unlabeledGestures
#
#
#
# Post Probabilities
# def postProbabilities_weights_Calculation(unlabeledGestures, model, classes, type_DA, numberGestures,
#                                           labeledGesturesFeatures,
#                                           labeledGesturesLabels, weights=True, post=True):
#     if post:
#         post_probabilities = []
#     if weights:
#         weightsMean = []
#         weightsCov = []
#
#     for i in range(numberGestures):
#         if post:
#             post_probabilities.append(
#                 post_probabilities_Calculation(unlabeledGestures['features'].loc[i], model, classes, type_DA))
#         if weights:
#             wGestureMean = []
#             # wGestureCov = []
#
#             # for cla in range(classes):
#             #     wGestureMean.append(
#             #         weightPerPersonMean(model, unlabeledGestures['mean'].loc[i], cla, classes, labeledGesturesFeatures,
#             #                             labeledGesturesLabels, type_DA))
#             #     wGestureCov.append(
#             #         weightPerPersonCov(model, unlabeledGestures['cov'].loc[i], cla, classes, labeledGesturesFeatures,
#             #                            labeledGesturesLabels, type_DA))
#             #
#             # weightsMean.append(np.array(wGestureMean))
#             # weightsCov.append(np.array(wGestureCov))
#
#             for cla in range(classes):
#                 # wGestureMean.append(
#                 #     weightPerPersonMean(currentModel, gestureMean, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,
#                 #                         type_DA))
#                 #
#                 # wGestureCov.append(
#                 #     weightPerPersonCov(currentModel, gestureCov, cla, classes, labeledGesturesFeatures, labeledGesturesLabels,
#                 #                        type_DA))
#
#                 wGestureMean.append(
#                     weightMSDA(model, unlabeledGestures['mean'].loc[i], unlabeledGestures['cov'].loc[i], cla, classes,
#                                labeledGesturesFeatures, labeledGesturesLabels, type_DA))
#
#             wGestureMean = np.array(wGestureMean)
#             if wGestureMean.sum() != 0:
#                 wGestureMean = wGestureMean / (2 * wGestureMean.sum())
#             wGestureMean = list(wGestureMean)
#             wGestureCov = wGestureMean.copy()
#             weightsMean.append(np.array(wGestureMean))
#             weightsCov.append(np.array(wGestureCov))
#
#     if post:
#         unlabeledGestures['postProb'] = post_probabilities
#     if weights:
#         unlabeledGestures['wMean'] = weightsMean
#         unlabeledGestures['wCov'] = weightsCov
#     return unlabeledGestures
#

#
#

#
#
# # def weightPerPersonMean_KL(currentValues, personMean, currentClass, classes, trainFeatures, trainLabels, type_DA):
# #     personValues = currentValues.copy()
# #     personValues['mean'].at[currentClass] = personMean
# #     return klDivergenceModel(trainFeatures, trainLabels, personValues, classes, currentClass)
# #
# #
# # def weightPerPersonCov_KL(currentValues, personCov, currentClass, classes, trainFeatures, trainLabels, type_DA):
# #     personValues = currentValues.copy()
# #     personValues['cov'].at[currentClass] = personCov
# #     return klDivergenceModel(trainFeatures, trainLabels, personValues, classes, currentClass)
#
#
# # def predictedKL(mean, cov, model, classes):
# #     d = np.zeros([classes])
# #     for cla in range(classes):
# #         d[cla] = JSdivergence(model['mean'].loc[cla], mean, np.size(mean), model['cov'].loc[cla], cov)
# #         # d = d - d[np.argmin(d)]
# #     # d / d.sum()
# #     return np.argmin(d)
#
#
# # def klDivergenceModel(testFeatures, testLabels, model, classes, currentClass):
# #     TP = 0
# #     TN = 0
# #     FP = 0
# #     FN = 0
# #     for i in range(classes):
# #         meanAux = np.mean(testFeatures[testLabels == i + 1, :], axis=0)
# #         covAux = np.cov(testFeatures[testLabels == i + 1, :], rowvar=False)
# #         KLresult = predictedKL(meanAux, covAux, model, classes)
# #         if KLresult == i:
# #             if KLresult == currentClass:
# #                 TP += 1
# #             else:
# #                 TN += 1
# #         else:
# #             if i == currentClass:
# #                 FN += 1
# #             else:
# #                 FP += 1
# #     return mcc(TP, TN, FP, FN)
#
#
#
#
#
# %% Weight Calculation
# def weightMSDA(currentValues, personMean, personCov, currentClass, classes, trainFeatures, trainLabels, type_DA):
#     personValues = currentValues.copy()
#     personValues['mean'].at[currentClass] = personMean
#     personValues['cov'].at[currentClass] = personCov
#     if type_DA == 'LDA':
#         weight = mccModelLDA(trainFeatures, trainLabels, personValues, classes, currentClass)
#     elif type_DA == 'QDA':
#         weight = mccModelQDA(trainFeatures, trainLabels, personValues, classes, currentClass)
#     return weight
#
#
# def weightPerPersonMean(currentValues, personMean, currentClass, classes, trainFeatures, trainLabels, type_DA):
#     personValues = currentValues.copy()
#     personValues['mean'].at[currentClass] = personMean
#     if type_DA == 'LDA':
#         weight = mccModelLDA(trainFeatures, trainLabels, personValues, classes, currentClass)
#     elif type_DA == 'QDA':
#         weight = mccModelQDA(trainFeatures, trainLabels, personValues, classes, currentClass)
#     return weight
#
#
# def weightPerPersonCov(currentValues, personCov, currentClass, classes, trainFeatures, trainLabels, type_DA):
#     personValues = currentValues.copy()
#     personValues['cov'].at[currentClass] = personCov
#     if type_DA == 'LDA':
#         weight = mccModelLDA(trainFeatures, trainLabels, personValues, classes, currentClass)
#     elif type_DA == 'QDA':
#         weight = mccModelQDA(trainFeatures, trainLabels, personValues, classes, currentClass)
#     return weight
# def mccModelLDA(testFeatures, testLabels, model, classes, currentClass):
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#     currentClass += 1
#     LDACov = DA_Classifiers.LDA_Cov(model, classes)
#     for i in range(np.size(testLabels)):
#         currentPredictor = DA_Classifiers.predictedModelLDA(testFeatures[i, :], model, classes, LDACov)
#         if currentPredictor == testLabels[i]:
#             if currentPredictor == currentClass:
#                 TP += 1
#             else:
#                 TN += 1
#         else:
#             if testLabels[i] == currentClass:
#                 FN += 1
#             else:
#                 FP += 1
#     return mcc(TP, TN, FP, FN)


# def mccModelQDA(testFeatures, testLabels, model, classes, currentClass):
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#     currentClass += 1
#     for i in range(np.size(testLabels)):
#         currentPredictor = DA_Classifiers.predictedModelQDA(testFeatures[i, :], model, classes)
#
#         if currentPredictor == testLabels[i]:
#             if currentPredictor == currentClass:
#                 TP += 1
#             else:
#                 TN += 1
#         else:
#             if testLabels[i] == currentClass:
#                 FN += 1
#             else:
#                 FP += 1
#
#     return mcc(TP, TN, FP, FN)
#
