# %% Libraries
import time
# import math
import numpy as np
import pandas as pd
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

def model_Labels(currentModel, classes, trainFeatures, postProb_trainFeatures, fewModel, labeledGesturesFeatures,
                 labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
    t = time.time()
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])

    gestureMean = np.mean(trainFeatures, axis=0)
    gestureCov = np.cov(trainFeatures, rowvar=False)

    aux = np.zeros(classes)
    aux[np.argmax(postProb_trainFeatures)] = 1
    postProb_trainFeatures = aux

    if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
        w_labeledGestures = np.ones(classes) * shotStart

        weightsUnlabeledGesture = postProb_trainFeatures

        for cla in range(classes):
            adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures[cla] + \
                                              weightsUnlabeledGesture[cla] * gestureMean
            adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]

            adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures[cla] + \
                                             weightsUnlabeledGesture[cla] * gestureCov

        sumMean = w_labeledGestures + weightsUnlabeledGesture
        weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
        w_labeledGestures /= sumMean

        means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm

        covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla] + fewModel['mean'].loc[cla] * w_labeledGestures[cla]
            adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + fewModel['cov'].loc[cla] * w_labeledGestures[cla]

    elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
        wJMean = np.zeros(classes)
        JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
        weightsUnlabeledGesture = postProb_trainFeatures

        JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))

        for cla in range(classes):
            JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
                                              gestureMean
            wJMean[cla] = currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]

            JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
                                             gestureCov

        sumMean = wJMean + weightsUnlabeledGesture
        weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
        wJMean /= sumMean

        means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm

        covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]

            adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]

    adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
    return adaptiveModel, time.time() - t, weightsUnlabeledGesture


def model_PostProb(currentModel, classes, trainFeatures, postProb_trainFeatures, fewModel, labeledGesturesFeatures,
                   labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
    t = time.time()
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])

    gestureMean = np.mean(trainFeatures, axis=0)
    gestureCov = np.cov(trainFeatures, rowvar=False)

    if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
        w_labeledGestures = np.ones(classes) * shotStart

        weightsUnlabeledGesture = postProb_trainFeatures

        for cla in range(classes):
            adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures[cla] + \
                                              weightsUnlabeledGesture[cla] * gestureMean
            adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]

            adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures[cla] + \
                                             weightsUnlabeledGesture[cla] * gestureCov

        sumMean = w_labeledGestures + weightsUnlabeledGesture
        weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
        w_labeledGestures /= sumMean

        means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm

        covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla] + fewModel['mean'].loc[cla] * w_labeledGestures[cla]
            adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + fewModel['cov'].loc[cla] * w_labeledGestures[cla]

    elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
        wJMean = np.zeros(classes)
        JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
        weightsUnlabeledGesture = postProb_trainFeatures

        JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))

        for cla in range(classes):
            JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
                                              gestureMean
            wJMean[cla] = currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]

            JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
                                             gestureCov

        sumMean = wJMean + weightsUnlabeledGesture
        weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
        wJMean /= sumMean

        means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm

        covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]

            adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]

    adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
    return adaptiveModel, time.time() - t, weightsUnlabeledGesture


def model_PostProb_MSDA(currentModel, classes, trainFeatures, postProb_trainFeatures, fewModel, labeledGesturesFeatures,
                        labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
    t = time.time()
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])

    numSamples = 50
    labeledGesturesFeatures, labeledGesturesLabels = DA_BasedAdaptiveModels.subsetTraining(
        labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)

    gestureMean = np.mean(trainFeatures, axis=0)
    gestureCov = np.cov(trainFeatures, rowvar=False)

    weightsMSDA = weightMSDA_reduce(currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures,
                                    labeledGesturesLabels, type_DA)

    weightsMSDA = np.array(weightsMSDA) / 2
    postProb_trainFeatures /= 2

    print(type_DA)
    print(weightsMSDA)
    print(postProb_trainFeatures)

    if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
        w_labeledGestures = np.ones(classes) * shotStart

        weightsUnlabeledGesture = postProb_trainFeatures + weightsMSDA

        for cla in range(classes):
            adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures[cla] + \
                                              weightsUnlabeledGesture[cla] * gestureMean
            adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]

            adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures[cla] + \
                                             weightsUnlabeledGesture[cla] * gestureCov

        sumMean = w_labeledGestures + weightsUnlabeledGesture
        weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
        w_labeledGestures /= sumMean

        means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm

        covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla] + fewModel['mean'].loc[cla] * w_labeledGestures[cla]
            adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + fewModel['cov'].loc[cla] * w_labeledGestures[cla]

    elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
        wJMean = np.zeros(classes)
        JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
        weightsUnlabeledGesture = postProb_trainFeatures + weightsMSDA

        JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))

        for cla in range(classes):
            JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
                                              gestureMean
            wJMean[cla] = currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]

            JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
                                             gestureCov

        sumMean = wJMean + weightsUnlabeledGesture
        weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
        wJMean /= sumMean

        means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm

        covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]

            adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]

    adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
    return adaptiveModel, time.time() - t, weightsUnlabeledGesture


def model_MSDAlabels(currentModel, classes, trainFeatures, postProb_trainFeatures, fewModel, labeledGesturesFeatures,
                     labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
    t = time.time()
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])

    numSamples = 50
    labeledGesturesFeatures, labeledGesturesLabels = DA_BasedAdaptiveModels.subsetTraining(
        labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)

    gestureMean = np.mean(trainFeatures, axis=0)
    gestureCov = np.cov(trainFeatures, rowvar=False)

    weightsMSDA = weightMSDA_reduce(currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures,
                                    labeledGesturesLabels, type_DA)

    weightsMSDA = np.array(weightsMSDA)

    # print(type_DA)
    # print(weightsMSDA)
    # print(postProb_trainFeatures)


    # if weightsMSDA.sum() != 0:
    #     weightsMSDA /= weightsMSDA.sum()
    # print('norm')
    # print(weightsMSDA)

    # postProb_trainFeatures[weightsMSDA == 0] = 0
    # weightsMSDA[postProb_trainFeatures == 0] = 0
    # weightsMSDA += postProb_trainFeatures

    postProb_trainFeatures /= 2
    aux = np.zeros(classes)
    if weightsMSDA.sum() != 0:
        weightsMSDA /= 2
        if np.argmax(weightsMSDA) == np.argmax(postProb_trainFeatures):
            aux[np.argmax(weightsMSDA)] = weightsMSDA[np.argmax(weightsMSDA)] + postProb_trainFeatures[
                np.argmax(postProb_trainFeatures)]
        else:
            aux[np.argmax(weightsMSDA)] = weightsMSDA[np.argmax(weightsMSDA)] + \
                                          postProb_trainFeatures[np.argmax(weightsMSDA)]
            aux[np.argmax(postProb_trainFeatures)] = weightsMSDA[np.argmax(postProb_trainFeatures)] + \
                                                     postProb_trainFeatures[np.argmax(postProb_trainFeatures)]
    else:
        aux[np.argmax(postProb_trainFeatures)] = postProb_trainFeatures[np.argmax(postProb_trainFeatures)]

    weightsMSDA = aux

    # print('Final weight')
    # print(weightsMSDA)

    if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
        w_labeledGestures = np.ones(classes) * shotStart

        weightsUnlabeledGesture = weightsMSDA

        for cla in range(classes):
            adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures[cla] + \
                                              weightsUnlabeledGesture[cla] * gestureMean
            adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]

            adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures[cla] + \
                                             weightsUnlabeledGesture[cla] * gestureCov

        sumMean = w_labeledGestures + weightsUnlabeledGesture
        weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
        w_labeledGestures /= sumMean

        means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm

        covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla] + fewModel['mean'].loc[cla] * w_labeledGestures[cla]
            adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + fewModel['cov'].loc[cla] * w_labeledGestures[cla]

    elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
        wJMean = np.zeros(classes)
        JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
        weightsUnlabeledGesture = weightsMSDA

        JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))

        for cla in range(classes):
            JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
                                              gestureMean
            wJMean[cla] = currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]

            JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
                                             gestureCov

        sumMean = wJMean + weightsUnlabeledGesture
        weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
        wJMean /= sumMean

        means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm

        covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]

            adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]

    adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
    return adaptiveModel, time.time() - t, weightsUnlabeledGesture


def model_PostProb_MSDA_multiplication(currentModel, classes, trainFeatures, postProb_trainFeatures, fewModel,
                                       labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory,
                                       shotStart):
    t = time.time()
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])

    numSamples = 50
    labeledGesturesFeatures, labeledGesturesLabels = DA_BasedAdaptiveModels.subsetTraining(
        labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)

    gestureMean = np.mean(trainFeatures, axis=0)
    gestureCov = np.cov(trainFeatures, rowvar=False)

    weightsMSDA = weightMSDA_reduce(currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures,
                                    labeledGesturesLabels, type_DA)

    weightsMSDA = np.array(weightsMSDA)

    if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
        w_labeledGestures = np.ones(classes) * shotStart

        weightsUnlabeledGesture = postProb_trainFeatures * weightsMSDA

        for cla in range(classes):
            adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures[cla] + \
                                              weightsUnlabeledGesture[cla] * gestureMean
            adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]

            adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures[cla] + \
                                             weightsUnlabeledGesture[cla] * gestureCov

        sumMean = w_labeledGestures + weightsUnlabeledGesture
        weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
        w_labeledGestures /= sumMean

        means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm

        covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla] + fewModel['mean'].loc[cla] * w_labeledGestures[cla]
            adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + fewModel['cov'].loc[cla] * w_labeledGestures[cla]

    elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
        wJMean = np.zeros(classes)
        JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
        weightsUnlabeledGesture = postProb_trainFeatures * weightsMSDA

        JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))

        for cla in range(classes):
            JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
                                              gestureMean
            wJMean[cla] = currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]

            JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
                                             gestureCov

        sumMean = wJMean + weightsUnlabeledGesture
        weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
        wJMean /= sumMean

        means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm

        covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]

            adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]

    adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
    return adaptiveModel, time.time() - t, weightsUnlabeledGesture


def model_PostProb_JS_multiplication(currentModel, classes, trainFeatures, postProb_trainFeatures, fewModel,
                                     labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory,
                                     shotStart):
    t = time.time()
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])

    numSamples = 50
    labeledGesturesFeatures, labeledGesturesLabels = DA_BasedAdaptiveModels.subsetTraining(
        labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)

    gestureMean = np.mean(trainFeatures, axis=0)
    gestureCov = np.cov(trainFeatures, rowvar=False)

    weightsMSDA, _ = weight_MSDA_JS(currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures,
                                    labeledGesturesLabels, type_DA)
    weightsMSDA = np.array(weightsMSDA)

    if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
        w_labeledGestures = np.ones(classes) * shotStart

        weightsUnlabeledGesture = postProb_trainFeatures * weightsMSDA

        for cla in range(classes):
            adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures[cla] + \
                                              weightsUnlabeledGesture[cla] * gestureMean
            adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]

            adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures[cla] + \
                                             weightsUnlabeledGesture[cla] * gestureCov

        sumMean = w_labeledGestures + weightsUnlabeledGesture
        weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
        w_labeledGestures /= sumMean

        means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm

        covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla] + fewModel['mean'].loc[cla] * w_labeledGestures[cla]
            adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + fewModel['cov'].loc[cla] * w_labeledGestures[cla]

    elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
        wJMean = np.zeros(classes)
        JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
        weightsUnlabeledGesture = postProb_trainFeatures * weightsMSDA

        JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))

        for cla in range(classes):
            JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
                                              gestureMean
            wJMean[cla] = currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]

            JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
                                             gestureCov

        sumMean = wJMean + weightsUnlabeledGesture
        weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
        wJMean /= sumMean

        means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm

        covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]

            adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]

    adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
    return adaptiveModel, time.time() - t, weightsUnlabeledGesture


def model_PostProbOld(currentModel, classes, trainFeatures, postProb_trainFeatures, fewModel, labeledGesturesFeatures,
                      labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
    t = time.time()
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])

    gestureMean = np.mean(trainFeatures, axis=0)
    gestureCov = np.cov(trainFeatures, rowvar=False)
    postProb_trainFeatures = post_probabilities_Calculation(trainFeatures, fewModel, classes, type_DA)

    if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
        w_labeledGestures = np.ones(classes) * shotStart

        weightsUnlabeledGesture = postProb_trainFeatures

        for cla in range(classes):
            adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures[cla] + \
                                              weightsUnlabeledGesture[cla] * gestureMean
            adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]

            adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures[cla] + \
                                             weightsUnlabeledGesture[cla] * gestureCov

        sumMean = w_labeledGestures + weightsUnlabeledGesture
        weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
        w_labeledGestures /= sumMean

        means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm

        covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla] + fewModel['mean'].loc[cla] * w_labeledGestures[cla]
            adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + fewModel['cov'].loc[cla] * w_labeledGestures[cla]

    elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
        wJMean = np.zeros(classes)
        JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
        weightsUnlabeledGesture = postProb_trainFeatures

        JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))

        for cla in range(classes):
            JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
                                              gestureMean
            wJMean[cla] = currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]

            JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
                                             gestureCov

        sumMean = wJMean + weightsUnlabeledGesture
        weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
        wJMean /= sumMean

        means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm

        covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]

            adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]

    adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
    return adaptiveModel, time.time() - t, weightsUnlabeledGesture


def model_PostProb_PostProbOld_multiplication(currentModel, classes, trainFeatures, postProb_trainFeatures, fewModel,
                                              labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory,
                                              shotStart):
    t = time.time()
    adaptiveModel = pd.DataFrame(columns=['cov', 'mean', 'class', 'mean_J', 'wMean_J', 'cov_J', 'wCov_J', '# gestures'])

    numSamples = 50
    labeledGesturesFeatures, labeledGesturesLabels = DA_BasedAdaptiveModels.subsetTraining(
        labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)

    gestureMean = np.mean(trainFeatures, axis=0)
    gestureCov = np.cov(trainFeatures, rowvar=False)

    # weightsMSDA = weightMSDA_reduce(currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures,
    #                                 labeledGesturesLabels, type_DA)
    #
    # weightsMSDA = np.array(weightsMSDA)

    weightsMSDA = post_probabilities_Calculation(trainFeatures, fewModel, classes, type_DA)

    if samplesInMemory == currentModel['# gestures'].loc[0] - shotStart:
        w_labeledGestures = np.ones(classes) * shotStart

        weightsUnlabeledGesture = postProb_trainFeatures * weightsMSDA

        for cla in range(classes):
            adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures[cla] + \
                                              weightsUnlabeledGesture[cla] * gestureMean
            adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures[cla] + weightsUnlabeledGesture[cla]

            adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures[cla] + \
                                             weightsUnlabeledGesture[cla] * gestureCov

        sumMean = w_labeledGestures + weightsUnlabeledGesture
        weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
        w_labeledGestures /= sumMean

        means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm

        covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla] + fewModel['mean'].loc[cla] * w_labeledGestures[cla]
            adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + fewModel['cov'].loc[cla] * w_labeledGestures[cla]

    elif samplesInMemory < currentModel['# gestures'].loc[0] - shotStart:
        wJMean = np.zeros(classes)
        JMean = np.zeros((classes, np.size(trainFeatures, axis=1)))
        weightsUnlabeledGesture = postProb_trainFeatures * weightsMSDA

        JCov = np.zeros((classes, np.size(trainFeatures, axis=1), np.size(trainFeatures, axis=1)))

        for cla in range(classes):
            JMean[cla, :] = currentModel['mean_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'mean_J'] = currentModel['mean_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
                                              gestureMean
            wJMean[cla] = currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'wMean_J'] = currentModel['wMean_J'].loc[cla] + weightsUnlabeledGesture[cla]

            JCov[cla, :, :] = currentModel['cov_J'].loc[cla] / currentModel['wMean_J'].loc[cla]
            adaptiveModel.at[cla, 'cov_J'] = currentModel['cov_J'].loc[cla] + weightsUnlabeledGesture[cla] * \
                                             gestureCov

        sumMean = wJMean + weightsUnlabeledGesture
        weightsUnlabeledGesture_Norm = weightsUnlabeledGesture / sumMean
        wJMean /= sumMean

        means = np.resize(gestureMean, (classes, len(gestureMean))).T * weightsUnlabeledGesture_Norm

        covs = np.resize(gestureCov, (classes, len(gestureMean), len(gestureMean))).T * weightsUnlabeledGesture_Norm

        for cla in range(classes):
            adaptiveModel.at[cla, 'class'] = cla + 1
            adaptiveModel.at[cla, 'mean'] = means[:, cla] + JMean[cla, :] * wJMean[cla]

            adaptiveModel.at[cla, 'cov'] = covs[:, :, cla] + JCov[cla, :, :] * wJMean[cla]

    adaptiveModel.at[0, '# gestures'] = currentModel['# gestures'].loc[0] + 1
    return adaptiveModel, time.time() - t, weightsUnlabeledGesture




# def model_PostProb_MSDA_JS(currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewModel,
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
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
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
#             adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * unlabeledGestures['mean'].loc[0]
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#             # if type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
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
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             #
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
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

def postProbabilities_weights_Calculation_JS(unlabeledGestures, model, classes, type_DA, numberGestures,
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
            wGestureMean, wGestureCov = weight_MSDA_JS(model, unlabeledGestures['mean'].loc[i],
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


def weight_MSDA_JS(currentModel, gestureMean, gestureCov, classes, labeledGesturesFeatures, labeledGesturesLabels,
                   type_DA):
    matrix_D = matrixDivergences(labeledGesturesFeatures, labeledGesturesLabels, currentModel, classes, type_DA)

    wGestureMean = []
    wGestureCov = []
    aux = []

    wModelU_L = []
    wU_allL = []

    # if type_DA == 'LDA':
    #     LDACov = DA_Classifiers.LDA_Cov(currentModel, classes)
    #     auxRowLDA = rowDivergences(labeledGesturesFeatures, labeledGesturesLabels, gestureMean, LDACov, classes)
    for cla in range(classes):
        matrix_D_mean = matrix_D.copy()
        matrix_D_cov = matrix_D.copy()
        # if type_DA == 'LDA':
        #     matrix_D_mean[cla, :] = auxRowLDA
        #     auxCurrentModel = currentModel.copy()
        #     auxCurrentModel['cov'].at[cla] = gestureCov
        #     auxLDACov = DA_Classifiers.LDA_Cov(auxCurrentModel, classes)
        #     matrix_D_cov[cla, :] = rowDivergences(labeledGesturesFeatures, labeledGesturesLabels,
        #                                           currentModel['mean'].loc[cla], auxLDACov, classes)
        # elif type_DA == 'QDA':

        matrix_D_mean[cla, :] = rowDivergences(labeledGesturesFeatures, labeledGesturesLabels, gestureMean,
                                               currentModel['cov'].loc[cla], classes)
        matrix_D_cov[cla, :] = rowDivergences(labeledGesturesFeatures, labeledGesturesLabels,
                                              currentModel['mean'].loc[cla], gestureCov, classes)
        # if np.argmin(matrix_D_mean[:, cla]) != cla or np.argmin(matrix_D_mean[cla, :]) != cla:
        #     wGestureMean.append(0)
        # else:
        #     wGestureMean.append(1)
        #
        # if np.argmin(matrix_D_cov[:, cla]) != cla or np.argmin(matrix_D_cov[cla, :]) != cla:
        #     wGestureCov.append(0)
        # else:
        #     wGestureCov.append(1)

        #     matrix_D_mean[cla, :] = rowDivergences(labeledGesturesFeatures, labeledGesturesLabels, gestureMean,
        #                                            gestureCov, classes)
        #
        #     aux.append(JSdivergence(currentModel['mean'].loc[cla], gestureMean, np.size(currentModel['mean'].loc[cla]),
        #                             currentModel['cov'].loc[cla], gestureCov))
        #     value = 0
        #     # if np.argmin(aux) != cla:
        #     #     wGestureMean.append(0)
        #     #     wGestureCov.append(0)
        #     # else:
        #     #     wGestureMean.append(1)
        #     #     wGestureCov.append(1)
        #     #     value+=1
        #
        #     if np.argmin(matrix_D_mean[cla, :]) == cla:
        #         # wGestureMean.append(0)
        #         # wGestureCov.append(0)
        #         # else:
        #         # wGestureMean.append(1)
        #         # wGestureCov.append(1)
        #         value += 1
        #         wU_allL.append(1)
        #     else:
        #         wU_allL.append(0)
        #
        #     if np.argmin(matrix_D_mean[:, cla]) == cla:
        #         # wGestureMean.append(0)
        #         # wGestureCov.append(0)
        #         # else:
        #         # wGestureMean.append(1)
        #         # wGestureCov.append(1)
        #
        #         value += 1
        #         wModelU_L.append(1)
        #     else:
        #         wModelU_L.append(0)
        #     wGestureMean.append(value)
        #
        # aux = np.array(aux)
        # # print('wModel_U')
        # # print(np.argmin(aux))
        # # print('wModelU_L')
        # # print(wModelU_L)
        # # print('wU_allL')
        # # print(wU_allL)

        # wGestureMean = np.array(wGestureMean)
        # wGestureMean[np.argmin(aux)] += 1
        # wGestureMean = list(wGestureMean / wGestureMean.sum())
        # wGestureCov = wGestureMean.copy()
        # return wGestureMean, wGestureCov

        #     if np.argmin(matrix_D_mean[:, cla]) != cla or np.argmin(matrix_D_mean[cla, :]) != cla:
        #         wGestureMean.append(0)
        #     else:
        #         JS_2 = 0
        #         for i in range(classes):
        #             if i != cla:
        #                 JS_2 += matrix_D_mean[cla, i] / (matrix_D_mean[cla, i] + matrix_D_mean[i, cla])
        #         wGestureMean.append(JS_2 / (classes - 1))
        #
        #         # JS_1 = second_smallest(matrix_D_mean[:, cla])
        #         # JS_2 = second_smallest(matrix_D_mean[cla, :])
        #         # weightMean = JS_2 / (JS_1 + JS_2)
        #
        #     if np.argmin(matrix_D_cov[:, cla]) != cla or np.argmin(matrix_D_cov[cla, :]) != cla:
        #         wGestureCov.append(0)
        #     else:
        #         JS_2 = 0
        #         for i in range(classes):
        #             if i != cla:
        #                 JS_2 += matrix_D_cov[cla, i] / (matrix_D_cov[cla, i] + matrix_D_cov[i, cla])
        #         wGestureCov.append(JS_2 / (classes - 1))
        #
        #         # JS_1 = second_smallest(matrix_D_cov[:, cla])
        #         # JS_2 = second_smallest(matrix_D_cov[cla, :])
        #         # weightCov = JS_2 / (JS_1 + JS_2)
        #
        # return wGestureMean, wGestureCov

        wGestureMean.append(mcc_from_matrixDivergences(matrix_D_mean, classes, cla))
        wGestureCov.append(mcc_from_matrixDivergences(matrix_D_cov, classes, cla))
    return wGestureMean, wGestureCov


def matrixDivergences(Features, Labels, model, classes, type_DA):
    # if type_DA == 'LDA':
    #     LDACov = DA_Classifiers.LDA_Cov(model, classes)
    #     matrix = np.zeros((classes, classes))
    #     for i in range(classes):
    #         meanSamples = np.mean(Features[Labels == i + 1, :], axis=0)
    #         covSamples = np.cov(Features[Labels == i + 1, :], rowvar=False)
    #         for j in range(classes):
    #             # covModel = model.loc[j, 'cov']
    #             meanModel = model.loc[j, 'mean']
    #             matrix[j, i] = JSdivergence(meanSamples, meanModel, np.size(meanSamples), covSamples, LDACov)
    # elif type_DA == 'QDA':
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
                FN += 1
            else:
                FP += 1
    return mcc(TP, TN, FP, FN)


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


######
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

    # elif type_DA == 'QDA':
    #     weights = []
    #     for cla in range(classes):
    #
    #         determinants = np.zeros(classes)
    #         for cla2 in range(classes):
    #             if cla == cla2:
    #                 determinants[cla2] = np.linalg.det(personCov)
    #             else:
    #                 determinants[cla2] = np.linalg.det(currentValues['cov'].at[cla2])
    #         tabDiscriminantValues = []
    #         if determinants.all() > 0:
    #             for cla2 in range(classes):
    #                 if cla == cla2:
    #                     tabDiscriminantValues.append([-.5 * np.log(determinants[cla2]) - .5 * np.dot(
    #                         np.dot((trainFeatures[a, :] - personMean), np.linalg.inv(personCov)),
    #                         (trainFeatures[a, :] - personMean).T) for a in range(len(trainFeatures))])
    #
    #                 else:
    #                     mean = currentValues['mean'].at[cla2]
    #                     covariance = currentValues['cov'].at[cla2]
    #                     tabDiscriminantValues.append([-.5 * np.log(determinants[cla2]) - .5 * np.dot(
    #                         np.dot((trainFeatures[a, :] - mean), np.linalg.inv(covariance)),
    #                         (trainFeatures[a, :] - mean).T) for a in range(len(trainFeatures))])
    #
    #         else:
    #             for cla2 in range(classes):
    #                 if cla == cla2:
    #                     tabDiscriminantValues.append([- .5 * np.dot(
    #                         np.dot((trainFeatures[a, :] - personMean), np.linalg.pinv(personCov)),
    #                         (trainFeatures[a, :] - personMean).T) for a in range(len(trainFeatures))])
    #
    #                 else:
    #                     mean = currentValues['mean'].at[cla2]
    #                     covariance = currentValues['cov'].at[cla2]
    #                     tabDiscriminantValues.append([- .5 * np.dot(
    #                         np.dot((trainFeatures[a, :] - mean), np.linalg.pinv(covariance)),
    #                         (trainFeatures[a, :] - mean).T) for a in range(len(trainFeatures))])
    #
    #         weights.append(calculationMcc(trainLabels, np.argmax(np.array(tabDiscriminantValues), axis=0) + 1, cla))
    #     return weights

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

        # if tabDiscriminant and not tabPseudoDiscriminant:
        #     personDiscriminantValues, tabDiscriminantValues = discriminantTab(
        #         trainFeatures, personMean, personCov, classes, currentValues)
        #     weights = []
        #     for cla in range(classes):
        #         auxTab = tabDiscriminantValues.copy()
        #         auxTab[cla, :] = personDiscriminantValues.copy()
        #         weights.append(calculationMcc(trainLabels, np.argmax(auxTab, axis=0) + 1, cla))
        #     return weights
        # elif not tabDiscriminant and tabPseudoDiscriminant:
        #     personPseudoDiscriminantValues, tabPseudoDiscriminantValues = pseudoDiscriminantTab(
        #         trainFeatures, personMean, personCov, classes, currentValues)
        #     weights = []
        #     for cla in range(classes):
        #         auxTab = tabPseudoDiscriminantValues.copy()
        #         auxTab[cla, :] = personPseudoDiscriminantValues.copy()
        #         weights.append(calculationMcc(trainLabels, np.argmax(auxTab, axis=0) + 1, cla))
        #     return weights
        # elif tabDiscriminant and tabPseudoDiscriminant:
        #     personDiscriminantValues, tabDiscriminantValues = discriminantTab(
        #         trainFeatures, personMean, personCov, classes, currentValues)
        #     personPseudoDiscriminantValues, tabPseudoDiscriminantValues = pseudoDiscriminantTab(
        #         trainFeatures, personMean, personCov, classes, currentValues)
        #     weights = []
        #     for cla in range(classes):
        #         if determinantsCurrentModel[cla] == float('NaN'):
        #             auxTab = tabPseudoDiscriminantValues.copy()
        #             auxTab[cla, :] = personPseudoDiscriminantValues.copy()
        #         else:
        #             auxTab = tabDiscriminantValues.copy()
        #             auxTab[cla, :] = personDiscriminantValues.copy()
        #
        #         weights.append(calculationMcc(trainLabels, np.argmax(auxTab, axis=0) + 1, cla))
        #     return weights


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

# %% Models Old
# %% reduce random the datasets
# def subsetTraining_One(trainFeatures, numSamples):
#     return trainFeatures[np.random.choice(len(trainFeatures), size=numSamples)]
# def model_PostProb_MSDA_reduce(currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures,
#                                fewModel, labeledGesturesFeatures, labeledGesturesLabels, type_DA,
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
#     # postProb_trainFeatures2 = post_probabilities_Calculation(trainFeatures, fewModel, classes, type_DA)
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
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
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
#             adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * unlabeledGestures['mean'].loc[0]
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#
#             adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
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
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
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
#                                            postProb_trainFeatures, fewModel,
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
#     postProb_trainFeatures2 = post_probabilities_Calculation(trainFeatures, fewModel, classes, type_DA)
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
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
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
#             adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * unlabeledGestures['mean'].loc[0]
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#
#             adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
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
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
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
#                                       fewModel,
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
#     postProb_trainFeatures2 = post_probabilities_Calculation(trainFeatures, fewModel, classes, type_DA)
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
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
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
#             adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * unlabeledGestures['mean'].loc[0]
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#
#             adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
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
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
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
#                                       fewModel, labeledGesturesFeatures, labeledGesturesLabels, type_DA,
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
#     postProb_trainFeatures2 = post_probabilities_Calculation(trainFeatures, fewModel, classes, type_DA)
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
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
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
#             adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * unlabeledGestures['mean'].loc[0]
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#
#             adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
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
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
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
#                                fewModel,
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
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Mean[cla]
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
#             adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * unlabeledGestures['mean'].loc[0]
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#
#             adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
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
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Mean[cla]
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
# def model_MSDA(currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewModel,
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
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
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
#             adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * \
#                                               unlabeledGestures['mean'].loc[0]
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#             # if type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
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
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             #
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
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
# def model_PostProb(currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewModel,
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
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
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
#             adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * \
#                                               unlabeledGestures['mean'].loc[0]
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#             # if type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
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
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             #
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
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
#                    fewModel, labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart):
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
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
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
#             adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * \
#                                               unlabeledGestures['mean'].loc[0]
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#             # if type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
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
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             #
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
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
# def model_MSDA_JS(currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewModel,
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
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
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
#             adaptiveModel.at[cla, 'mean_J'] = fewModel['mean'].loc[cla] * w_labeledGestures_Mean[cla] + weightsMean[0][
#                 cla] * \
#                                               unlabeledGestures['mean'].loc[0]
#             adaptiveModel.at[cla, 'wMean_J'] = w_labeledGestures_Mean[cla] + weightsMean[0][cla]
#             # if type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov_J'] = fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla] + weightsCov[0][
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
#             adaptiveModel.at[cla, 'mean'] = means[:, cla].sum() + fewModel['mean'].loc[cla] * w_labeledGestures_Mean[
#                 cla]
#             # if type_DA == 'LDA':
#             #     adaptiveModel.at[cla, 'cov'] = (gestureCov + currentModel['cov'].loc[cla]) / (
#             #             1 + currentModel['# gestures'].loc[0])
#             #
#             # elif type_DA == 'QDA':
#             adaptiveModel.at[cla, 'cov'] = covs[:, cla].sum() + fewModel['cov'].loc[cla] * w_labeledGestures_Cov[cla]
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
