import math

import numpy as np
import pandas as pd
from sklearn import preprocessing

import DA_BasedAdaptiveModels as adaptive
import DA_Classifiers as DA_Classifiers


# Upload Databases

def uploadDatabases(Database, featureSet=1):
    # Setting general variables
    path = '../'
    CH = 8
    segment = '_295ms'

    if Database == 'EPN':

        classes = 5
        peoplePriorK = 30
        peopleTest = 30
        combinationSet = list(range(1, 26))
        numberShots = 25
    elif Database == 'Nina5':

        classes = 18
        peoplePriorK = 10
        peopleTest = 10
        combinationSet = list(range(1, 5))
        numberShots = 4
    elif Database == 'Cote':

        classes = 7
        peoplePriorK = 19
        peopleTest = 17
        combinationSet = list(range(1, 5))
        numberShots = 4

    if featureSet == 1:
        # Setting variables
        Feature1 = 'logvar'

        numberFeatures = 1
        allFeatures = numberFeatures * CH
        # Getting Data
        logvarMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature1 + segment + '.csv',
                                     delimiter=',')

        if Database == 'Nina5':
            dataMatrix = logvarMatrix[:, 8:]
        else:
            dataMatrix = logvarMatrix.copy()

        labelsDataMatrix = dataMatrix[:, allFeatures + 2]


    elif featureSet == 2:
        # Setting variables
        Feature1 = 'mav'
        Feature2 = 'wl'
        Feature3 = 'zc'
        Feature4 = 'ssc'

        numberFeatures = 4
        allFeatures = numberFeatures * CH
        mavMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature1 + segment + '.csv', delimiter=',')
        wlMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature2 + segment + '.csv', delimiter=',')
        zcMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature3 + segment + '.csv', delimiter=',')
        sscMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature4 + segment + '.csv', delimiter=',')

        if Database == 'Nina5':
            dataMatrix = np.hstack(
                (mavMatrix[:, 8:CH * 2], wlMatrix[:, 8:CH * 2], zcMatrix[:, 8:CH * 2], sscMatrix[:, 8:]))
        else:
            dataMatrix = np.hstack((mavMatrix[:, :CH], wlMatrix[:, :CH], zcMatrix[:, :CH], sscMatrix[:, :]))

        labelsDataMatrix = dataMatrix[:, allFeatures + 2]

    elif featureSet == 3:
        # Setting variables
        Feature1 = 'lscale'
        Feature2 = 'mfl'
        Feature3 = 'msr'
        Feature4 = 'wamp'

        numberFeatures = 4
        allFeatures = numberFeatures * CH
        # Getting Data
        lscaleMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature1 + segment + '.csv',
                                     delimiter=',')
        mflMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature2 + segment + '.csv', delimiter=',')
        msrMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature3 + segment + '.csv', delimiter=',')
        wampMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature4 + segment + '.csv',
                                   delimiter=',')

        if Database == 'Nina5':
            dataMatrix = np.hstack(
                (lscaleMatrix[:, 8:CH * 2], mflMatrix[:, 8:CH * 2], msrMatrix[:, 8:CH * 2], wampMatrix[:, 8:]))
        else:
            dataMatrix = np.hstack((lscaleMatrix[:, :CH], mflMatrix[:, :CH], msrMatrix[:, :CH], wampMatrix[:, :]))

        labelsDataMatrix = dataMatrix[:, allFeatures + 2]

    return dataMatrix, numberFeatures, CH, classes, peoplePriorK, peopleTest, numberShots, combinationSet, allFeatures, labelsDataMatrix


def evaluation(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson, endPerson,
               allFeatures, typeDatabase, printR, shotStart):
    scaler = preprocessing.MinMaxScaler()
    results = pd.DataFrame(
        columns=['person', 'subset', '# shots', 'Feature Set'])
    idx = 0

    # if typeDatabase != 'Nina5':
    #     trainFeaturesGenPre = dataMatrix[dataMatrix[:, allFeatures + 1] <= peoplePriorK, :allFeatures]
    #     trainLabelsGenPre = dataMatrix[dataMatrix[:, allFeatures + 1] <= peoplePriorK, allFeatures + 2]

    for person in range(startPerson, endPerson + 1):

        # if typeDatabase == 'Nina5':
        #     trainFeaturesGenPre = dataMatrix[dataMatrix[:, allFeatures + 1] != person, :allFeatures]
        #     trainLabelsGenPre = dataMatrix[dataMatrix[:, allFeatures + 1] != person, allFeatures + 2]

        testFeatures = \
            dataMatrix[(dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 1), :allFeatures]
        testLabels = dataMatrix[
            (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 1), allFeatures + 2].T

        # oneShotFeatures = np.empty((0, allFeatures))
        # oneShotLabels = []
        # for shot in initialShots:
        fewShotFeatures = dataMatrix[
                          (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 0) & (
                                  dataMatrix[:, allFeatures + 3] <= shotStart), 0:allFeatures]
        fewShotLabels = dataMatrix[
            (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 0) & (
                    dataMatrix[:, allFeatures + 3] <= shotStart), allFeatures + 2].T

        fewShotFeatures = scaler.fit_transform(fewShotFeatures)
        fewShotModel = currentDistributionValues(fewShotFeatures, fewShotLabels, classes, allFeatures)

        dataPK, allFeaturesPK = preprocessingPK(dataMatrix, allFeatures, scaler)
        preTrainedDataMatrix = PKModels(dataPK, classes, peoplePriorK, person, allFeatures)

        k = 1 - (np.log(shotStart) / np.log(numberShots + 1))
        step = 1
        adaptedModel, _, _, _, _, _ = adaptive.OurModel(
            fewShotModel, preTrainedDataMatrix, classes, allFeatures, fewShotFeatures, fewShotLabels, step, 'QDA', k)

        semiSupervisedLearningModelLDA_shot = fewShotModel.copy()
        semiSupervisedLearningModelQDA_shot = fewShotModel.copy()
        semiSupervisedLearning_adaptationModelLDA_shot = adaptedModel.copy()
        semiSupervisedLearning_adaptationModelQDA_shot = adaptedModel.copy()

        semiSupervisedLearningModelLDA_accumulative = fewShotModel.copy()
        semiSupervisedLearningModelQDA_accumulative = fewShotModel.copy()
        semiSupervisedLearning_adaptationModelLDA_accumulative = adaptedModel.copy()
        semiSupervisedLearning_adaptationModelQDA_accumulative = adaptedModel.copy()

        for shot in range(shotStart + 1, numberShots + 1):
            trainFeatures_shot = dataMatrix[
                                 (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 0) &
                                 (shot == dataMatrix[:, allFeatures + 3]), 0:allFeatures]
            trainLabels_shot = dataMatrix[
                (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 0) &
                (shot == dataMatrix[:, allFeatures + 3]), allFeatures + 2].T
            trainRep_shot = dataMatrix[
                (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 0) &
                (shot == dataMatrix[:, allFeatures + 3]), allFeatures + 3].T

            trainFeatures = dataMatrix[
                            (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 0) & (
                                    shotStart < dataMatrix[:, allFeatures + 3]) &
                            (shot >= dataMatrix[:, allFeatures + 3]), 0:allFeatures]
            trainLabels = dataMatrix[
                (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 0) & (
                        shotStart < dataMatrix[:, allFeatures + 3]) &
                (shot >= dataMatrix[:, allFeatures + 3]), allFeatures + 2].T

            trainRep = dataMatrix[
                (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 0) & (
                        shotStart < dataMatrix[:, allFeatures + 3]) &
                (shot >= dataMatrix[:, allFeatures + 3]), allFeatures + 3].T

            subset = list(range(shotStart, shot + 1))

            trainFeatures = scaler.transform(trainFeatures)
            trainFeatures_shot = scaler.transform(trainFeatures_shot)
            # trainFeaturesGen = scaler.transform(trainFeaturesGen)
            testFeaturesTransform = scaler.transform(testFeatures)

            # currentValues = currentDistributionValues(trainFeatures, trainLabels, classes, allFeaturesPK)
            # pkValues = currentDistributionValues(trainFeaturesGen, trainLabelsGen, classes, allFeaturesPK)

            results, idx, semiSupervisedLearningModelLDA_shot, semiSupervisedLearningModelQDA_shot, \
            semiSupervisedLearning_adaptationModelLDA_shot, semiSupervisedLearning_adaptationModelQDA_shot, \
            semiSupervisedLearningModelLDA_accumulative, semiSupervisedLearningModelQDA_accumulative, \
            semiSupervisedLearning_adaptationModelLDA_accumulative, semiSupervisedLearning_adaptationModelQDA_accumulative = \
                resultsDataframeUnsupervised(
                    trainFeatures, trainLabels, trainRep, trainFeatures_shot, trainLabels_shot, trainRep_shot, classes,
                    allFeaturesPK, results,
                    testFeaturesTransform, testLabels, idx, person, subset, featureSet, nameFile, printR,
                    fewShotFeatures, fewShotLabels, semiSupervisedLearningModelLDA_shot,
                    semiSupervisedLearningModelQDA_shot, semiSupervisedLearning_adaptationModelLDA_shot,
                    semiSupervisedLearning_adaptationModelQDA_shot, semiSupervisedLearningModelLDA_accumulative,
                    semiSupervisedLearningModelQDA_accumulative, semiSupervisedLearning_adaptationModelLDA_accumulative,
                    semiSupervisedLearning_adaptationModelQDA_accumulative, fewShotModel, adaptedModel, typeDatabase)

    return results


def PKModels(dataMatrix, classes, peoplePriorK, evaluatedPerson, allFeatures):
    preTrainedDataMatrix = pd.DataFrame(columns=['mean', 'cov', 'class', 'person'])
    indx = 0

    people = list(range(1, peoplePriorK + 1))
    for cl in range(1, classes + 1):
        for person in people:
            if person != evaluatedPerson:
                auxData = dataMatrix[
                          (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures + 2] == cl),
                          0:allFeatures]
                preTrainedDataMatrix.at[indx, 'cov'] = np.cov(auxData, rowvar=False)
                preTrainedDataMatrix.at[indx, 'mean'] = np.mean(auxData, axis=0)
                preTrainedDataMatrix.at[indx, 'class'] = cl
                preTrainedDataMatrix.at[indx, 'person'] = person
                indx += 1

    return preTrainedDataMatrix


# Unsupervised

def resultsDataframeUnsupervised(
        trainFeatures, trainLabels, trainRep, trainFeatures_shot, trainLabels_shot, trainRep_shot, classes, allFeatures,
        results, testFeatures, testLabels, idx, person, subset, featureSet, nameFile, printR, fewShotFeatures,
        fewShotLabels, semiSupervisedLearningModelLDA_shot, semiSupervisedLearningModelQDA_shot,
        semiSupervisedLearning_adaptationModelLDA_shot, semiSupervisedLearning_adaptationModelQDA_shot,
        semiSupervisedLearningModelLDA_accumulative, semiSupervisedLearningModelQDA_accumulative,
        semiSupervisedLearning_adaptationModelLDA_accumulative, semiSupervisedLearning_adaptationModelQDA_accumulative,
        fewShotModel, adaptedModel, typeDatabase):
    step = 1
    numSamples = 50
    fewShotFeatures, fewShotLabels = adaptive.subsetTraining(fewShotFeatures, fewShotLabels, numSamples, classes)

    # lenOneShot = len(oneShotLabels)
    # ModelLDA = ldaUnsupervisedModel.copy()
    # oneModelQDA = qdaUnsupervisedModel.copy()

    semiPostProbabilitiesLDA = DA_Classifiers.scoreModelLDA_ClassificationUnsupervised(
        trainFeatures, fewShotModel, classes, trainLabels, trainRep)

    semiPostProbabilitiesQDA = DA_Classifiers.scoreModelQDA_ClassificationUnsupervised(
        trainFeatures, fewShotModel, classes, trainLabels, trainRep)

    semi_adaptationPostProbabilitiesLDA = DA_Classifiers.scoreModelLDA_ClassificationUnsupervised(
        trainFeatures, adaptedModel, classes, trainLabels, trainRep)

    semi_adaptationPostProbabilitiesQDA = DA_Classifiers.scoreModelQDA_ClassificationUnsupervised(
        trainFeatures, adaptedModel, classes, trainLabels, trainRep)

    semiSupervisedLearningModelLDA, _, _, _, _, _ = adaptive.OurModelUnsupervisedAllProb(
        fewShotModel, semiPostProbabilitiesLDA, classes, allFeatures, fewShotFeatures, fewShotLabels,
        fewShotModel, step, 'LDA', typeDatabase)

    semiSupervisedLearningModelQDA, _, _, _, _, _ = adaptive.OurModelUnsupervisedAllProb(
        fewShotModel, semiPostProbabilitiesQDA, classes, allFeatures, fewShotFeatures, fewShotLabels,
        fewShotModel, step, 'QDA', typeDatabase)

    semiSupervisedLearning_adaptationModelLDA, _, _, _, _, _ = adaptive.OurModelUnsupervisedAllProb(
        adaptedModel, semi_adaptationPostProbabilitiesLDA, classes, allFeatures, fewShotFeatures, fewShotLabels,
        adaptedModel, step, 'LDA', typeDatabase)

    semiSupervisedLearning_adaptationModelQDA, _, _, _, _, _ = adaptive.OurModelUnsupervisedAllProb(
        adaptedModel, semi_adaptationPostProbabilitiesQDA, classes, allFeatures, fewShotFeatures, fewShotLabels,
        adaptedModel, step, 'QDA', typeDatabase)

    # shot

    semiPostProbabilitiesLDA_shot = DA_Classifiers.scoreModelLDA_ClassificationUnsupervised(
        trainFeatures_shot, semiSupervisedLearningModelLDA_shot, classes, trainLabels_shot, trainRep_shot)

    semiPostProbabilitiesQDA_shot = DA_Classifiers.scoreModelQDA_ClassificationUnsupervised(
        trainFeatures_shot, semiSupervisedLearningModelQDA_shot, classes, trainLabels_shot, trainRep_shot)

    semi_adaptationPostProbabilitiesLDA_shot = DA_Classifiers.scoreModelLDA_ClassificationUnsupervised(
        trainFeatures_shot, semiSupervisedLearning_adaptationModelLDA_shot, classes, trainLabels_shot, trainRep_shot)

    semi_adaptationPostProbabilitiesQDA_shot = DA_Classifiers.scoreModelQDA_ClassificationUnsupervised(
        trainFeatures_shot, semiSupervisedLearning_adaptationModelQDA_shot, classes, trainLabels_shot, trainRep_shot)

    semiSupervisedLearningModelLDA_shot, _, _, _, _, _ = adaptive.OurModelUnsupervisedAllProb(
        semiSupervisedLearningModelLDA_shot, semiPostProbabilitiesLDA_shot, classes, allFeatures, fewShotFeatures,
        fewShotLabels, fewShotModel, step, 'LDA', typeDatabase)

    semiSupervisedLearningModelQDA_shot, _, _, _, _, _ = adaptive.OurModelUnsupervisedAllProb(
        semiSupervisedLearningModelQDA_shot, semiPostProbabilitiesQDA_shot, classes, allFeatures, fewShotFeatures,
        fewShotLabels, fewShotModel, step, 'QDA', typeDatabase)

    semiSupervisedLearning_adaptationModelLDA_shot, _, _, _, _, _ = adaptive.OurModelUnsupervisedAllProb(
        semiSupervisedLearning_adaptationModelLDA_shot, semi_adaptationPostProbabilitiesLDA_shot, classes, allFeatures,
        fewShotFeatures, fewShotLabels, adaptedModel, step, 'LDA', typeDatabase)

    semiSupervisedLearning_adaptationModelQDA_shot, _, _, _, _, _ = adaptive.OurModelUnsupervisedAllProb(
        semiSupervisedLearning_adaptationModelQDA_shot, semi_adaptationPostProbabilitiesQDA_shot, classes, allFeatures,
        fewShotFeatures, fewShotLabels, adaptedModel, step, 'QDA', typeDatabase)

    # accumulative

    semiPostProbabilitiesLDA_accumulative = DA_Classifiers.scoreModelLDA_ClassificationUnsupervised(
        trainFeatures, semiSupervisedLearningModelLDA_accumulative, classes, trainLabels, trainRep)

    semiPostProbabilitiesQDA_accumulative = DA_Classifiers.scoreModelQDA_ClassificationUnsupervised(
        trainFeatures, semiSupervisedLearningModelQDA_accumulative, classes, trainLabels, trainRep)

    semi_adaptationPostProbabilitiesLDA_accumulative = DA_Classifiers.scoreModelLDA_ClassificationUnsupervised(
        trainFeatures, semiSupervisedLearning_adaptationModelLDA_accumulative, classes, trainLabels, trainRep)

    semi_adaptationPostProbabilitiesQDA_accumulative = DA_Classifiers.scoreModelQDA_ClassificationUnsupervised(
        trainFeatures, semiSupervisedLearning_adaptationModelQDA_accumulative, classes, trainLabels, trainRep)

    semiSupervisedLearningModelLDA_accumulative, _, _, _, _, _ = adaptive.OurModelUnsupervisedAllProb(
        semiSupervisedLearningModelLDA_accumulative, semiPostProbabilitiesLDA_accumulative, classes, allFeatures,
        fewShotFeatures, fewShotLabels, fewShotModel, step, 'LDA', typeDatabase)

    semiSupervisedLearningModelQDA_accumulative, _, _, _, _, _ = adaptive.OurModelUnsupervisedAllProb(
        semiSupervisedLearningModelQDA_accumulative, semiPostProbabilitiesQDA_accumulative, classes, allFeatures,
        fewShotFeatures, fewShotLabels, fewShotModel, step, 'QDA', typeDatabase)

    semiSupervisedLearning_adaptationModelLDA_accumulative, _, _, _, _, _ = adaptive.OurModelUnsupervisedAllProb(
        semiSupervisedLearning_adaptationModelLDA_accumulative, semi_adaptationPostProbabilitiesLDA_accumulative,
        classes, allFeatures, fewShotFeatures, fewShotLabels, adaptedModel, step, 'LDA', typeDatabase)

    semiSupervisedLearning_adaptationModelQDA_accumulative, _, _, _, _, _ = adaptive.OurModelUnsupervisedAllProb(
        semiSupervisedLearning_adaptationModelQDA_accumulative, semi_adaptationPostProbabilitiesQDA_accumulative,
        classes, allFeatures, fewShotFeatures, fewShotLabels, adaptedModel, step, 'QDA', typeDatabase)

    results.at[idx, 'person'] = person
    results.at[idx, 'subset'] = subset
    results.at[idx, '# shots'] = np.size(subset)
    results.at[idx, 'Feature Set'] = featureSet

    # LDA results
    print('AccLDAfew')
    results.at[idx, 'AccLDAfew'], _, _, _ = DA_Classifiers.accuracyModelLDAconfusionMatrix(
        testFeatures, testLabels, fewShotModel, classes)

    print('AccLDAadapted')
    results.at[idx, 'AccLDAadapted'], _, _, _ = DA_Classifiers.accuracyModelLDAconfusionMatrix(
        testFeatures, testLabels, adaptedModel, classes)

    print('AccLDAsemi')
    results.at[idx, 'AccLDAsemi'], _, _, _ = DA_Classifiers.accuracyModelLDAconfusionMatrix(
        testFeatures, testLabels, semiSupervisedLearningModelLDA, classes)

    print('AccLDAsemiAdapt')
    results.at[idx, 'AccLDAsemiAdapt'], _, _, _ = DA_Classifiers.accuracyModelLDAconfusionMatrix(
        testFeatures, testLabels, semiSupervisedLearning_adaptationModelLDA, classes)

    print('AccLDAsemi_shot')
    results.at[idx, 'AccLDAsemi_shot'], _, _, _ = DA_Classifiers.accuracyModelLDAconfusionMatrix(
        testFeatures, testLabels, semiSupervisedLearningModelLDA_shot, classes)

    print('AccLDAsemiAdapt_shot')
    results.at[idx, 'AccLDAsemiAdapt_shot'], _, _, _ = DA_Classifiers.accuracyModelLDAconfusionMatrix(
        testFeatures, testLabels, semiSupervisedLearning_adaptationModelLDA_shot, classes)

    print('AccLDAsemi_accumulative')
    results.at[idx, 'AccLDAsemi_accumulative'], _, _, _ = DA_Classifiers.accuracyModelLDAconfusionMatrix(
        testFeatures, testLabels, semiSupervisedLearningModelLDA_accumulative, classes)

    print('AccLDAsemiAdapt_accumulative')
    results.at[idx, 'AccLDAsemiAdapt_accumulative'], _, _, _ = DA_Classifiers.accuracyModelLDAconfusionMatrix(
        testFeatures, testLabels, semiSupervisedLearning_adaptationModelLDA_accumulative, classes)

    # QDA

    print('AccQDAfew')
    results.at[idx, 'AccQDAfew'], _, _, _ = DA_Classifiers.accuracyModelQDAconfusionMatrix(
        testFeatures, testLabels, fewShotModel, classes)

    print('AccQDAadapted')
    results.at[idx, 'AccQDAadapted'], _, _, _ = DA_Classifiers.accuracyModelQDAconfusionMatrix(
        testFeatures, testLabels, adaptedModel, classes)

    print('AccQDAsemi')
    results.at[idx, 'AccQDAsemi'], _, _, _ = DA_Classifiers.accuracyModelQDAconfusionMatrix(
        testFeatures, testLabels, semiSupervisedLearningModelQDA, classes)

    print('AccQDAsemiAdapt')
    results.at[idx, 'AccQDAsemiAdapt'], _, _, _ = DA_Classifiers.accuracyModelQDAconfusionMatrix(
        testFeatures, testLabels, semiSupervisedLearning_adaptationModelQDA, classes)

    print('AccQDAsemi_shot')
    results.at[idx, 'AccQDAsemi_shot'], _, _, _ = DA_Classifiers.accuracyModelQDAconfusionMatrix(
        testFeatures, testLabels, semiSupervisedLearningModelQDA_shot, classes)

    print('AccQDAsemiAdapt_shot')
    results.at[idx, 'AccQDAsemiAdapt_shot'], _, _, _ = DA_Classifiers.accuracyModelQDAconfusionMatrix(
        testFeatures, testLabels, semiSupervisedLearning_adaptationModelQDA_shot, classes)

    print('AccQDAsemi_accumulative')
    results.at[idx, 'AccQDAsemi_accumulative'], _, _, _ = DA_Classifiers.accuracyModelQDAconfusionMatrix(
        testFeatures, testLabels, semiSupervisedLearningModelQDA_accumulative, classes)

    print('AccQDAsemiAdapt_accumulative')
    results.at[idx, 'AccQDAsemiAdapt_accumulative'], _, _, _ = DA_Classifiers.accuracyModelQDAconfusionMatrix(
        testFeatures, testLabels, semiSupervisedLearning_adaptationModelQDA_accumulative, classes)

    if nameFile is not None:
        results.to_csv(nameFile)
    if printR:
        print(featureSet)
        print('Results: person= ', person, ' shot set= ', subset)
        print(results.loc[idx])

    idx += 1

    return results, idx, semiSupervisedLearningModelLDA_shot, semiSupervisedLearningModelQDA_shot, \
           semiSupervisedLearning_adaptationModelLDA_shot, semiSupervisedLearning_adaptationModelQDA_shot, \
           semiSupervisedLearningModelLDA_accumulative, semiSupervisedLearningModelQDA_accumulative, \
           semiSupervisedLearning_adaptationModelLDA_accumulative, semiSupervisedLearning_adaptationModelQDA_accumulative


# Auxiliar functions of the evaluation


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


def preprocessingPK(dataMatrix, allFeatures, scaler):
    dataMatrixFeatures = scaler.transform(dataMatrix[:, :allFeatures])
    return np.hstack((dataMatrixFeatures, dataMatrix[:, allFeatures:])), np.size(dataMatrixFeatures, axis=1)

