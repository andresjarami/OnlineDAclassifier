import math

import numpy as np
import pandas as pd
from sklearn import preprocessing

import DA_BasedAdaptiveModels as adaptive
import DA_Classifiers as DA_Classifiers
import SemiSupervised as SemiSupervised


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
               allFeatures, typeDatabase, printR, k, shotStart):
    scaler = preprocessing.MinMaxScaler()
    results = pd.DataFrame(
        columns=['person', 'subset', '# shots', 'Feature Set'])
    idx = 0

    for person in range(startPerson, endPerson + 1):

        testFeatures = \
            dataMatrix[(dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 1), :allFeatures]
        testLabels = dataMatrix[
            (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 1), allFeatures + 2].T

        fewShotFeatures = dataMatrix[
                          (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 0) & (
                                  dataMatrix[:, allFeatures + 3] <= shotStart), 0:allFeatures]
        fewShotLabels = dataMatrix[
            (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 0) & (
                    dataMatrix[:, allFeatures + 3] <= shotStart), allFeatures + 2].T

        fewShotFeatures = scaler.fit_transform(fewShotFeatures)
        testFeatures = scaler.transform(testFeatures)
        fewShotModel = currentDistributionValues(fewShotFeatures, fewShotLabels, classes, allFeatures, shotStart)

        fewShotFeatures, fewShotLabels = adaptive.subsetTraining(fewShotFeatures, fewShotLabels, 50, classes)

        unlabeledGesturesLDA = pd.DataFrame(columns=['mean', 'cov', 'postProb', 'wMean', 'wCov', 'features'])
        unlabeledGesturesQDA = pd.DataFrame(columns=['mean', 'cov', 'postProb', 'wMean', 'wCov', 'features'])
        '''              
        # adaptive model
        
        dataPK, allFeaturesPK = preprocessingPK(dataMatrix, allFeatures, scaler)
        preTrainedDataMatrix = PKModels(dataPK, classes, peoplePriorK, person, allFeatures)
        
        k = 1 - (np.log(shotStart) / np.log(numberShots + 1))
        step = 1
        adaptedModel, _, _, _, _, _ = adaptive.OurModel(
            fewShotModel, preTrainedDataMatrix, classes, allFeatures, fewShotFeatures, fewShotLabels, step, 'QDA', k)
        '''
        adaptedModel = np.zeros(5)

        proposedModelLDA = fewShotModel.copy()
        proposedModelQDA = fewShotModel.copy()
        nShots = 0
        for shot in range(shotStart + 1, numberShots + 1):
            for cl in range(1, classes + 1):
                trainFeatures = dataMatrix[
                                (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 0) &
                                (dataMatrix[:, allFeatures + 3] == shot) & (dataMatrix[:, allFeatures + 2] == cl),
                                0:allFeatures]

                nShots += 1

                trainFeatures = scaler.transform(trainFeatures)

                results, idx, proposedModelLDA, proposedModelQDA, unlabeledGesturesLDA, unlabeledGesturesQDA = \
                    resultsDataframeUnsupervised(trainFeatures, classes, results, testFeatures, testLabels, idx, person,
                                                 nShots, featureSet, nameFile, printR, fewShotFeatures, fewShotLabels,
                                                 proposedModelLDA, proposedModelQDA, fewShotModel, unlabeledGesturesLDA,
                                                 unlabeledGesturesQDA, k, shotStart)

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

def adaptPrint(currentModel, unlabeledGestures, type_DA, trainFeatures, classes, fewShotFeatures, fewShotLabels,
               fewShotModel, results, idx, testFeatures, testLabels, k, N, typeModel):
    name = type_DA + '_ACC_' + typeModel
    print(name)

    postProb_trainFeatures = SemiSupervised.post_probabilities_Calculation(trainFeatures, currentModel, classes,
                                                                           type_DA)
    if type_DA == 'LDA':
        if typeModel == 'PostProb_MSDA':
            adaptedModel, results.at[
                idx, 'time_' + name], unlabeledGestures = SemiSupervised.model_PostProb_MSDA(
                currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewShotModel,
                fewShotFeatures, fewShotLabels, type_DA, k, N)
        elif typeModel == 'PostProb':
            adaptedModel, results.at[
                idx, 'time_' + name], unlabeledGestures = SemiSupervised.model_PostProb(
                currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewShotModel,
                fewShotFeatures, fewShotLabels, type_DA, k, N)
        elif typeModel == 'MSDA':
            adaptedModel, results.at[
                idx, 'time_' + name], unlabeledGestures = SemiSupervised.model_MSDA(
                currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewShotModel,
                fewShotFeatures, fewShotLabels, type_DA, k, N)

        results.at[idx, name], _ = DA_Classifiers.accuracyModelLDA(testFeatures, testLabels, adaptedModel, classes)

    elif type_DA == 'QDA':
        if typeModel == 'PostProb_MSDA':
            adaptedModel, results.at[
                idx, 'time_' + name], unlabeledGestures = SemiSupervised.model_PostProb_MSDA(
                currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewShotModel,
                fewShotFeatures, fewShotLabels, type_DA, k, N)
        elif typeModel == 'PostProb':
            adaptedModel, results.at[
                idx, 'time_' + name], unlabeledGestures = SemiSupervised.model_PostProb(
                currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewShotModel,
                fewShotFeatures, fewShotLabels, type_DA, k, N)
        elif typeModel == 'MSDA':
            adaptedModel, results.at[
                idx, 'time_' + name], unlabeledGestures = SemiSupervised.model_MSDA(
                currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewShotModel,
                fewShotFeatures, fewShotLabels, type_DA, k, N)

        results.at[idx, name], _ = DA_Classifiers.accuracyModelQDA(testFeatures, testLabels, adaptedModel, classes)

    return adaptedModel, results, unlabeledGestures


def resultsDataframeUnsupervised(
        trainFeatures, classes, results, testFeatures, testLabels, idx, person, nShots, featureSet, nameFile, printR,
        fewShotFeatures, fewShotLabels, proposedModelLDA, proposedModelQDA, fewShotModel, unlabeledGesturesLDA,
        unlabeledGesturesQDA, k, N):
    # step = 1
    # numSamples = 50
    # fewShotFeatures, fewShotLabels = adaptive.subsetTraining(fewShotFeatures, fewShotLabels, numSamples, classes)

    type_DA_set = ['LDA', 'QDA']
    for type_DA in type_DA_set:
        if type_DA == 'LDA':

            print('AccLDAfew')
            results.at[idx, 'AccLDAfew'], _ = DA_Classifiers.accuracyModelLDA(
                testFeatures, testLabels, fewShotModel, classes)

            # print(unlabeledGesturesLDA[['postProb']])

            proposedModelLDA, results, unlabeledGesturesLDA = adaptPrint(
                proposedModelLDA, unlabeledGesturesLDA, type_DA, trainFeatures, classes, fewShotFeatures,
                fewShotLabels, fewShotModel, results, idx, testFeatures, testLabels, k, N,typeModel='PostProb_MSDA')
            proposedModelLDA, results, unlabeledGesturesLDA = adaptPrint(
                proposedModelLDA, unlabeledGesturesLDA, type_DA, trainFeatures, classes, fewShotFeatures,
                fewShotLabels, fewShotModel, results, idx, testFeatures, testLabels, k, N, typeModel='PostProb')
            proposedModelLDA, results, unlabeledGesturesLDA = adaptPrint(
                proposedModelLDA, unlabeledGesturesLDA, type_DA, trainFeatures, classes, fewShotFeatures,
                fewShotLabels, fewShotModel, results, idx, testFeatures, testLabels, k, N, typeModel='MSDA')


        elif type_DA == 'QDA':

            print('AccQDAfew')
            results.at[idx, 'AccQDAfew'], _ = DA_Classifiers.accuracyModelQDA(
                testFeatures, testLabels, fewShotModel, classes)

            proposedModelQDA, results, unlabeledGesturesQDA = adaptPrint(
                proposedModelQDA, unlabeledGesturesQDA, type_DA, trainFeatures, classes, fewShotFeatures,
                fewShotLabels, fewShotModel, results, idx, testFeatures, testLabels, k, N,typeModel='PostProb_MSDA')
            proposedModelQDA, results, unlabeledGesturesQDA = adaptPrint(
                proposedModelQDA, unlabeledGesturesQDA, type_DA, trainFeatures, classes, fewShotFeatures,
                fewShotLabels, fewShotModel, results, idx, testFeatures, testLabels, k, N, typeModel='PostProb')
            proposedModelQDA, results, unlabeledGesturesQDA = adaptPrint(
                proposedModelQDA, unlabeledGesturesQDA, type_DA, trainFeatures, classes, fewShotFeatures,
                fewShotLabels, fewShotModel, results, idx, testFeatures, testLabels, k, N, typeModel='MSDA')

    results.at[idx, 'person'] = person
    # results.at[idx, 'subset'] = subset
    results.at[idx, '# shots'] = nShots
    results.at[idx, 'Feature Set'] = featureSet

    if nameFile is not None:
        results.to_csv(nameFile)
    if printR:
        print(featureSet)
        print('Results: person= ', person, ' shot = ', nShots)
        print(results.loc[idx])

    idx += 1

    return results, idx, proposedModelLDA, proposedModelQDA, unlabeledGesturesLDA, unlabeledGesturesQDA


# Auxiliar functions of the evaluation


def currentDistributionValues(trainFeatures, trainLabels, classes, allFeatures, shotStart):
    currentValues = pd.DataFrame(columns=['cov', 'mean', 'class', 'weight_mean', 'weight_cov'])
    trainLabelsAux = trainLabels[np.newaxis]
    Matrix = np.hstack((trainFeatures, trainLabelsAux.T))
    for cla in range(classes):
        currentValues.at[cla, 'cov'] = np.cov(Matrix[np.where((Matrix[:, allFeatures] == cla + 1)), 0:allFeatures][0],
                                              rowvar=False)
        currentValues.at[cla, 'mean'] = np.mean(Matrix[np.where((Matrix[:, allFeatures] == cla + 1)), 0:allFeatures][0],
                                                axis=0)
        currentValues.at[cla, 'class'] = cla + 1
        currentValues.at[cla, 'weight_mean'] = shotStart
        currentValues.at[cla, 'weight_cov'] = shotStart

    return currentValues


def preprocessingPK(dataMatrix, allFeatures, scaler):
    dataMatrixFeatures = scaler.transform(dataMatrix[:, :allFeatures])
    return np.hstack((dataMatrixFeatures, dataMatrix[:, allFeatures:])), np.size(dataMatrixFeatures, axis=1)
