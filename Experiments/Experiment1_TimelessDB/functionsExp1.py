import math

import numpy as np
import pandas as pd
from sklearn import preprocessing

import DA_Classifiers as DA_Classifiers
import SemiSupervised as SemiSupervised
import DA_BasedAdaptiveModels as adaptive


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
               allFeatures, typeDatabase, printR, samplesInMemory, shotStart, randomSeed, expTimes):
    scaler = preprocessing.MinMaxScaler()
    results = pd.DataFrame(
        columns=['person', '# shots', 'shot_class', 'Feature Set'])
    idx = 0
    np.random.seed(randomSeed)
    for person in range(startPerson, endPerson + 1):

        for randomExperiments in range(expTimes):

            testFeatures = \
                dataMatrix[(dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 1), :allFeatures]
            testLabels = dataMatrix[
                (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 1), allFeatures + 2].T

            labeledGesturesFeatures = dataMatrix[
                                      (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 0) & (
                                              dataMatrix[:, allFeatures + 3] <= shotStart), 0:allFeatures]
            labeledGesturesLabels = dataMatrix[
                (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 0) & (
                        dataMatrix[:, allFeatures + 3] <= shotStart), allFeatures + 2].T

            labeledGesturesFeatures = scaler.fit_transform(labeledGesturesFeatures)
            testFeatures = scaler.transform(testFeatures)
            fewShotModel = currentDistributionValues(labeledGesturesFeatures, labeledGesturesLabels, classes,
                                                     allFeatures,
                                                     shotStart)

            labeledGesturesFeatures, labeledGesturesLabels = adaptive.subsetTraining(
                labeledGesturesFeatures, labeledGesturesLabels, 50, classes)

            unlabeledGesturesLDA_PostProb_MSDA = pd.DataFrame(
                columns=['mean', 'cov', 'postProb', 'wMean', 'wCov', 'features'])
            unlabeledGesturesQDA_PostProb_MSDA = pd.DataFrame(
                columns=['mean', 'cov', 'postProb', 'wMean', 'wCov', 'features'])
            unlabeledGesturesLDA_PostProb = pd.DataFrame(
                columns=['mean', 'cov', 'postProb', 'wMean', 'wCov', 'features'])
            unlabeledGesturesQDA_PostProb = pd.DataFrame(
                columns=['mean', 'cov', 'postProb', 'wMean', 'wCov', 'features'])
            unlabeledGesturesLDA_MSDA = pd.DataFrame(columns=['mean', 'cov', 'postProb', 'wMean', 'wCov', 'features'])
            unlabeledGesturesQDA_MSDA = pd.DataFrame(columns=['mean', 'cov', 'postProb', 'wMean', 'wCov', 'features'])
            unlabeledGesturesLDA_Baseline = pd.DataFrame(
                columns=['mean', 'cov', 'postProb', 'wMean', 'wCov', 'features'])
            unlabeledGesturesQDA_Baseline = pd.DataFrame(
                columns=['mean', 'cov', 'postProb', 'wMean', 'wCov', 'features'])
            unlabeledGesturesLDA_MSDA_KL = pd.DataFrame(
                columns=['mean', 'cov', 'postProb', 'wMean', 'wCov', 'features'])
            unlabeledGesturesQDA_MSDA_KL = pd.DataFrame(
                columns=['mean', 'cov', 'postProb', 'wMean', 'wCov', 'features'])
            unlabeledGesturesLDA_PostProb_MSDA_KL = pd.DataFrame(
                columns=['mean', 'cov', 'postProb', 'wMean', 'wCov', 'features'])
            unlabeledGesturesQDA_PostProb_MSDA_KL = pd.DataFrame(
                columns=['mean', 'cov', 'postProb', 'wMean', 'wCov', 'features'])

            '''              
            # adaptive model
            
            dataPK, allFeaturesPK = preprocessingPK(dataMatrix, allFeatures, scaler)
            preTrainedDataMatrix = PKModels(dataPK, classes, peoplePriorK, person, allFeatures)
            
            k = 1 - (np.log(shotStart) / np.log(numberShots + 1))
            step = 1
            adaptedModel, _, _, _, _, _ = adaptive.OurModel(
                fewShotModel, preTrainedDataMatrix, classes, allFeatures, labeledGesturesFeatures, labeledGesturesLabels, step, 'QDA', k)
            '''
            adaptedModel = np.zeros(5)

            proposedModelLDA_PostProb_MSDA = fewShotModel.copy()
            proposedModelQDA_PostProb_MSDA = fewShotModel.copy()
            proposedModelLDA_PostProb = fewShotModel.copy()
            proposedModelQDA_PostProb = fewShotModel.copy()
            proposedModelLDA_MSDA = fewShotModel.copy()
            proposedModelQDA_MSDA = fewShotModel.copy()
            proposedModelLDA_Baseline = fewShotModel.copy()
            proposedModelQDA_Baseline = fewShotModel.copy()
            proposedModelLDA_MSDA_KL = fewShotModel.copy()
            proposedModelQDA_MSDA_KL = fewShotModel.copy()
            proposedModelLDA_PostProb_MSDA_KL = fewShotModel.copy()
            proposedModelQDA_PostProb_MSDA_KL = fewShotModel.copy()

            nShots = 0
            randomGestures = [[shot, cl] for shot in range(shotStart + 1, numberShots + 1) for cl in
                              range(1, classes + 1)]
            for rand in list(np.random.permutation(randomGestures)):
                # print('Gesture', rand)
                trainFeatures = dataMatrix[
                                (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 0) &
                                (dataMatrix[:, allFeatures + 3] == rand[0]) & (
                                        dataMatrix[:, allFeatures + 2] == rand[1]), 0:allFeatures]
                trainLabels = dataMatrix[
                    (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 0) &
                    (dataMatrix[:, allFeatures + 3] == rand[0]) & (
                            dataMatrix[:, allFeatures + 2] == rand[1]), allFeatures + 2]

                nShots += 1
                trainFeatures = scaler.transform(trainFeatures)

                results, idx, proposedModelLDA_PostProb_MSDA, proposedModelQDA_PostProb_MSDA, proposedModelLDA_PostProb, \
                proposedModelQDA_PostProb, proposedModelLDA_MSDA, proposedModelQDA_MSDA, proposedModelLDA_Baseline, \
                proposedModelQDA_Baseline, proposedModelLDA_MSDA_KL, proposedModelQDA_MSDA_KL, \
                proposedModelLDA_PostProb_MSDA_KL, proposedModelQDA_PostProb_MSDA_KL, \
                unlabeledGesturesLDA_PostProb_MSDA, unlabeledGesturesQDA_PostProb_MSDA, \
                unlabeledGesturesLDA_PostProb, unlabeledGesturesQDA_PostProb, unlabeledGesturesLDA_MSDA, \
                unlabeledGesturesQDA_MSDA, unlabeledGesturesLDA_Baseline, unlabeledGesturesQDA_Baseline, \
                unlabeledGesturesLDA_MSDA_KL, unlabeledGesturesQDA_MSDA_KL, unlabeledGesturesLDA_PostProb_MSDA_KL, \
                unlabeledGesturesQDA_PostProb_MSDA_KL = \
                    resultsDataframeUnsupervised(trainFeatures, trainLabels, classes, results, testFeatures, testLabels,
                                                 idx, person, nShots, rand, featureSet, nameFile, printR,
                                                 labeledGesturesFeatures, labeledGesturesLabels,
                                                 proposedModelLDA_PostProb_MSDA, proposedModelQDA_PostProb_MSDA,
                                                 proposedModelLDA_PostProb, proposedModelQDA_PostProb,
                                                 proposedModelLDA_MSDA, proposedModelQDA_MSDA,
                                                 proposedModelLDA_Baseline, proposedModelQDA_Baseline,
                                                 proposedModelLDA_MSDA_KL, proposedModelQDA_MSDA_KL,
                                                 proposedModelLDA_PostProb_MSDA_KL, proposedModelQDA_PostProb_MSDA_KL,
                                                 fewShotModel, unlabeledGesturesLDA_PostProb_MSDA,
                                                 unlabeledGesturesQDA_PostProb_MSDA, unlabeledGesturesLDA_PostProb,
                                                 unlabeledGesturesQDA_PostProb, unlabeledGesturesLDA_MSDA,
                                                 unlabeledGesturesQDA_MSDA, unlabeledGesturesLDA_Baseline,
                                                 unlabeledGesturesQDA_Baseline, unlabeledGesturesLDA_MSDA_KL,
                                                 unlabeledGesturesQDA_MSDA_KL, unlabeledGesturesLDA_PostProb_MSDA_KL,
                                                 unlabeledGesturesQDA_PostProb_MSDA_KL, samplesInMemory, shotStart)

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

def adaptPrint(currentModel, unlabeledGestures, type_DA, trainFeatures, trainLabels, classes, labeledGesturesFeatures,
               labeledGesturesLabels, fewShotModel, results, idx, testFeatures, testLabels, samplesInMemory, shotStart,
               typeModel):
    name = type_DA + '_ACC_' + typeModel
    # print(name)

    postProb_trainFeatures = SemiSupervised.post_probabilities_Calculation(trainFeatures, currentModel, classes,
                                                                           type_DA)
    if type_DA == 'LDA':
        if typeModel == 'PostProb_MSDA':
            adaptedModel, results.at[
                idx, 'time_' + name], unlabeledGestures = SemiSupervised.model_PostProb_MSDA(
                currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewShotModel,
                labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
        elif typeModel == 'PostProb':
            adaptedModel, results.at[
                idx, 'time_' + name], unlabeledGestures = SemiSupervised.model_PostProb(
                currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewShotModel,
                labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
        elif typeModel == 'MSDA':
            adaptedModel, results.at[
                idx, 'time_' + name], unlabeledGestures = SemiSupervised.model_MSDA(
                currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewShotModel,
                labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
        elif typeModel == 'Baseline':
            adaptedModel, results.at[
                idx, 'time_' + name], unlabeledGestures = SemiSupervised.model_Baseline(
                currentModel, unlabeledGestures, classes, trainFeatures, trainLabels, postProb_trainFeatures,
                fewShotModel, labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
        elif typeModel == 'MSDA_KL':
            adaptedModel, results.at[
                idx, 'time_' + name], unlabeledGestures = SemiSupervised.model_MSDA_KL(
                currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewShotModel,
                labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
        elif typeModel == 'PostProb_MSDA_KL':
            adaptedModel, results.at[
                idx, 'time_' + name], unlabeledGestures = SemiSupervised.model_PostProb_MSDA_KL(
                currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewShotModel,
                labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)

        results.at[idx, name], _ = DA_Classifiers.accuracyModelLDA(testFeatures, testLabels, adaptedModel, classes)

    elif type_DA == 'QDA':
        if typeModel == 'PostProb_MSDA':
            adaptedModel, results.at[
                idx, 'time_' + name], unlabeledGestures = SemiSupervised.model_PostProb_MSDA(
                currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewShotModel,
                labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
        elif typeModel == 'PostProb':
            adaptedModel, results.at[
                idx, 'time_' + name], unlabeledGestures = SemiSupervised.model_PostProb(
                currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewShotModel,
                labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
        elif typeModel == 'MSDA':
            adaptedModel, results.at[
                idx, 'time_' + name], unlabeledGestures = SemiSupervised.model_MSDA(
                currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewShotModel,
                labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
        elif typeModel == 'Baseline':
            adaptedModel, results.at[
                idx, 'time_' + name], unlabeledGestures = SemiSupervised.model_Baseline(
                currentModel, unlabeledGestures, classes, trainFeatures, trainLabels, postProb_trainFeatures,
                fewShotModel, labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
        elif typeModel == 'MSDA_KL':
            adaptedModel, results.at[
                idx, 'time_' + name], unlabeledGestures = SemiSupervised.model_MSDA_KL(
                currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewShotModel,
                labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
        elif typeModel == 'PostProb_MSDA_KL':
            adaptedModel, results.at[
                idx, 'time_' + name], unlabeledGestures = SemiSupervised.model_PostProb_MSDA_KL(
                currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, fewShotModel,
                labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)

        results.at[idx, name], _ = DA_Classifiers.accuracyModelQDA(testFeatures, testLabels, adaptedModel, classes)

    return adaptedModel, results, unlabeledGestures


def resultsDataframeUnsupervised(
        trainFeatures, trainLabels, classes, results, testFeatures, testLabels, idx, person, nShots, rand, featureSet,
        nameFile, printR, labeledGesturesFeatures, labeledGesturesLabels, proposedModelLDA_PostProb_MSDA,
        proposedModelQDA_PostProb_MSDA, proposedModelLDA_PostProb, proposedModelQDA_PostProb,
        proposedModelLDA_MSDA, proposedModelQDA_MSDA, proposedModelLDA_Baseline, proposedModelQDA_Baseline,
        proposedModelLDA_MSDA_KL, proposedModelQDA_MSDA_KL, proposedModelLDA_PostProb_MSDA_KL,
        proposedModelQDA_PostProb_MSDA_KL, fewShotModel, unlabeledGesturesLDA_PostProb_MSDA,
        unlabeledGesturesQDA_PostProb_MSDA, unlabeledGesturesLDA_PostProb, unlabeledGesturesQDA_PostProb,
        unlabeledGesturesLDA_MSDA, unlabeledGesturesQDA_MSDA, unlabeledGesturesLDA_Baseline,
        unlabeledGesturesQDA_Baseline, unlabeledGesturesLDA_MSDA_KL, unlabeledGesturesQDA_MSDA_KL,
        unlabeledGesturesLDA_PostProb_MSDA_KL, unlabeledGesturesQDA_PostProb_MSDA_KL, samplesInMemory, shotStart):
    type_DA_set = ['LDA', 'QDA']
    for type_DA in type_DA_set:
        if type_DA == 'LDA':

            # print('AccLDAfew')
            results.at[idx, 'AccLDAfew'], _ = DA_Classifiers.accuracyModelLDA(
                testFeatures, testLabels, fewShotModel, classes)

            # proposedModelLDA_PostProb_MSDA, results, unlabeledGesturesLDA_PostProb_MSDA = adaptPrint(
            #     proposedModelLDA_PostProb_MSDA, unlabeledGesturesLDA_PostProb_MSDA, type_DA, trainFeatures, trainLabels,
            #     classes, labeledGesturesFeatures, labeledGesturesLabels, fewShotModel, results, idx, testFeatures,
            #     testLabels, samplesInMemory, shotStart, typeModel='PostProb_MSDA')

            proposedModelLDA_PostProb, results, unlabeledGesturesLDA_PostProb = adaptPrint(
                proposedModelLDA_PostProb, unlabeledGesturesLDA_PostProb, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, fewShotModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, typeModel='PostProb')
            # proposedModelLDA_MSDA, results, unlabeledGesturesLDA_MSDA = adaptPrint(
            #     proposedModelLDA_MSDA, unlabeledGesturesLDA_MSDA, type_DA, trainFeatures, trainLabels, classes,
            #     labeledGesturesFeatures, labeledGesturesLabels, fewShotModel, results, idx, testFeatures, testLabels,
            #     samplesInMemory, shotStart, typeModel='MSDA')
            proposedModelLDA_Baseline, results, unlabeledGesturesLDA_Baseline = adaptPrint(
                proposedModelLDA_Baseline, unlabeledGesturesLDA_Baseline, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, fewShotModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, typeModel='Baseline')
            # proposedModelLDA_MSDA_KL, results, unlabeledGesturesLDA_MSDA_KL = adaptPrint(
            #     proposedModelLDA_MSDA_KL, unlabeledGesturesLDA_MSDA_KL, type_DA, trainFeatures, trainLabels, classes,
            #     labeledGesturesFeatures, labeledGesturesLabels, fewShotModel, results, idx, testFeatures, testLabels,
            #     samplesInMemory, shotStart, typeModel='MSDA_KL')
            proposedModelLDA_PostProb_MSDA_KL, results, unlabeledGesturesLDA_PostProb_MSDA_KL = adaptPrint(
                proposedModelLDA_PostProb_MSDA_KL, unlabeledGesturesLDA_PostProb_MSDA_KL, type_DA, trainFeatures,
                trainLabels, classes, labeledGesturesFeatures, labeledGesturesLabels, fewShotModel, results, idx,
                testFeatures, testLabels, samplesInMemory, shotStart, typeModel='PostProb_MSDA_KL')

        elif type_DA == 'QDA':

            # print('AccQDAfew')
            results.at[idx, 'AccQDAfew'], _ = DA_Classifiers.accuracyModelQDA(
                testFeatures, testLabels, fewShotModel, classes)

            # proposedModelQDA_PostProb_MSDA, results, unlabeledGesturesQDA_PostProb_MSDA = adaptPrint(
            #     proposedModelQDA_PostProb_MSDA, unlabeledGesturesQDA_PostProb_MSDA, type_DA, trainFeatures, trainLabels,
            #     classes, labeledGesturesFeatures, labeledGesturesLabels, fewShotModel, results, idx, testFeatures,
            #     testLabels, samplesInMemory, shotStart, typeModel='PostProb_MSDA')
            proposedModelQDA_PostProb, results, unlabeledGesturesQDA_PostProb = adaptPrint(
                proposedModelQDA_PostProb, unlabeledGesturesQDA_PostProb, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, fewShotModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, typeModel='PostProb')
            # proposedModelQDA_MSDA, results, unlabeledGesturesQDA_MSDA = adaptPrint(
            #     proposedModelQDA_MSDA, unlabeledGesturesQDA_MSDA, type_DA, trainFeatures, trainLabels, classes,
            #     labeledGesturesFeatures, labeledGesturesLabels, fewShotModel, results, idx, testFeatures, testLabels,
            #     samplesInMemory, shotStart, typeModel='MSDA')
            proposedModelQDA_Baseline, results, unlabeledGesturesQDA_Baseline = adaptPrint(
                proposedModelQDA_Baseline, unlabeledGesturesQDA_Baseline, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, fewShotModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, typeModel='Baseline')
            # proposedModelQDA_MSDA_KL, results, unlabeledGesturesQDA_MSDA_KL = adaptPrint(
            #     proposedModelQDA_MSDA_KL, unlabeledGesturesQDA_MSDA_KL, type_DA, trainFeatures, trainLabels, classes,
            #     labeledGesturesFeatures, labeledGesturesLabels, fewShotModel, results, idx, testFeatures, testLabels,
            #     samplesInMemory, shotStart, typeModel='MSDA_KL')
            proposedModelQDA_PostProb_MSDA_KL, results, unlabeledGesturesQDA_PostProb_MSDA_KL = adaptPrint(
                proposedModelQDA_PostProb_MSDA_KL, unlabeledGesturesQDA_PostProb_MSDA_KL, type_DA, trainFeatures,
                trainLabels, classes, labeledGesturesFeatures, labeledGesturesLabels, fewShotModel, results, idx,
                testFeatures, testLabels, samplesInMemory, shotStart, typeModel='PostProb_MSDA_KL')

    results.at[idx, 'person'] = person
    results.at[idx, 'shot_class'] = rand
    results.at[idx, '# shots'] = nShots
    results.at[idx, 'Feature Set'] = featureSet

    if nameFile is not None:
        results.to_csv(nameFile)
    if printR:
        print(featureSet)
        print('Results: person= ', person, ' shot = ', nShots)
        print(results.loc[idx])

    idx += 1

    return results, idx, proposedModelLDA_PostProb_MSDA, proposedModelQDA_PostProb_MSDA, \
           proposedModelLDA_PostProb, proposedModelQDA_PostProb, proposedModelLDA_MSDA, \
           proposedModelQDA_MSDA, proposedModelLDA_Baseline, proposedModelQDA_Baseline, \
           proposedModelLDA_MSDA_KL, proposedModelQDA_MSDA_KL, proposedModelLDA_PostProb_MSDA_KL, \
           proposedModelQDA_PostProb_MSDA_KL, unlabeledGesturesLDA_PostProb_MSDA, \
           unlabeledGesturesQDA_PostProb_MSDA, unlabeledGesturesLDA_PostProb, \
           unlabeledGesturesQDA_PostProb, unlabeledGesturesLDA_MSDA, unlabeledGesturesQDA_MSDA, \
           unlabeledGesturesLDA_Baseline, unlabeledGesturesQDA_Baseline, unlabeledGesturesLDA_MSDA_KL, \
           unlabeledGesturesQDA_MSDA_KL, unlabeledGesturesLDA_PostProb_MSDA_KL, unlabeledGesturesQDA_PostProb_MSDA_KL


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
