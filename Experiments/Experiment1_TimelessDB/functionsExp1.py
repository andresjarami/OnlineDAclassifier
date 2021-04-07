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

    return dataMatrix, numberFeatures, CH, classes, peoplePriorK, peopleTest, numberShots, combinationSet, \
           allFeatures, labelsDataMatrix


def evaluation(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson, endPerson,
               allFeatures, printR, samplesInMemory, shotStart, initialTime, finalTime, typeDatabase):
    scaler = preprocessing.MinMaxScaler()

    results = pd.DataFrame(
        columns=['Feature Set', 'person', 'exp_time', '# shots', 'shot_class', 'precision_LDA_Ideal',
                 'precision_LDA_NoAdapted', 'precision_LDA_PostProb', 'precision_LDA_Labels',
                 'precision_LDA_PostProb_MSDA', 'precision_LDA_Adapted', 'precision_LDA_PostProb_Adapted',
                 'precision_LDA_Labels_Adapted', 'precision_LDA_PostProb_MSDA_Adapted', 'precision_QDA_Ideal',
                 'precision_QDA_NoAdapted', 'precision_QDA_PostProb', 'precision_QDA_Labels',
                 'precision_QDA_PostProb_MSDA', 'precision_QDA_Adapted', 'precision_QDA_PostProb_Adapted',
                 'precision_QDA_Labels_Adapted', 'precision_QDA_PostProb_MSDA_Adapted', 'recall_LDA_Ideal',
                 'recall_LDA_NoAdapted', 'recall_LDA_PostProb', 'recall_LDA_Labels', 'recall_LDA_PostProb_MSDA',
                 'recall_LDA_Adapted', 'recall_LDA_PostProb_Adapted', 'recall_LDA_Labels_Adapted',
                 'recall_LDA_PostProb_MSDA_Adapted', 'recall_QDA_Ideal', 'recall_QDA_NoAdapted', 'recall_QDA_PostProb',
                 'recall_QDA_Labels', 'recall_QDA_PostProb_MSDA', 'recall_QDA_Adapted', 'recall_QDA_PostProb_Adapted',
                 'recall_QDA_Labels_Adapted', 'recall_QDA_PostProb_MSDA_Adapted', 'AccLDA_Ideal', 'AccLDA_NoAdapted',
                 'AccLDA_PostProb', 'AccLDA_Labels', 'AccLDA_PostProb_MSDA', 'AccLDA_Adapted',
                 'AccLDA_PostProb_Adapted', 'AccLDA_Labels_Adapted', 'AccLDA_PostProb_MSDA_Adapted', 'AccQDA_Ideal',
                 'AccQDA_NoAdapted', 'AccQDA_PostProb', 'AccQDA_Labels', 'AccQDA_PostProb_MSDA', 'AccQDA_Adapted',
                 'AccQDA_PostProb_Adapted', 'AccQDA_Labels_Adapted', 'AccQDA_PostProb_MSDA_Adapted'])

    # results = pd.DataFrame(
    #     columns=['Feature Set', 'person', 'exp_time', '# shots', 'shot_class', 'precision_LDA_Ideal',
    #              'precision_LDA_NoAdapted', 'precision_LDA_PostProb', 'precision_LDA_Labels',
    #              'precision_LDA_PostProb_MSDA', 'precision_QDA_Ideal', 'precision_QDA_NoAdapted',
    #              'precision_QDA_PostProb', 'precision_QDA_Labels', 'precision_QDA_PostProb_MSDA', 'recall_LDA_Ideal',
    #              'recall_LDA_NoAdapted', 'recall_LDA_PostProb', 'recall_LDA_Labels', 'recall_LDA_PostProb_MSDA',
    #              'recall_QDA_Ideal', 'recall_QDA_NoAdapted', 'recall_QDA_PostProb', 'recall_QDA_Labels',
    #              'recall_QDA_PostProb_MSDA', 'AccLDA_Ideal', 'AccLDA_NoAdapted', 'AccLDA_PostProb', 'AccLDA_Labels',
    #              'AccLDA_PostProb_MSDA', 'AccQDA_Ideal', 'AccQDA_NoAdapted', 'AccQDA_PostProb', 'AccQDA_Labels',
    #              'AccQDA_PostProb_MSDA'])

    idx = 0
    unlabeledGestures = [[shot, cl] for shot in range(shotStart + 1, numberShots + 1) for cl in range(1, classes + 1)]
    for person in range(startPerson, endPerson + 1):
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

        if initialTime == -1:

            weakModel = currentDistributionValues(labeledGesturesFeatures, labeledGesturesLabels, classes, allFeatures,
                                                  shotStart)
            weakModel.to_pickle(
                'pretrainedModels/weakModel_' + typeDatabase + '_featureSet_' + str(featureSet) + '_person_' + str(
                    person) + '_shotStart_' + str(shotStart) + '.pkl')

            # adaptive model
            dataPK, _ = preprocessingPK(dataMatrix, allFeatures, scaler)
            preTrainedDataMatrix = PKModels(dataPK, classes, peoplePriorK, person, allFeatures)

            k = 1 - (np.log(shotStart) / np.log(numberShots + 1))
            step = 1

            adaptedModelLDA, _, _, _, _, _ = adaptive.OurModel(weakModel, preTrainedDataMatrix, classes, allFeatures,
                                                               labeledGesturesFeatures, labeledGesturesLabels,
                                                               step, 'LDA', k, shotStart)
            adaptedModelQDA, _, _, _, _, _ = adaptive.OurModel(weakModel, preTrainedDataMatrix, classes, allFeatures,
                                                               labeledGesturesFeatures, labeledGesturesLabels,
                                                               step, 'QDA', k, shotStart)
            adaptedModelLDA.to_pickle(
                'pretrainedModels/adaptedModelLDA_' + typeDatabase + '_featureSet_' + str(
                    featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '.pkl')
            adaptedModelQDA.to_pickle(
                'pretrainedModels/adaptedModelQDA_' + typeDatabase + '_featureSet_' + str(
                    featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '.pkl')

            results.at[idx, 'AccLDA_NoAdapted'], results.at[idx, 'precision_LDA_NoAdapted'], results.at[
                idx, 'recall_LDA_NoAdapted'], _ = DA_Classifiers.accuracyModelLDA(
                testFeatures, testLabels, weakModel, classes)

            results.at[idx, 'AccQDA_NoAdapted'], results.at[idx, 'precision_QDA_NoAdapted'], results.at[
                idx, 'recall_QDA_NoAdapted'], _ = DA_Classifiers.accuracyModelQDA(
                testFeatures, testLabels, weakModel, classes)

            results.at[idx, 'AccLDA_Adapted'], results.at[idx, 'precision_LDA_Adapted'], results.at[
                idx, 'recall_LDA_Adapted'], _ = DA_Classifiers.accuracyModelLDA(
                testFeatures, testLabels, adaptedModelLDA, classes)

            results.at[idx, 'AccQDA_Adapted'], results.at[idx, 'precision_QDA_Adapted'], results.at[
                idx, 'recall_QDA_Adapted'], _ = DA_Classifiers.accuracyModelQDA(
                testFeatures, testLabels, adaptedModelQDA, classes)

        else:
            adaptedModelLDA = pd.read_pickle(
                'pretrainedModels/adaptedModelLDA_' + typeDatabase + '_featureSet_' + str(
                    featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '.pkl')
            adaptedModelQDA = pd.read_pickle(
                'pretrainedModels/adaptedModelQDA_' + typeDatabase + '_featureSet_' + str(
                    featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '.pkl')

            weakModel = pd.read_pickle(
                'pretrainedModels/weakModel_' + typeDatabase + '_featureSet_' + str(featureSet) + '_person_' + str(
                    person) + '_shotStart_' + str(shotStart) + '.pkl')

            results.at[idx, 'AccLDA_NoAdapted'], results.at[idx, 'precision_LDA_NoAdapted'], results.at[
                idx, 'recall_LDA_NoAdapted'], _ = DA_Classifiers.accuracyModelLDA(
                testFeatures, testLabels, weakModel, classes)

            results.at[idx, 'AccQDA_NoAdapted'], results.at[idx, 'precision_QDA_NoAdapted'], results.at[
                idx, 'recall_QDA_NoAdapted'], _ = DA_Classifiers.accuracyModelQDA(
                testFeatures, testLabels, weakModel, classes)

            results.at[idx, 'AccLDA_Adapted'], results.at[idx, 'precision_LDA_Adapted'], results.at[
                idx, 'recall_LDA_Adapted'], _ = DA_Classifiers.accuracyModelLDA(
                testFeatures, testLabels, adaptedModelLDA, classes)

            results.at[idx, 'AccQDA_Adapted'], results.at[idx, 'precision_QDA_Adapted'], results.at[
                idx, 'recall_QDA_Adapted'], _ = DA_Classifiers.accuracyModelQDA(
                testFeatures, testLabels, adaptedModelQDA, classes)

        for randomExperiments in range(initialTime, finalTime + 1):
            nShots = 0
            datasetIdeal = np.hstack(
                (labeledGesturesFeatures, np.resize(labeledGesturesLabels, (len(labeledGesturesLabels), 1))))

            proposedModelLDA_PostProb_MSDA = weakModel.copy()
            proposedModelQDA_PostProb_MSDA = weakModel.copy()
            proposedModelLDA_PostProb = weakModel.copy()
            proposedModelQDA_PostProb = weakModel.copy()
            proposedModelLDA_Labels = weakModel.copy()
            proposedModelQDA_Labels = weakModel.copy()
            proposedModelLDA_PostProb_adap = adaptedModelLDA.copy()
            proposedModelQDA_PostProb_adap = adaptedModelQDA.copy()
            proposedModelLDA_PostProb_MSDA_adap = adaptedModelLDA.copy()
            proposedModelQDA_PostProb_MSDA_adap = adaptedModelQDA.copy()
            proposedModelLDA_Labels_adap = adaptedModelLDA.copy()
            proposedModelQDA_Labels_adap = adaptedModelLDA.copy()

            np.random.seed(randomExperiments)
            for rand in list(np.random.permutation(unlabeledGestures)):
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

                datasetIdeal = np.append(datasetIdeal,
                                         np.hstack((trainFeatures, np.resize(trainLabels, (len(trainLabels), 1)))),
                                         axis=0)
                idealModel = currentDistributionValues(datasetIdeal[:, :allFeatures], datasetIdeal[:, allFeatures],
                                                       classes, allFeatures, shotStart)
                results.at[idx, 'AccLDA_Ideal'], results.at[idx, 'precision_LDA_Ideal'], results.at[
                    idx, 'recall_LDA_Ideal'], _ = DA_Classifiers.accuracyModelLDA(
                    testFeatures, testLabels, idealModel, classes)

                results.at[idx, 'AccQDA_Ideal'], results.at[idx, 'precision_QDA_Ideal'], results.at[
                    idx, 'recall_QDA_Ideal'], _ = DA_Classifiers.accuracyModelQDA(
                    testFeatures, testLabels, idealModel, classes)

                # results.at[idx, 'AccLDA_Ideal'], _, _, _ = DA_Classifiers.accuracyModelLDA(
                #     testFeatures, testLabels, idealModel, classes)
                #
                # results.at[idx, 'AccQDA_Ideal'], _, _, _ = DA_Classifiers.accuracyModelQDA(
                #     testFeatures, testLabels, idealModel, classes)

                results, idx, proposedModelLDA_PostProb_MSDA, proposedModelQDA_PostProb_MSDA, \
                proposedModelLDA_PostProb_MSDA_adap, proposedModelQDA_PostProb_MSDA_adap, proposedModelLDA_PostProb, \
                proposedModelQDA_PostProb, proposedModelLDA_PostProb_adap, proposedModelQDA_PostProb_adap, \
                proposedModelLDA_Labels, proposedModelQDA_Labels, proposedModelLDA_Labels_adap, proposedModelQDA_Labels_adap = \
                    resultsDataframe(
                        trainFeatures, trainLabels, classes, results, testFeatures, testLabels, idx, person,
                        randomExperiments, nShots, rand, featureSet, nameFile, printR, labeledGesturesFeatures,
                        labeledGesturesLabels, proposedModelLDA_PostProb_MSDA, proposedModelQDA_PostProb_MSDA,
                        proposedModelLDA_PostProb, proposedModelQDA_PostProb, proposedModelLDA_Labels,
                        proposedModelQDA_Labels, weakModel, samplesInMemory, shotStart,
                        adaptedModelLDA, adaptedModelQDA, proposedModelLDA_PostProb_adap,
                        proposedModelQDA_PostProb_adap, proposedModelLDA_PostProb_MSDA_adap,
                        proposedModelQDA_PostProb_MSDA_adap, proposedModelLDA_Labels_adap, proposedModelQDA_Labels_adap)

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


def resultsDataframe(
        trainFeatures, trainLabels, classes, results, testFeatures, testLabels, idx, person, exp_time, nShots, rand,
        featureSet, nameFile, printR, labeledGesturesFeatures, labeledGesturesLabels, proposedModelLDA_PostProb_MSDA,
        proposedModelQDA_PostProb_MSDA, proposedModelLDA_PostProb, proposedModelQDA_PostProb, proposedModelLDA_Labels,
        proposedModelQDA_Labels, weakModel, samplesInMemory, shotStart, adaptedModelLDA, adaptedModelQDA,
        proposedModelLDA_PostProb_adap, proposedModelQDA_PostProb_adap, proposedModelLDA_PostProb_MSDA_adap,
        proposedModelQDA_PostProb_MSDA_adap, proposedModelLDA_Labels_adap, proposedModelQDA_Labels_adap):
    type_DA_set = ['LDA', 'QDA']
    for type_DA in type_DA_set:
        if type_DA == 'LDA':

            proposedModelLDA_PostProb_MSDA, results, = SemiSupervisedModels(
                proposedModelLDA_PostProb_MSDA, type_DA, trainFeatures, trainLabels,
                classes, labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures,
                testLabels, samplesInMemory, shotStart, nShots, typeModel='PostProb_MSDA')
            proposedModelLDA_Labels, results = SemiSupervisedModels(
                proposedModelLDA_Labels, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, typeModel='Labels')
            proposedModelLDA_PostProb, results = SemiSupervisedModels(
                proposedModelLDA_PostProb, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, typeModel='PostProb')
            # proposedModelLDA_MSDA, results, unlabeledGesturesLDA_MSDA = SemiSupervisedModels(
            #     proposedModelLDA_MSDA, unlabeledGesturesLDA_MSDA, type_DA, trainFeatures, trainLabels, classes,
            #     labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
            #     samplesInMemory, shotStart, typeModel='MSDA')
            # # proposedModelLDA_Baseline, results, unlabeledGesturesLDA_Baseline = SemiSupervisedModels(
            # #     proposedModelLDA_Baseline, unlabeledGesturesLDA_Baseline, type_DA, trainFeatures, trainLabels, classes,
            # #     labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
            # #     samplesInMemory, shotStart, typeModel='Baseline')
            # proposedModelLDA_MSDA_JS, results, unlabeledGesturesLDA_MSDA_JS = SemiSupervisedModels(
            #     proposedModelLDA_MSDA_JS, unlabeledGesturesLDA_MSDA_JS, type_DA, trainFeatures, trainLabels, classes,
            #     labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
            #     samplesInMemory, shotStart, typeModel='JS')
            # proposedModelLDA_PostProb_MSDA_JS, results, unlabeledGesturesLDA_PostProb_MSDA_JS = SemiSupervisedModels(
            #     proposedModelLDA_PostProb_MSDA_JS, unlabeledGesturesLDA_PostProb_MSDA_JS, type_DA, trainFeatures,
            #     trainLabels, classes, labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx,
            #     testFeatures, testLabels, samplesInMemory, shotStart, typeModel='PostProb-JS')

            # ## WITH ADAPTATION
            proposedModelLDA_PostProb_MSDA_adap, results = SemiSupervisedModels(
                proposedModelLDA_PostProb_MSDA_adap, type_DA, trainFeatures,
                trainLabels,
                classes, labeledGesturesFeatures, labeledGesturesLabels, adaptedModelLDA, results, idx, testFeatures,
                testLabels, samplesInMemory, shotStart, nShots, typeModel='PostProb_MSDA', adaptation='_Adapted')
            proposedModelLDA_Labels_adap, results = SemiSupervisedModels(
                proposedModelLDA_Labels_adap, type_DA, trainFeatures, trainLabels,
                classes, labeledGesturesFeatures, labeledGesturesLabels, adaptedModelLDA, results, idx, testFeatures,
                testLabels, samplesInMemory, shotStart, nShots, typeModel='Labels', adaptation='_Adapted')
            proposedModelLDA_PostProb_adap, results = SemiSupervisedModels(
                proposedModelLDA_PostProb_adap, type_DA, trainFeatures, trainLabels,
                classes, labeledGesturesFeatures, labeledGesturesLabels, adaptedModelLDA, results, idx, testFeatures,
                testLabels, samplesInMemory, shotStart, nShots, typeModel='PostProb', adaptation='_Adapted')
            # # proposedModelLDA_Baseline_adap, results, unlabeledGesturesLDA_Baseline_adap = SemiSupervisedModels(
            # #     proposedModelLDA_Baseline_adap, unlabeledGesturesLDA_Baseline_adap, type_DA, trainFeatures, trainLabels,
            # #     classes, labeledGesturesFeatures, labeledGesturesLabels, adaptedModelLDA, results, idx, testFeatures,
            # #     testLabels, samplesInMemory, shotStart, typeModel='Baseline', adaptation='_Adapted')
            # proposedModelLDA_PostProb_MSDA_JS_adap, results, unlabeledGesturesLDA_PostProb_MSDA_JS_adap = SemiSupervisedModels(
            #     proposedModelLDA_PostProb_MSDA_JS_adap, unlabeledGesturesLDA_PostProb_MSDA_JS_adap, type_DA,
            #     trainFeatures, trainLabels, classes, labeledGesturesFeatures, labeledGesturesLabels, adaptedModelLDA,
            #     results, idx, testFeatures, testLabels, samplesInMemory, shotStart, typeModel='PostProb-JS',
            #     adaptation='_Adapted')

        elif type_DA == 'QDA':

            proposedModelQDA_PostProb_MSDA, results = SemiSupervisedModels(
                proposedModelQDA_PostProb_MSDA, type_DA, trainFeatures, trainLabels,
                classes, labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures,
                testLabels, samplesInMemory, shotStart, nShots, typeModel='PostProb_MSDA')
            proposedModelQDA_Labels, results = SemiSupervisedModels(
                proposedModelQDA_Labels, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, typeModel='Labels')
            proposedModelQDA_PostProb, results = SemiSupervisedModels(
                proposedModelQDA_PostProb, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, typeModel='PostProb')
            # proposedModelQDA_MSDA, results, unlabeledGesturesQDA_MSDA = SemiSupervisedModels(
            #     proposedModelQDA_MSDA, unlabeledGesturesQDA_MSDA, type_DA, trainFeatures, trainLabels, classes,
            #     labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
            #     samplesInMemory, shotStart, typeModel='MSDA')
            # # proposedModelQDA_Baseline, results, unlabeledGesturesQDA_Baseline = SemiSupervisedModels(
            # #     proposedModelQDA_Baseline, unlabeledGesturesQDA_Baseline, type_DA, trainFeatures, trainLabels, classes,
            # #     labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
            # #     samplesInMemory, shotStart, typeModel='Baseline')
            # proposedModelQDA_MSDA_JS, results, unlabeledGesturesQDA_MSDA_JS = SemiSupervisedModels(
            #     proposedModelQDA_MSDA_JS, unlabeledGesturesQDA_MSDA_JS, type_DA, trainFeatures, trainLabels, classes,
            #     labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
            #     samplesInMemory, shotStart, typeModel='JS')
            # proposedModelQDA_PostProb_MSDA_JS, results, unlabeledGesturesQDA_PostProb_MSDA_JS = SemiSupervisedModels(
            #     proposedModelQDA_PostProb_MSDA_JS, unlabeledGesturesQDA_PostProb_MSDA_JS, type_DA, trainFeatures,
            #     trainLabels, classes, labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx,
            #     testFeatures, testLabels, samplesInMemory, shotStart, typeModel='PostProb-JS')

            ## WITH ADAPTATION
            proposedModelQDA_PostProb_MSDA_adap, results = SemiSupervisedModels(
                proposedModelQDA_PostProb_MSDA_adap, type_DA, trainFeatures,
                trainLabels,
                classes, labeledGesturesFeatures, labeledGesturesLabels, adaptedModelQDA, results, idx, testFeatures,
                testLabels, samplesInMemory, shotStart, nShots, typeModel='PostProb_MSDA', adaptation='_Adapted')
            proposedModelQDA_Labels_adap, results = SemiSupervisedModels(
                proposedModelQDA_Labels_adap, type_DA, trainFeatures, trainLabels,
                classes, labeledGesturesFeatures, labeledGesturesLabels, adaptedModelQDA, results, idx, testFeatures,
                testLabels, samplesInMemory, shotStart, nShots, typeModel='Labels', adaptation='_Adapted')
            proposedModelQDA_PostProb_adap, results = SemiSupervisedModels(
                proposedModelQDA_PostProb_adap, type_DA, trainFeatures, trainLabels,
                classes, labeledGesturesFeatures, labeledGesturesLabels, adaptedModelQDA, results, idx, testFeatures,
                testLabels, samplesInMemory, shotStart, nShots, typeModel='PostProb', adaptation='_Adapted')
            # # proposedModelQDA_Baseline_adap, results, unlabeledGesturesQDA_Baseline_adap = SemiSupervisedModels(
            # #     proposedModelQDA_Baseline_adap, unlabeledGesturesQDA_Baseline_adap, type_DA, trainFeatures, trainLabels,
            # #     classes, labeledGesturesFeatures, labeledGesturesLabels, adaptedModelQDA, results, idx, testFeatures,
            # #     testLabels, samplesInMemory, shotStart, typeModel='Baseline', adaptation='_Adapted')
            # proposedModelQDA_PostProb_MSDA_JS_adap, results, unlabeledGesturesQDA_PostProb_MSDA_JS_adap = SemiSupervisedModels(
            #     proposedModelQDA_PostProb_MSDA_JS_adap, unlabeledGesturesQDA_PostProb_MSDA_JS_adap, type_DA,
            #     trainFeatures, trainLabels, classes, labeledGesturesFeatures, labeledGesturesLabels, adaptedModelQDA,
            #     results, idx, testFeatures, testLabels, samplesInMemory, shotStart, typeModel='PostProb-JS',
            #     adaptation='_Adapted')

    results.at[idx, 'Feature Set'] = featureSet
    results.at[idx, 'person'] = person
    results.at[idx, 'exp_time'] = exp_time
    results.at[idx, 'shot_class'] = rand
    results.at[idx, '# shots'] = nShots

    if nameFile is not None:
        results.to_csv(nameFile)
    if printR:
        print(featureSet)
        print('Results: person= ', person, ' shot = ', nShots)
        print(results.loc[idx])

    idx += 1

    return results, idx, proposedModelLDA_PostProb_MSDA, proposedModelQDA_PostProb_MSDA, \
           proposedModelLDA_PostProb_MSDA_adap, proposedModelQDA_PostProb_MSDA_adap, proposedModelLDA_PostProb, \
           proposedModelQDA_PostProb, proposedModelLDA_PostProb_adap, proposedModelQDA_PostProb_adap, \
           proposedModelLDA_Labels, proposedModelQDA_Labels, proposedModelLDA_Labels_adap, proposedModelQDA_Labels_adap


def SemiSupervisedModels(currentModel, type_DA, trainFeatures, trainLabels, classes, labeledGesturesFeatures,
                         labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels, samplesInMemory,
                         shotStart, nShots, typeModel, adaptation=''):
    name = 'Acc' + type_DA + '_' + typeModel + adaptation

    postProb_trainFeatures = SemiSupervised.post_probabilities_Calculation(trainFeatures, currentModel, classes,
                                                                           type_DA)
    if typeModel == 'PostProb_MSDA':
        updatedModel, results.at[
            idx, 'time_' + name], predictedWeight = SemiSupervised.model_MSDAlabels(
            currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
            labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
        # updatedModel, _, predictedWeight = SemiSupervised.model_MSDAlabels(
        #     currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
        #     labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
    elif typeModel == 'Labels':
        updatedModel, results.at[idx, 'time_' + name], predictedWeight = SemiSupervised.model_Labels(
            currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
            labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
        # updatedModel, _, predictedWeight = SemiSupervised.model_Labels(
        #     currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
        #     labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
    elif typeModel == 'PostProb':
        updatedModel, results.at[idx, 'time_' + name], predictedWeight = SemiSupervised.model_PostProb(
            currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
            labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
        # updatedModel, _, predictedWeight = SemiSupervised.model_PostProb(
        #     currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
        #     labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
    # elif typeModel == 'MSDA':
    #     typeModel = 'PostProb_MSDA_2w_sum_Nonorm'
    #     name = 'Acc' + type_DA + '_' + typeModel + adaptation
    #     updatedModel, _, unlabeledGestures = SemiSupervised.model_PostProb_MSDA_2w_sum_Nonorm(
    #         currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, weakModel,
    #         labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
    # elif typeModel == 'Baseline':
    #     updatedModel, results.at[
    #         idx, 'time_' + name], unlabeledGestures = SemiSupervised.model_Baseline(
    #         currentModel, unlabeledGestures, classes, trainFeatures, trainLabels, postProb_trainFeatures,
    #         weakModel, labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
    # elif typeModel == 'JS':
    #     typeModel = 'PostProb_MSDA_1w_sum_Nonorm'
    #     name = 'Acc' + type_DA + '_' + typeModel + adaptation
    #     updatedModel, _, unlabeledGestures = SemiSupervised.model_PostProb_MSDA_1w_sum_Nonorm(
    #         currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, weakModel,
    #         labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
    # elif typeModel == 'PostProb-JS':
    #     updatedModel, _, unlabeledGestures = SemiSupervised.model_PostProb_MSDA_JS(
    #         currentModel, unlabeledGestures, classes, trainFeatures, postProb_trainFeatures, weakModel,
    #         labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)

    if type_DA == 'LDA':
        results.at[idx, name], results.at[idx, 'precision_' + type_DA + '_' + typeModel + adaptation], results.at[
            idx, 'recall_' + type_DA + '_' + typeModel + adaptation], _ = DA_Classifiers.accuracyModelLDA(
            testFeatures, testLabels, updatedModel, classes)

    elif type_DA == 'QDA':
        results.at[idx, name], results.at[idx, 'precision_' + type_DA + '_' + typeModel + adaptation], results.at[
            idx, 'recall_' + type_DA + '_' + typeModel + adaptation], _ = DA_Classifiers.accuracyModelQDA(
            testFeatures, testLabels, updatedModel, classes)

    # if type_DA == 'LDA':
    #     results.at[idx, name], _, _, _ = DA_Classifiers.accuracyModelLDA(
    #         testFeatures, testLabels, updatedModel, classes)
    # elif type_DA == 'QDA':
    #     results.at[idx, name], _, _, _ = DA_Classifiers.accuracyModelQDA(
    #         testFeatures, testLabels, updatedModel, classes)

    observedWeight = np.zeros(classes)
    observedWeight[int(trainLabels[0] - 1)] = 1
    name = 'Error' + type_DA + '_' + typeModel + adaptation
    if nShots != 1:
        results.at[idx, 'MAE_' + name] = (results.at[idx - 1, 'MAE_' + name] * (
                nShots - 1) + SemiSupervised.MAE(predictedWeight, observedWeight)) / nShots
        results.at[idx, '1_' + name] = (results.at[idx - 1, '1_' + name] * (
                nShots - 1) + SemiSupervised.errorWeights_type1(predictedWeight, observedWeight)) / nShots
        results.at[idx, '2_' + name] = (results.at[idx - 1, '2_' + name] * (
                nShots - 1) + SemiSupervised.errorWeights_type2(predictedWeight, observedWeight)) / nShots

    else:
        results.at[idx, 'MAE_' + name] = SemiSupervised.MAE(predictedWeight, observedWeight)
        results.at[idx, '1_' + name] = SemiSupervised.errorWeights_type1(predictedWeight, observedWeight)
        results.at[idx, '2_' + name] = SemiSupervised.errorWeights_type2(predictedWeight, observedWeight)

    return updatedModel, results


# Auxiliar functions of the evaluation


def currentDistributionValues(trainFeatures, trainLabels, classes, allFeatures, shotStart):
    currentValues = pd.DataFrame(columns=['cov', 'mean', 'class', 'weight_mean', 'weight_cov', '# gestures'])
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
    currentValues.at[0, '# gestures'] = shotStart
    return currentValues


def preprocessingPK(dataMatrix, allFeatures, scaler):
    dataMatrixFeatures = scaler.transform(dataMatrix[:, :allFeatures])
    return np.hstack((dataMatrixFeatures, dataMatrix[:, allFeatures:])), np.size(dataMatrixFeatures, axis=1)
