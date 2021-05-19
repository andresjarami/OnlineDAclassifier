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

    # results = pd.DataFrame(
    #     columns=['Feature Set', 'person', 'exp_time', '# shots', 'shot_class', 'precision_LDA_Ideal',
    #              'precision_LDA_NoAdapted', 'precision_LDA_PostProb', 'precision_LDA_Labels',
    #              'precision_LDA_PostProb_MSDA', 'precision_LDA_Adapted', 'precision_LDA_PostProb_Adapted',
    #              'precision_LDA_Labels_Adapted', 'precision_LDA_PostProb_MSDA_Adapted', 'precision_QDA_Ideal',
    #              'precision_QDA_NoAdapted', 'precision_QDA_PostProb', 'precision_QDA_Labels',
    #              'precision_QDA_PostProb_MSDA', 'precision_QDA_Adapted', 'precision_QDA_PostProb_Adapted',
    #              'precision_QDA_Labels_Adapted', 'precision_QDA_PostProb_MSDA_Adapted', 'recall_LDA_Ideal',
    #              'recall_LDA_NoAdapted', 'recall_LDA_PostProb', 'recall_LDA_Labels', 'recall_LDA_PostProb_MSDA',
    #              'recall_LDA_Adapted', 'recall_LDA_PostProb_Adapted', 'recall_LDA_Labels_Adapted',
    #              'recall_LDA_PostProb_MSDA_Adapted', 'recall_QDA_Ideal', 'recall_QDA_NoAdapted', 'recall_QDA_PostProb',
    #              'recall_QDA_Labels', 'recall_QDA_PostProb_MSDA', 'recall_QDA_Adapted', 'recall_QDA_PostProb_Adapted',
    #              'recall_QDA_Labels_Adapted', 'recall_QDA_PostProb_MSDA_Adapted', 'AccLDA_Ideal', 'AccLDA_NoAdapted',
    #              'AccLDA_PostProb', 'AccLDA_Labels', 'AccLDA_PostProb_MSDA', 'AccLDA_Adapted',
    #              'AccLDA_PostProb_Adapted', 'AccLDA_Labels_Adapted', 'AccLDA_PostProb_MSDA_Adapted', 'AccQDA_Ideal',
    #              'AccQDA_NoAdapted', 'AccQDA_PostProb', 'AccQDA_Labels', 'AccQDA_PostProb_MSDA', 'AccQDA_Adapted',
    #              'AccQDA_PostProb_Adapted', 'AccQDA_Labels_Adapted', 'AccQDA_PostProb_MSDA_Adapted'])

    results = pd.DataFrame(
        columns=['Feature Set', 'person', 'exp_time', '# shots', 'shot_class', 'precision_LDA_Ideal',
                 'precision_LDA_NoAdapted', 'precision_LDA_PostProb', 'precision_LDA_Labels',
                 'precision_LDA_PostProb_MSDA', 'precision_QDA_Ideal', 'precision_QDA_NoAdapted',
                 'precision_QDA_PostProb', 'precision_QDA_Labels', 'precision_QDA_PostProb_MSDA', 'recall_LDA_Ideal',
                 'recall_LDA_NoAdapted', 'recall_LDA_PostProb', 'recall_LDA_Labels', 'recall_LDA_PostProb_MSDA',
                 'recall_QDA_Ideal', 'recall_QDA_NoAdapted', 'recall_QDA_PostProb', 'recall_QDA_Labels',
                 'recall_QDA_PostProb_MSDA', 'AccLDA_Ideal', 'AccLDA_NoAdapted', 'AccLDA_PostProb', 'AccLDA_Labels',
                 'AccLDA_PostProb_MSDA', 'AccQDA_Ideal', 'AccQDA_NoAdapted', 'AccQDA_PostProb', 'AccQDA_Labels',
                 'AccQDA_PostProb_MSDA'])

    idx = 0
    unlabeledGestures = [[shot, cla] for shot in range(shotStart + 1, numberShots + 1) for cla in range(1, classes + 1)]
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

            # results.at[idx, 'AccLDA_Adapted'], results.at[idx, 'precision_LDA_Adapted'], results.at[
            #     idx, 'recall_LDA_Adapted'], _ = DA_Classifiers.accuracyModelLDA(
            #     testFeatures, testLabels, adaptedModelLDA, classes)
            #
            # results.at[idx, 'AccQDA_Adapted'], results.at[idx, 'precision_QDA_Adapted'], results.at[
            #     idx, 'recall_QDA_Adapted'], _ = DA_Classifiers.accuracyModelQDA(
            #     testFeatures, testLabels, adaptedModelQDA, classes)

        for seed in range(initialTime, finalTime + 1):
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

            np.random.seed(seed)
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
                        seed, nShots, rand, featureSet, nameFile, printR, labeledGesturesFeatures,
                        labeledGesturesLabels, proposedModelLDA_PostProb_MSDA, proposedModelQDA_PostProb_MSDA,
                        proposedModelLDA_PostProb, proposedModelQDA_PostProb, proposedModelLDA_Labels,
                        proposedModelQDA_Labels, weakModel, samplesInMemory, shotStart,
                        adaptedModelLDA, adaptedModelQDA, proposedModelLDA_PostProb_adap,
                        proposedModelQDA_PostProb_adap, proposedModelLDA_PostProb_MSDA_adap,
                        proposedModelQDA_PostProb_MSDA_adap, proposedModelLDA_Labels_adap, proposedModelQDA_Labels_adap)

    return results


def evaluation2(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson, endPerson,
                allFeatures, printR, samplesInMemory, shotStart, initialTime, finalTime, typeDatabase):
    scaler = preprocessing.MinMaxScaler()
    # print(numberShots * classes - shotStart * classes)
    if typeDatabase == 'EPN':
        evaluatedGestures = np.array([1, 5, 10, 20, 40, 60, 80, 100, 120, numberShots * classes - shotStart * classes])
    elif typeDatabase == 'Nina5':
        evaluatedGestures = np.array([1, 5, 10, 20, 30, 40, 50, 54, numberShots * classes - shotStart * classes])
    elif typeDatabase == 'Cote':
        evaluatedGestures = np.array([1, 6, 11, 16, 21, numberShots * classes - shotStart * classes])

    results = pd.DataFrame(
        columns=['Feature Set', 'person', 'exp_time', '# shots', 'shot_class', 'unlabeled Gesture'])

    for metric in ['precision', 'recall', 'w_predicted', 'time']:
        for typeDA in ['LDA', 'QDA']:
            if metric == 'precision' or metric == 'recall':
                results[metric + '_' + typeDA + '_' + 'weak'] = ''
                results[metric + '_' + typeDA + '_' + 'adapted'] = ''
            results[metric + '_' + typeDA + '_' + 'incre_proposedMcc_adapt'] = ''
            results[metric + '_' + typeDA + '_' + 'incre_proposedMcc'] = ''
            results[metric + '_' + typeDA + '_' + 'incre_proposedLabel'] = ''
            results[metric + '_' + typeDA + '_' + 'incre_sequential'] = ''
            results[metric + '_' + typeDA + '_' + 'incre_supervised'] = ''
            for l in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
                results[metric + '_' + typeDA + '_' + 'incre_Nigam_' + str(l)] = ''

    # results = pd.DataFrame(
    #     columns=['Feature Set', 'person', 'exp_time', '# shots', 'shot_class', 'precision_LDA_Ideal',
    #              'precision_LDA_NoAdapted', 'precision_LDA_incre_gestures_labels',
    #              'precision_LDA_incre_gestures_weight', 'precision_LDA_incre_gestures_weight_MSDA',
    #              'precision_LDA_incre_samples_labels', 'precision_LDA_incre_samples_prob',
    #              'precision_LDA_semi_gestures_labels', 'precision_LDA_semi_gestures_weight',
    #              'precision_LDA_semi_gestures_weight_MSDA', 'precision_LDA_semi_samples_labels',
    #              'precision_LDA_semi_samples_prob', 'precision_LDA_Adapted',
    #              'precision_LDA_incre_gestures_labels_Adapted', 'precision_LDA_incre_gestures_weight_Adapted',
    #              'precision_LDA_incre_gestures_weight_MSDA_Adapted', 'precision_LDA_incre_samples_labels_Adapted',
    #              'precision_LDA_incre_samples_prob_Adapted', 'precision_LDA_semi_gestures_labels_Adapted',
    #              'precision_LDA_semi_gestures_weight_Adapted', 'precision_LDA_semi_gestures_weight_MSDA_Adapted',
    #              'precision_LDA_semi_samples_labels_Adapted', 'precision_LDA_semi_samples_prob_Adapted',
    #              'precision_QDA_Ideal', 'precision_QDA_NoAdapted', 'precision_QDA_incre_gestures_labels',
    #              'precision_QDA_incre_gestures_weight', 'precision_QDA_incre_gestures_weight_MSDA',
    #              'precision_QDA_incre_samples_labels', 'precision_QDA_incre_samples_prob',
    #              'precision_QDA_semi_gestures_labels', 'precision_QDA_semi_gestures_weight',
    #              'precision_QDA_semi_gestures_weight_MSDA', 'precision_QDA_semi_samples_labels',
    #              'precision_QDA_semi_samples_prob', 'precision_QDA_Adapted',
    #              'precision_QDA_incre_gestures_labels_Adapted', 'precision_QDA_incre_gestures_weight_Adapted',
    #              'precision_QDA_incre_gestures_weight_MSDA_Adapted', 'precision_QDA_incre_samples_labels_Adapted',
    #              'precision_QDA_incre_samples_prob_Adapted', 'precision_QDA_semi_gestures_labels_Adapted',
    #              'precision_QDA_semi_gestures_weight_Adapted', 'precision_QDA_semi_gestures_weight_MSDA_Adapted',
    #              'precision_QDA_semi_samples_labels_Adapted', 'precision_QDA_semi_samples_prob_Adapted',
    #              'recall_LDA_Ideal', 'recall_LDA_NoAdapted', 'recall_LDA_incre_gestures_labels',
    #              'recall_LDA_incre_gestures_weight', 'recall_LDA_incre_gestures_weight_MSDA',
    #              'recall_LDA_incre_samples_labels', 'recall_LDA_incre_samples_prob', 'recall_LDA_semi_gestures_labels',
    #              'recall_LDA_semi_gestures_weight', 'recall_LDA_semi_gestures_weight_MSDA',
    #              'recall_LDA_semi_samples_labels', 'recall_LDA_semi_samples_prob', 'recall_LDA_Adapted',
    #              'recall_LDA_incre_gestures_labels_Adapted', 'recall_LDA_incre_gestures_weight_Adapted',
    #              'recall_LDA_incre_gestures_weight_MSDA_Adapted', 'recall_LDA_incre_samples_labels_Adapted',
    #              'recall_LDA_incre_samples_prob_Adapted', 'recall_LDA_semi_gestures_labels_Adapted',
    #              'recall_LDA_semi_gestures_weight_Adapted', 'recall_LDA_semi_gestures_weight_MSDA_Adapted',
    #              'recall_LDA_semi_samples_labels_Adapted', 'recall_LDA_semi_samples_prob_Adapted', 'recall_QDA_Ideal',
    #              'recall_QDA_NoAdapted', 'recall_QDA_incre_gestures_labels', 'recall_QDA_incre_gestures_weight',
    #              'recall_QDA_incre_gestures_weight_MSDA', 'recall_QDA_incre_samples_labels',
    #              'recall_QDA_incre_samples_prob', 'recall_QDA_semi_gestures_labels', 'recall_QDA_semi_gestures_weight',
    #              'recall_QDA_semi_gestures_weight_MSDA', 'recall_QDA_semi_samples_labels',
    #              'recall_QDA_semi_samples_prob', 'recall_QDA_Adapted', 'recall_QDA_incre_gestures_labels_Adapted',
    #              'recall_QDA_incre_gestures_weight_Adapted', 'recall_QDA_incre_gestures_weight_MSDA_Adapted',
    #              'recall_QDA_incre_samples_labels_Adapted', 'recall_QDA_incre_samples_prob_Adapted',
    #              'recall_QDA_semi_gestures_labels_Adapted', 'recall_QDA_semi_gestures_weight_Adapted',
    #              'recall_QDA_semi_gestures_weight_MSDA_Adapted', 'recall_QDA_semi_samples_labels_Adapted',
    #              'recall_QDA_semi_samples_prob_Adapted', 'AccLDA_Ideal', 'AccLDA_NoAdapted',
    #              'AccLDA_incre_gestures_labels', 'AccLDA_incre_gestures_weight', 'AccLDA_incre_gestures_weight_MSDA',
    #              'AccLDA_incre_samples_labels', 'AccLDA_incre_samples_prob', 'AccLDA_semi_gestures_labels',
    #              'AccLDA_semi_gestures_weight', 'AccLDA_semi_gestures_weight_MSDA', 'AccLDA_semi_samples_labels',
    #              'AccLDA_semi_samples_prob', 'AccLDA_Adapted', 'AccLDA_incre_gestures_labels_Adapted',
    #              'AccLDA_incre_gestures_weight_Adapted', 'AccLDA_incre_gestures_weight_MSDA_Adapted',
    #              'AccLDA_incre_samples_labels_Adapted', 'AccLDA_incre_samples_prob_Adapted',
    #              'AccLDA_semi_gestures_labels_Adapted', 'AccLDA_semi_gestures_weight_Adapted',
    #              'AccLDA_semi_gestures_weight_MSDA_Adapted', 'AccLDA_semi_samples_labels_Adapted',
    #              'AccLDA_semi_samples_prob_Adapted', 'AccQDA_Ideal', 'AccQDA_NoAdapted', 'AccQDA_incre_gestures_labels',
    #              'AccQDA_incre_gestures_weight', 'AccQDA_incre_gestures_weight_MSDA', 'AccQDA_incre_samples_labels',
    #              'AccQDA_incre_samples_prob', 'AccQDA_semi_gestures_labels', 'AccQDA_semi_gestures_weight',
    #              'AccQDA_semi_gestures_weight_MSDA', 'AccQDA_semi_samples_labels', 'AccQDA_semi_samples_prob',
    #              'AccQDA_Adapted', 'AccQDA_incre_gestures_labels_Adapted', 'AccQDA_incre_gestures_weight_Adapted',
    #              'AccQDA_incre_gestures_weight_MSDA_Adapted', 'AccQDA_incre_samples_labels_Adapted',
    #              'AccQDA_incre_samples_prob_Adapted', 'AccQDA_semi_gestures_labels_Adapted',
    #              'AccQDA_semi_gestures_weight_Adapted', 'AccQDA_semi_gestures_weight_MSDA_Adapted',
    #              'AccQDA_semi_samples_labels_Adapted', 'AccQDA_semi_samples_prob_Adapted'])

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

    for person in range(startPerson, endPerson + 1):

        for seed in range(initialTime, finalTime + 1):
            testFeatures = \
                dataMatrix[(dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 1), :allFeatures]
            testLabels = dataMatrix[
                (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 1), allFeatures + 2].T

            np.random.seed(seed)
            unlabeledGestures = []
            labeledGestures = []
            for cla in range(1, classes + 1):
                repetitions = np.random.choice(numberShots, numberShots, replace=False) + 1
                unlabeledGestures += [[shot, cla] for shot in repetitions[shotStart:]]
                labeledGestures += [[shot, cla] for shot in repetitions[:shotStart]]
            # print(labeledGestures)
            # print(unlabeledGestures)
            permutationUnlabeledGestures = np.random.permutation(unlabeledGestures)
            # print(permutationUnlabeledGestures)

            labeledGesturesFeatures = []
            labeledGesturesLabels = []
            numberLabeledGestureList = []
            probLabeledGestures = []
            numberLabeledGestures = 1
            for Lgesture in labeledGestures:
                labeledGesturesFeatures += list(dataMatrix[
                                                (dataMatrix[:, allFeatures + 1] == person) &
                                                (dataMatrix[:, allFeatures] == 0) &
                                                (dataMatrix[:, allFeatures + 3] == Lgesture[0]) &
                                                (dataMatrix[:, allFeatures + 2] == Lgesture[1]), 0:allFeatures])
                labeledGesturesLabels += list(dataMatrix[
                                                  (dataMatrix[:, allFeatures + 1] == person) &
                                                  (dataMatrix[:, allFeatures] == 0) &
                                                  (dataMatrix[:, allFeatures + 3] == Lgesture[0]) &
                                                  (dataMatrix[:, allFeatures + 2] == Lgesture[1]), allFeatures + 2].T)
                numberSegments = len(dataMatrix[(dataMatrix[:, allFeatures + 1] == person) &
                                                (dataMatrix[:, allFeatures] == 0) &
                                                (dataMatrix[:, allFeatures + 3] == Lgesture[0]) &
                                                (dataMatrix[:, allFeatures + 2] == Lgesture[
                                                    1]), allFeatures + 2].T)
                numberLabeledGestureList += list(np.ones(numberSegments) * numberLabeledGestures)
                auxProb = np.zeros((numberSegments, classes))
                auxProb[:, Lgesture[1] - 1] = np.ones(numberSegments)
                probLabeledGestures += list(auxProb)
                numberLabeledGestures += 1

            labeledGesturesFeatures = np.array(labeledGesturesFeatures)
            labeledGesturesLabels = np.array(labeledGesturesLabels)
            # numberLabeledGestureList = np.array(numberLabeledGestureList)
            # probLabeledGestures = np.array(probLabeledGestures)

            labeledGesturesFeatures = scaler.fit_transform(labeledGesturesFeatures)
            testFeatures = scaler.transform(testFeatures)

            # datasetIdeal = np.hstack(
            #     (labeledGesturesFeatures, np.resize(labeledGesturesLabels, (len(labeledGesturesLabels), 1))))

            # dataTotal = np.hstack(
            #     (labeledGesturesFeatures, np.resize(labeledGesturesLabels, (len(labeledGesturesLabels), 1)),
            #      np.resize(numberLabeledGestureList, (len(labeledGesturesLabels), 1)),
            #      np.resize(probLabeledGestures, (len(labeledGesturesLabels), classes))))

            try:
                adaptedModelLDA = pd.read_pickle(
                    'pretrainedModels/adaptedModelLDA_' + typeDatabase + '_featureSet_' + str(
                        featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
                        seed) + '.pkl')
                adaptedModelQDA = pd.read_pickle(
                    'pretrainedModels/adaptedModelQDA_' + typeDatabase + '_featureSet_' + str(
                        featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
                        seed) + '.pkl')

                weakModel = pd.read_pickle(
                    'pretrainedModels/weakModel_' + typeDatabase + '_featureSet_' + str(featureSet) + '_person_' + str(
                        person) + '_shotStart_' + str(shotStart) + '_seed_' + str(seed) + '.pkl')

                print('Found the models')
            except:
                print('Did not find the models')

                weakModel = currentDistributionValues(
                    labeledGesturesFeatures, labeledGesturesLabels, classes, allFeatures, shotStart)

                weakModel.to_pickle(
                    'pretrainedModels/weakModel_' + typeDatabase + '_featureSet_' + str(featureSet) + '_person_' + str(
                        person) + '_shotStart_' + str(shotStart) + '_seed_' + str(seed) + '.pkl')

                # adaptive model
                dataPK, _ = preprocessingPK(dataMatrix, allFeatures, scaler)
                preTrainedDataMatrix = PKModels(dataPK, classes, peoplePriorK, person, allFeatures)

                k = 1 - (np.log(shotStart) / np.log(numberShots + 1))
                step = 1

                adaptedModelLDA, _, _, _, _, _ = adaptive.OurModel(
                    weakModel, preTrainedDataMatrix, classes, allFeatures, labeledGesturesFeatures,
                    labeledGesturesLabels, step, 'LDA', k, shotStart)
                adaptedModelQDA, _, _, _, _, _ = adaptive.OurModel(
                    weakModel, preTrainedDataMatrix, classes, allFeatures, labeledGesturesFeatures,
                    labeledGesturesLabels, step, 'QDA', k, shotStart)

                adaptedModelLDA.to_pickle(
                    'pretrainedModels/adaptedModelLDA_' + typeDatabase + '_featureSet_' + str(
                        featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
                        seed) + '.pkl')
                adaptedModelQDA.to_pickle(
                    'pretrainedModels/adaptedModelQDA_' + typeDatabase + '_featureSet_' + str(
                        featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
                        seed) + '.pkl')


            numberUnlabeledGestures = 0

            name = 'LDA' + '_' + 'weak'
            results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelLDA(
                testFeatures, testLabels, weakModel, classes)
            # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

            name = 'QDA' + '_' + 'weak'
            results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelQDA(
                testFeatures, testLabels, weakModel, classes)
            # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

            name = 'LDA' + '_' + 'adapted'
            results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelLDA(
                testFeatures, testLabels, adaptedModelLDA, classes)
            # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

            name = 'QDA' + '_' + 'adapted'
            results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelQDA(
                testFeatures, testLabels, adaptedModelQDA, classes)
            # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

            numSamples = 50
            labeledGesturesFeatures, labeledGesturesLabels = adaptive.subsetTraining(
                labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)

            LDA_incre_proposedMcc = weakModel.copy()
            LDA_incre_proposedLabel = weakModel.copy()
            LDA_incre_sequential = weakModel.copy()
            LDA_incre_supervised = weakModel.copy()
            LDA_incre_Nigam_01 = weakModel.copy()
            LDA_incre_Nigam_02 = weakModel.copy()
            LDA_incre_Nigam_04 = weakModel.copy()
            LDA_incre_Nigam_06 = weakModel.copy()
            LDA_incre_Nigam_08 = weakModel.copy()
            LDA_incre_Nigam_10 = weakModel.copy()

            QDA_incre_proposedMcc = weakModel.copy()
            QDA_incre_proposedLabel = weakModel.copy()
            QDA_incre_sequential = weakModel.copy()
            QDA_incre_supervised = weakModel.copy()
            QDA_incre_Nigam_01 = weakModel.copy()
            QDA_incre_Nigam_02 = weakModel.copy()
            QDA_incre_Nigam_04 = weakModel.copy()
            QDA_incre_Nigam_06 = weakModel.copy()
            QDA_incre_Nigam_08 = weakModel.copy()
            QDA_incre_Nigam_10 = weakModel.copy()

            LDA_incre_proposedMcc_adapt = adaptedModelLDA.copy()
            QDA_incre_proposedMcc_adapt = adaptedModelQDA.copy()

            for rand in list(permutationUnlabeledGestures):
                trainFeatures = dataMatrix[
                                (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 0) &
                                (dataMatrix[:, allFeatures + 3] == rand[0]) & (
                                        dataMatrix[:, allFeatures + 2] == rand[1]), 0:allFeatures]
                # trainLabels = dataMatrix[
                #     (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 0) &
                #     (dataMatrix[:, allFeatures + 3] == rand[0]) & (
                #             dataMatrix[:, allFeatures + 2] == rand[1]), allFeatures + 2]

                trainFeatures = scaler.transform(trainFeatures)



                numberUnlabeledGestures += 1

                type_DA = 'LDA'

                name = type_DA + '_' + 'incre_proposedMcc_adapt'
                LDA_incre_proposedMcc_adapt, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposedProbMSDA(
                    LDA_incre_proposedMcc_adapt, classes, trainFeatures, labeledGesturesFeatures, labeledGesturesLabels,
                    'LDA')

                name = type_DA + '_' + 'incre_proposedMcc'
                LDA_incre_proposedMcc, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposedProbMSDA(
                    LDA_incre_proposedMcc, classes, trainFeatures, labeledGesturesFeatures, labeledGesturesLabels,
                    'LDA')

                name = type_DA + '_' + 'incre_proposedLabel'
                LDA_incre_proposedLabel, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposedLabel(
                    LDA_incre_proposedLabel, classes, trainFeatures, 'LDA')

                name = type_DA + '_' + 'incre_sequential'
                LDA_incre_sequential, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_sequential_labels(
                    LDA_incre_sequential, classes, trainFeatures, 'LDA')

                name = type_DA + '_' + 'incre_supervised'
                LDA_incre_supervised, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_supervised(
                    LDA_incre_supervised, classes, trainFeatures, 'LDA', rand[1])

                name = type_DA + '_' + 'incre_Nigam_' + str(0.1)
                LDA_incre_Nigam_01, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposedNigam(
                    LDA_incre_Nigam_01, classes, trainFeatures, 'LDA', weight_Nigam=0.1)
                name = type_DA + '_' + 'incre_Nigam_' + str(0.2)
                LDA_incre_Nigam_02, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposedNigam(
                    LDA_incre_Nigam_02, classes, trainFeatures, 'LDA', weight_Nigam=0.2)
                name = type_DA + '_' + 'incre_Nigam_' + str(0.4)
                LDA_incre_Nigam_04, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposedNigam(
                    LDA_incre_Nigam_04, classes, trainFeatures, 'LDA', weight_Nigam=0.4)
                name = type_DA + '_' + 'incre_Nigam_' + str(0.6)
                LDA_incre_Nigam_06, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposedNigam(
                    LDA_incre_Nigam_06, classes, trainFeatures, 'LDA', weight_Nigam=0.6)
                name = type_DA + '_' + 'incre_Nigam_' + str(0.8)
                LDA_incre_Nigam_08, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposedNigam(
                    LDA_incre_Nigam_08, classes, trainFeatures, 'LDA', weight_Nigam=0.8)
                name = type_DA + '_' + 'incre_Nigam_' + str(1.0)
                LDA_incre_Nigam_10, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposedNigam(
                    LDA_incre_Nigam_10, classes, trainFeatures, 'LDA', weight_Nigam=1)

                ###################################
                type_DA = 'QDA'

                name = type_DA + '_' + 'incre_proposedMcc_adapt'
                QDA_incre_proposedMcc_adapt, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposedProbMSDA(
                    QDA_incre_proposedMcc_adapt, classes, trainFeatures, labeledGesturesFeatures, labeledGesturesLabels,
                    'QDA')

                name = type_DA + '_' + 'incre_proposedMcc'
                QDA_incre_proposedMcc, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposedProbMSDA(
                    QDA_incre_proposedMcc, classes, trainFeatures, labeledGesturesFeatures, labeledGesturesLabels,
                    'QDA')

                name = type_DA + '_' + 'incre_proposedLabel'
                QDA_incre_proposedLabel, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposedLabel(
                    QDA_incre_proposedLabel, classes, trainFeatures, 'QDA')

                name = type_DA + '_' + 'incre_sequential'
                QDA_incre_sequential, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_sequential_labels(
                    QDA_incre_sequential, classes, trainFeatures, 'QDA')

                name = type_DA + '_' + 'incre_supervised'
                QDA_incre_supervised, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_supervised(
                    QDA_incre_supervised, classes, trainFeatures, 'QDA', rand[1])

                name = type_DA + '_' + 'incre_Nigam_' + str(0.1)
                QDA_incre_Nigam_01, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposedNigam(
                    QDA_incre_Nigam_01, classes, trainFeatures, 'QDA', weight_Nigam=0.1)
                name = type_DA + '_' + 'incre_Nigam_' + str(0.2)
                QDA_incre_Nigam_02, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposedNigam(
                    QDA_incre_Nigam_02, classes, trainFeatures, 'QDA', weight_Nigam=0.2)
                name = type_DA + '_' + 'incre_Nigam_' + str(0.4)
                QDA_incre_Nigam_04, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposedNigam(
                    QDA_incre_Nigam_04, classes, trainFeatures, 'QDA', weight_Nigam=0.4)
                name = type_DA + '_' + 'incre_Nigam_' + str(0.6)
                QDA_incre_Nigam_06, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposedNigam(
                    QDA_incre_Nigam_06, classes, trainFeatures, 'QDA', weight_Nigam=0.6)
                name = type_DA + '_' + 'incre_Nigam_' + str(0.8)
                QDA_incre_Nigam_08, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposedNigam(
                    QDA_incre_Nigam_08, classes, trainFeatures, 'QDA', weight_Nigam=0.8)
                name = type_DA + '_' + 'incre_Nigam_' + str(1.0)
                QDA_incre_Nigam_10, results.at[idx, 'time' + '_' + name], results.at[
                    idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposedNigam(
                    QDA_incre_Nigam_10, classes, trainFeatures, 'QDA', weight_Nigam=1)

                results.at[idx, 'Feature Set'] = featureSet
                results.at[idx, 'person'] = person
                results.at[idx, 'exp_time'] = seed
                results.at[idx, 'shot_class'] = rand
                results.at[idx, 'unlabeled Gesture'] = numberUnlabeledGestures

                idx += 1
                if np.any(evaluatedGestures == np.ones(len(evaluatedGestures)) * (numberUnlabeledGestures)):
                    idx -= 1
                    type_DA = 'LDA'

                    name = type_DA + '_' + 'incre_proposedMcc_adapt'
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelLDA(
                        testFeatures, testLabels, LDA_incre_proposedMcc_adapt, classes)
                    # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    name = type_DA + '_' + 'incre_proposedMcc'
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelLDA(
                        testFeatures, testLabels, LDA_incre_proposedMcc, classes)
                    # print('\n' + name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    name = type_DA + '_' + 'incre_proposedLabel'
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelLDA(
                        testFeatures, testLabels, LDA_incre_proposedLabel, classes)
                    # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    name = type_DA + '_' + 'incre_sequential'
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelLDA(
                        testFeatures, testLabels, LDA_incre_sequential, classes)
                    # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    name = type_DA + '_' + 'incre_supervised'
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelLDA(
                        testFeatures, testLabels, LDA_incre_supervised, classes)
                    # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    name = type_DA + '_' + 'incre_Nigam_' + str(0.1)
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelLDA(
                        testFeatures, testLabels, LDA_incre_Nigam_01, classes)
                    # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    name = type_DA + '_' + 'incre_Nigam_' + str(0.2)
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelLDA(
                        testFeatures, testLabels, LDA_incre_Nigam_02, classes)
                    # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    name = type_DA + '_' + 'incre_Nigam_' + str(0.4)
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelLDA(
                        testFeatures, testLabels, LDA_incre_Nigam_04, classes)
                    # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    name = type_DA + '_' + 'incre_Nigam_' + str(0.6)
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelLDA(
                        testFeatures, testLabels, LDA_incre_Nigam_06, classes)
                    # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    name = type_DA + '_' + 'incre_Nigam_' + str(0.8)
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelLDA(
                        testFeatures, testLabels, LDA_incre_Nigam_08, classes)
                    # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    name = type_DA + '_' + 'incre_Nigam_' + str(1.0)
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelLDA(
                        testFeatures, testLabels, LDA_incre_Nigam_10, classes)
                    # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    ###QDA
                    type_DA = 'QDA'

                    name = type_DA + '_' + 'incre_proposedMcc_adapt'
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelQDA(
                        testFeatures, testLabels, QDA_incre_proposedMcc_adapt, classes)
                    # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    name = type_DA + '_' + 'incre_proposedMcc'
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelQDA(
                        testFeatures, testLabels, QDA_incre_proposedMcc, classes)
                    # print('\n' + name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    name = type_DA + '_' + 'incre_proposedLabel'
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelQDA(
                        testFeatures, testLabels, QDA_incre_proposedLabel, classes)
                    # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    name = type_DA + '_' + 'incre_sequential'
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelQDA(
                        testFeatures, testLabels, QDA_incre_sequential, classes)
                    # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    name = type_DA + '_' + 'incre_supervised'
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelQDA(
                        testFeatures, testLabels, QDA_incre_supervised, classes)
                    # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    name = type_DA + '_' + 'incre_Nigam_' + str(0.1)
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelQDA(
                        testFeatures, testLabels, QDA_incre_Nigam_01, classes)
                    # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    name = type_DA + '_' + 'incre_Nigam_' + str(0.2)
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelQDA(
                        testFeatures, testLabels, QDA_incre_Nigam_02, classes)
                    # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    name = type_DA + '_' + 'incre_Nigam_' + str(0.4)
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelQDA(
                        testFeatures, testLabels, QDA_incre_Nigam_04, classes)
                    # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    name = type_DA + '_' + 'incre_Nigam_' + str(0.6)
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelQDA(
                        testFeatures, testLabels, QDA_incre_Nigam_06, classes)
                    # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    name = type_DA + '_' + 'incre_Nigam_' + str(0.8)
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelQDA(
                        testFeatures, testLabels, QDA_incre_Nigam_08, classes)
                    # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    name = type_DA + '_' + 'incre_Nigam_' + str(1.0)
                    results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
                        idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelQDA(
                        testFeatures, testLabels, QDA_incre_Nigam_10, classes)
                    # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])

                    idx += 1
                    results.to_csv(nameFile)


def PKModels(dataMatrix, classes, peoplePriorK, evaluatedPerson, allFeatures):
    preTrainedDataMatrix = pd.DataFrame(columns=['mean', 'cov', 'class', 'person'])
    indx = 0

    people = list(range(1, peoplePriorK + 1))
    for cla in range(1, classes + 1):
        for person in people:
            if person != evaluatedPerson:
                auxData = dataMatrix[
                          (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures + 2] == cla),
                          0:allFeatures]
                preTrainedDataMatrix.at[indx, 'cov'] = np.cov(auxData, rowvar=False)
                preTrainedDataMatrix.at[indx, 'mean'] = np.mean(auxData, axis=0)
                preTrainedDataMatrix.at[indx, 'class'] = cla
                preTrainedDataMatrix.at[indx, 'person'] = person
                indx += 1

    return preTrainedDataMatrix


def resultsDataframe(
        trainFeatures, trainLabels, classes, results, testFeatures, testLabels, idx, person, exp_time, nShots, rand,
        featureSet, nameFile, printR, labeledGesturesFeatures, labeledGesturesLabels, weakModel, samplesInMemory,
        shotStart, adaptedModelLDA, adaptedModelQDA, unlabeledGesturesTotal, dataTotal, LDA_incre_gestures_labels,
        LDA_incre_gestures_weight, LDA_incre_gestures_weight_MSDA, LDA_incre_samples_labels, LDA_incre_samples_prob,
        LDA_incre_gestures_labels_adapt, LDA_incre_gestures_weight_adapt, LDA_incre_gestures_weight_MSDA_adapt,
        LDA_incre_samples_labels_adapt, LDA_incre_samples_prob_adapt, QDA_incre_gestures_labels,
        QDA_incre_gestures_weight, QDA_incre_gestures_weight_MSDA, QDA_incre_samples_labels, QDA_incre_samples_prob,
        QDA_incre_gestures_labels_adapt, QDA_incre_gestures_weight_adapt, QDA_incre_gestures_weight_MSDA_adapt,
        QDA_incre_samples_labels_adapt, QDA_incre_samples_prob_adapt):
    print('label: ', trainLabels.mean())
    type_DA_set = ['LDA', 'QDA']
    for type_DA in type_DA_set:
        if type_DA == 'LDA':

            LDA_incre_gestures_labels, results = SemiSupervisedModels(
                LDA_incre_gestures_labels, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='incre_gestures_labels')
            LDA_incre_gestures_weight, results = SemiSupervisedModels(
                LDA_incre_gestures_weight, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='incre_gestures_weight')
            LDA_incre_gestures_weight_MSDA, results = SemiSupervisedModels(
                LDA_incre_gestures_weight_MSDA, type_DA, trainFeatures, trainLabels,
                classes, labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures,
                testLabels, samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='incre_gestures_weight_MSDA')
            LDA_incre_samples_labels, results = SemiSupervisedModels(
                LDA_incre_samples_labels, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='incre_samples_labels')
            LDA_incre_samples_prob, results = SemiSupervisedModels(
                LDA_incre_samples_prob, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='incre_samples_prob')

            noModel = None
            _, results = SemiSupervisedModels(
                noModel, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='semi_gestures_labels')
            _, results = SemiSupervisedModels(
                noModel, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='semi_gestures_weight')
            _, results = SemiSupervisedModels(
                noModel, type_DA, trainFeatures, trainLabels,
                classes, labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures,
                testLabels, samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='semi_gestures_weight_MSDA')
            _, results = SemiSupervisedModels(
                noModel, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='semi_samples_labels')
            _, results = SemiSupervisedModels(
                noModel, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='semi_samples_prob')

            ## WITH ADAPTATION

            LDA_incre_gestures_labels_adapt, results = SemiSupervisedModels(
                LDA_incre_gestures_labels_adapt, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, adaptedModelLDA, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='incre_gestures_labels', adaptation='_Adapted')
            LDA_incre_gestures_weight_adapt, results = SemiSupervisedModels(
                LDA_incre_gestures_weight_adapt, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, adaptedModelLDA, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='incre_gestures_weight', adaptation='_Adapted')
            LDA_incre_gestures_weight_MSDA_adapt, results = SemiSupervisedModels(
                LDA_incre_gestures_weight_MSDA_adapt, type_DA, trainFeatures, trainLabels,
                classes, labeledGesturesFeatures, labeledGesturesLabels, adaptedModelLDA, results, idx, testFeatures,
                testLabels, samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='incre_gestures_weight_MSDA', adaptation='_Adapted')
            LDA_incre_samples_labels_adapt, results = SemiSupervisedModels(
                LDA_incre_samples_labels_adapt, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, adaptedModelLDA, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='incre_samples_labels', adaptation='_Adapted')
            LDA_incre_samples_prob_adapt, results = SemiSupervisedModels(
                LDA_incre_samples_prob_adapt, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, adaptedModelLDA, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='incre_samples_prob', adaptation='_Adapted')

            noModel = None
            _, results = SemiSupervisedModels(
                noModel, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, adaptedModelLDA, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='semi_gestures_labels', adaptation='_Adapted')
            _, results = SemiSupervisedModels(
                noModel, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, adaptedModelLDA, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='semi_gestures_weight', adaptation='_Adapted')
            _, results = SemiSupervisedModels(
                noModel, type_DA, trainFeatures, trainLabels,
                classes, labeledGesturesFeatures, labeledGesturesLabels, adaptedModelLDA, results, idx, testFeatures,
                testLabels, samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='semi_gestures_weight_MSDA', adaptation='_Adapted')
            _, results = SemiSupervisedModels(
                noModel, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, adaptedModelLDA, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='semi_samples_labels', adaptation='_Adapted')
            _, results = SemiSupervisedModels(
                noModel, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, adaptedModelLDA, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='semi_samples_prob', adaptation='_Adapted')

        elif type_DA == 'QDA':

            QDA_incre_gestures_labels, results = SemiSupervisedModels(
                QDA_incre_gestures_labels, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='incre_gestures_labels')
            QDA_incre_gestures_weight, results = SemiSupervisedModels(
                QDA_incre_gestures_weight, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='incre_gestures_weight')
            QDA_incre_gestures_weight_MSDA, results = SemiSupervisedModels(
                QDA_incre_gestures_weight_MSDA, type_DA, trainFeatures, trainLabels,
                classes, labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures,
                testLabels, samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='incre_gestures_weight_MSDA')
            QDA_incre_samples_labels, results = SemiSupervisedModels(
                QDA_incre_samples_labels, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='incre_samples_labels')
            QDA_incre_samples_prob, results = SemiSupervisedModels(
                QDA_incre_samples_prob, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='incre_samples_prob')

            noModel = None
            _, results = SemiSupervisedModels(
                noModel, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='semi_gestures_labels')
            _, results = SemiSupervisedModels(
                noModel, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='semi_gestures_weight')
            _, results = SemiSupervisedModels(
                noModel, type_DA, trainFeatures, trainLabels,
                classes, labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures,
                testLabels, samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='semi_gestures_weight_MSDA')
            _, results = SemiSupervisedModels(
                noModel, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='semi_samples_labels')
            _, results = SemiSupervisedModels(
                noModel, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='semi_samples_prob')

            ## WITH ADAPTATION

            QDA_incre_gestures_labels_adapt, results = SemiSupervisedModels(
                QDA_incre_gestures_labels_adapt, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, adaptedModelQDA, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='incre_gestures_labels', adaptation='_Adapted')
            QDA_incre_gestures_weight_adapt, results = SemiSupervisedModels(
                QDA_incre_gestures_weight_adapt, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, adaptedModelQDA, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='incre_gestures_weight', adaptation='_Adapted')
            QDA_incre_gestures_weight_MSDA_adapt, results = SemiSupervisedModels(
                QDA_incre_gestures_weight_MSDA_adapt, type_DA, trainFeatures, trainLabels,
                classes, labeledGesturesFeatures, labeledGesturesLabels, adaptedModelQDA, results, idx, testFeatures,
                testLabels, samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='incre_gestures_weight_MSDA', adaptation='_Adapted')
            QDA_incre_samples_labels_adapt, results = SemiSupervisedModels(
                QDA_incre_samples_labels_adapt, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, adaptedModelQDA, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='incre_samples_labels', adaptation='_Adapted')
            QDA_incre_samples_prob_adapt, results = SemiSupervisedModels(
                QDA_incre_samples_prob_adapt, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, adaptedModelQDA, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='incre_samples_prob', adaptation='_Adapted')

            noModel = None
            _, results = SemiSupervisedModels(
                noModel, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, adaptedModelQDA, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='semi_gestures_labels', adaptation='_Adapted')
            _, results = SemiSupervisedModels(
                noModel, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, adaptedModelQDA, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='semi_gestures_weight', adaptation='_Adapted')
            _, results = SemiSupervisedModels(
                noModel, type_DA, trainFeatures, trainLabels,
                classes, labeledGesturesFeatures, labeledGesturesLabels, adaptedModelQDA, results, idx, testFeatures,
                testLabels, samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='semi_gestures_weight_MSDA', adaptation='_Adapted')
            _, results = SemiSupervisedModels(
                noModel, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, adaptedModelQDA, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='semi_samples_labels', adaptation='_Adapted')
            _, results = SemiSupervisedModels(
                noModel, type_DA, trainFeatures, trainLabels, classes,
                labeledGesturesFeatures, labeledGesturesLabels, adaptedModelQDA, results, idx, testFeatures, testLabels,
                samplesInMemory, shotStart, nShots, unlabeledGesturesTotal, dataTotal,
                typeModel='semi_samples_prob', adaptation='_Adapted')

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

    return results, idx, LDA_incre_gestures_labels, LDA_incre_gestures_weight, LDA_incre_gestures_weight_MSDA, \
           LDA_incre_samples_labels, LDA_incre_samples_prob, LDA_incre_gestures_labels_adapt, \
           LDA_incre_gestures_weight_adapt, LDA_incre_gestures_weight_MSDA_adapt, LDA_incre_samples_labels_adapt, \
           LDA_incre_samples_prob_adapt, QDA_incre_gestures_labels, QDA_incre_gestures_weight, \
           QDA_incre_gestures_weight_MSDA, QDA_incre_samples_labels, QDA_incre_samples_prob, \
           QDA_incre_gestures_labels_adapt, QDA_incre_gestures_weight_adapt, QDA_incre_gestures_weight_MSDA_adapt, \
           QDA_incre_samples_labels_adapt, QDA_incre_samples_prob_adapt


def SemiSupervisedModels(currentModel, type_DA, trainFeatures, trainLabels, classes, labeledGesturesFeatures,
                         labeledGesturesLabels, weakModel, results, idx, testFeatures, testLabels, samplesInMemory,
                         shotStart, nShots, unlabeledGesturesTotal, dataTotal, typeModel, adaptation=''):
    name = 'Acc' + type_DA + '_' + typeModel + adaptation

    if typeModel == 'incre_gestures_labels' or typeModel == 'incre_gestures_weight' or \
            typeModel == 'incre_gestures_weight_MSDA' or typeModel == 'incre_samples_labels' or \
            typeModel == 'incre_samples_prob':
        postProb_trainFeatures = SemiSupervised.post_probabilities_Calculation(trainFeatures, currentModel, classes,
                                                                               type_DA)
    else:
        postProb_trainFeatures = 0
    # print(name)

    if typeModel == 'incre_gestures_labels':
        updatedModel, results.at[
            idx, 'time_' + name], predictedWeight = SemiSupervised.model_incre_gestures_labels(
            currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
            labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)

    elif typeModel == 'incre_gestures_weight':
        updatedModel, results.at[idx, 'time_' + name], predictedWeight = SemiSupervised.model_incre_gestures_weight(
            currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
            labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)

    elif typeModel == 'incre_gestures_weight_MSDA':
        updatedModel, results.at[
            idx, 'time_' + name], predictedWeight = SemiSupervised.model_incre_gestures_weight_MSDA(
            currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
            labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
    elif typeModel == 'incre_samples_labels':
        updatedModel, results.at[idx, 'time_' + name], predictedWeight = SemiSupervised.model_incre_samples_labels(
            currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
            labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)
    elif typeModel == 'incre_samples_prob':
        updatedModel, results.at[idx, 'time_' + name], predictedWeight = SemiSupervised.model_incre_samples_prob(
            currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
            labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart)

    elif typeModel == 'semi_gestures_labels':
        updatedModel, results.at[
            idx, 'time_' + name], predictedWeight = SemiSupervised.model_semi_gestures_labels(
            currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
            labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart, unlabeledGesturesTotal,
            dataTotal)

    elif typeModel == 'semi_gestures_weight':
        updatedModel, results.at[idx, 'time_' + name], predictedWeight = SemiSupervised.model_semi_gestures_weight(
            currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
            labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart, unlabeledGesturesTotal,
            dataTotal)

    elif typeModel == 'semi_gestures_weight_MSDA':
        updatedModel, results.at[
            idx, 'time_' + name], predictedWeight = SemiSupervised.model_semi_gestures_weight_MSDA(
            currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
            labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart, unlabeledGesturesTotal,
            dataTotal)
    elif typeModel == 'semi_samples_labels':
        updatedModel, results.at[idx, 'time_' + name], predictedWeight = SemiSupervised.model_semi_samples_labels(
            currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
            labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart, unlabeledGesturesTotal,
            dataTotal)
    elif typeModel == 'semi_samples_prob':
        updatedModel, results.at[idx, 'time_' + name], predictedWeight = SemiSupervised.model_semi_samples_prob(
            currentModel, classes, trainFeatures, postProb_trainFeatures, weakModel,
            labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart, unlabeledGesturesTotal,
            dataTotal)

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
    currentValues = pd.DataFrame(
        columns=['cov', 'mean', 'class', 'N', 'LDAcov'])
    trainLabelsAux = trainLabels[np.newaxis]
    Matrix = np.hstack((trainFeatures, trainLabelsAux.T))
    for cla in range(classes):
        X = Matrix[np.where((Matrix[:, allFeatures] == cla + 1)), 0:allFeatures][0]
        currentValues.at[cla, 'cov'] = np.cov(X, rowvar=False)
        # currentValues.at[cla, 'cov_accumulated'] = currentValues.loc[cla, 'cov']
        currentValues.at[cla, 'mean'] = np.mean(X, axis=0)
        # currentValues.at[cla, 'mean_accumulated'] = currentValues.loc[cla, 'mean']
        currentValues.at[cla, 'class'] = cla + 1
        N = np.size(X, axis=0)
        currentValues.at[cla, 'N'] = N
    currentValues.at[0, 'LDAcov'] = DA_Classifiers.LDA_Cov_weights(currentValues)
    return currentValues


def preprocessingPK(dataMatrix, allFeatures, scaler):
    dataMatrixFeatures = scaler.transform(dataMatrix[:, :allFeatures])
    return np.hstack((dataMatrixFeatures, dataMatrix[:, allFeatures:])), np.size(dataMatrixFeatures, axis=1)
