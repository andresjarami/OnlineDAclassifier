import math

import numpy as np
import pandas as pd
import time
from sklearn import preprocessing
from sklearn.decomposition import PCA

import DA_classifiers as DA_classifiers
import models as models


# Upload Databases

def uploadDatabases(Database, person, featureSet):
    # Setting general variables
    path = '../'
    segment = '_290ms'
    personFile = '_' + str(person)

    if Database == 'EPN_612':
        classes = 6
        CH = 8
        numberShots = 50
        # numSamples = 250  # to holds real-time
        numSamples = 20  # to holds real-time
        days = 1
        testSamples = 25
        # best parameters (according accuracy) for classifer using posterior probabilities weights and threshold
        b_wp = 1.0
        b_t = 0.5
    elif Database == 'Nina5':
        classes = 18
        CH = 16
        numberShots = 6
        numSamples = 20  # to holds real-time
        days = 1
        testSamples = 4
        # best parameters (according accuracy) for classifer using posterior probabilities weights and threshold
        b_wp = 0.6
        b_t = 0.1
    elif Database == 'Cote':
        classes = 7
        CH = 8
        numberShots = 12
        # numSamples = 120  # to holds real-time
        numSamples = 20  # to holds real-time
        days = 1
        testSamples = 9
        # best parameters (according accuracy) for classifer using posterior probabilities weights and threshold
        b_wp = 1.0
        b_t = 0.4
    elif Database == 'Capgmyo_dbb':
        classes = 8
        CH = 128
        numberShots = 10
        numSamples = 20  # to holds real-time
        days = 2
        testSamples = 7
        # best parameters (according accuracy) for classifer using posterior probabilities weights and threshold
        b_wp = 1.0
        b_t = 0.5
    elif Database == 'LongTerm3DC':
        classes = 11
        CH = 10
        numberShots = 3
        numSamples = 20  # to holds real-time
        days = 3
        testSamples = 2
        # best parameters (according accuracy) for classifer using posterior probabilities weights and threshold
        b_wp = 1.0
        b_t = 0.7

    if featureSet == 1:
        # Setting variables
        Feature1 = 'logvar'

        numberFeatures = 1
        allFeatures = numberFeatures * CH
        # Getting Data
        logvarMatrix = np.load(path + 'ExtractedData/' + Database + '/' + Feature1 + segment + personFile + '.npy')

        # if Database == 'Nina5':
        #     dataMatrix = logvarMatrix[:, 8:]
        # else:
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
        mavMatrix = np.load(path + 'ExtractedData/' + Database + '/' + Feature1 + segment + personFile + '.npy')
        wlMatrix = np.load(path + 'ExtractedData/' + Database + '/' + Feature2 + segment + personFile + '.npy')
        zcMatrix = np.load(path + 'ExtractedData/' + Database + '/' + Feature3 + segment + personFile + '.npy')
        sscMatrix = np.load(path + 'ExtractedData/' + Database + '/' + Feature4 + segment + personFile + '.npy')

        # if Database == 'Nina5':
        #     dataMatrix = np.hstack(
        #         (mavMatrix[:, 8:CH * 2], wlMatrix[:, 8:CH * 2], zcMatrix[:, 8:CH * 2], sscMatrix[:, 8:]))
        # else:
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
        lscaleMatrix = np.load(path + 'ExtractedData/' + Database + '/' + Feature1 + segment + personFile + '.npy')
        mflMatrix = np.load(path + 'ExtractedData/' + Database + '/' + Feature2 + segment + personFile + '.npy')
        msrMatrix = np.load(path + 'ExtractedData/' + Database + '/' + Feature3 + segment + personFile + '.npy')
        wampMatrix = np.load(path + 'ExtractedData/' + Database + '/' + Feature4 + segment + personFile + '.npy')

        # if Database == 'Nina5':
        #     dataMatrix = np.hstack(
        #         (lscaleMatrix[:, 8:CH * 2], mflMatrix[:, 8:CH * 2], msrMatrix[:, 8:CH * 2], wampMatrix[:, 8:]))
        # else:
        dataMatrix = np.hstack((lscaleMatrix[:, :CH], mflMatrix[:, :CH], msrMatrix[:, :CH], wampMatrix[:, :]))

        labelsDataMatrix = dataMatrix[:, allFeatures + 2]

    return dataMatrix, classes, numberShots, allFeatures, numSamples, days, testSamples, b_wp, b_t


def evaluation(featureSet, nameFile, startPerson, endPerson, printR, initialSamples, initialTime, finalTime,
               typeDatabase):
    scaler = preprocessing.MinMaxScaler()

    results = pd.DataFrame(
        columns=['Feature Set', 'person', 'exp_time', 'shots in traning', 'shot_class', 'day', 'unlabeled Gesture'])

    for metric in ['precision', 'recall']:
        for type_DA in ['LDA', 'QDA']:
            results[metric + '_' + type_DA + '_' + 'weak'] = ''
            results[metric + '_' + type_DA + '_' + 'batch'] = ''

    for type_model in ['ours', 'state_art']:
        for metric in ['precision', 'recall']:
            for type_DA in ['LDA', 'QDA']:
                results[metric + '_' + type_DA + '_' + type_model + '_soft_labels'] = ''
                results[metric + '_' + type_DA + '_' + type_model + '_labels'] = ''
                for weight in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    results[metric + '_' + type_DA + '_' + type_model + '_probs_' + str(weight)] = ''
                for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    results[metric + '_' + type_DA + '_' + type_model + '_threshold_' + str(threshold)] = ''

    idx = 0

    for person in range(startPerson, endPerson + 1):
        # Upload Data

        dataMatrix, classes, numberShots, allFeatures, numSamples, days, testSamples, b_wp, b_t = uploadDatabases(
            typeDatabase, person, featureSet)

        for seed in range(initialTime, finalTime + 1):

            for day in range(1, days + 1):
                print('p', person, 's', seed, 'd', day)
                numberGestures = 1
                np.random.seed(person + seed * 1000 + day * 10000)  # to get a different seed per each person

                labeledGestures = []
                unlabeledGestures = []
                testGestures = []
                for cla in range(1, classes + 1):
                    repetitions = np.random.choice(numberShots, numberShots, replace=False) + 1
                    labeledGestures += [[shot, cla, day] for shot in repetitions[:initialSamples]]
                    unlabeledGestures += [[shot, cla, day] for shot in repetitions[initialSamples:testSamples]]
                    testGestures += [[shot, cla, day] for shot in repetitions[testSamples:]]

                permutationUnlabeledGestures = np.random.permutation(unlabeledGestures)
                # print(labeledGestures)
                # print(unlabeledGestures)
                # print(list(permutationUnlabeledGestures))
                # print(testGestures)

                labeledGesturesFeatures, labeledGesturesLabels = getGestureSet(dataMatrix, allFeatures,
                                                                               labeledGestures, person)
                labeledGesturesFeatures = scaler.fit_transform(labeledGesturesFeatures)

                testGesturesFeatures, testGesturesLabels = getGestureSet(dataMatrix, allFeatures, testGestures,
                                                                         person)
                testGesturesFeatures = scaler.transform(testGesturesFeatures)

                if typeDatabase == 'Capgmyo_dbb':
                    pca_model = PCA(n_components=0.99, svd_solver='full')
                    labeledGesturesFeatures = pca_model.fit_transform(labeledGesturesFeatures)
                    testGesturesFeatures = pca_model.transform(testGesturesFeatures)
                    print('features', np.size(labeledGesturesFeatures, axis=1))

                weak_model, _, _ = currentDistributionValues(
                    labeledGesturesFeatures, labeledGesturesLabels, classes, np.size(labeledGesturesFeatures, axis=1))

                accumLabeledGesturesFeatures = labeledGesturesFeatures
                accumLabeledGesturesLabels = labeledGesturesLabels

                labeledGesturesFeatures, labeledGesturesLabels = systematicSampling(
                    labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)

                LDA_ours_labels = weak_model.copy()
                LDA_ours_soft_labels = weak_model.copy()
                LDA_ours_probs_01 = weak_model.copy()
                LDA_ours_probs_02 = weak_model.copy()
                LDA_ours_probs_03 = weak_model.copy()
                LDA_ours_probs_04 = weak_model.copy()
                LDA_ours_probs_05 = weak_model.copy()
                LDA_ours_probs_06 = weak_model.copy()
                LDA_ours_probs_07 = weak_model.copy()
                LDA_ours_probs_08 = weak_model.copy()
                LDA_ours_probs_09 = weak_model.copy()
                LDA_ours_probs_10 = weak_model.copy()
                LDA_ours_threshold_00 = weak_model.copy()
                LDA_ours_threshold_01 = weak_model.copy()
                LDA_ours_threshold_02 = weak_model.copy()
                LDA_ours_threshold_03 = weak_model.copy()
                LDA_ours_threshold_04 = weak_model.copy()
                LDA_ours_threshold_05 = weak_model.copy()
                LDA_ours_threshold_06 = weak_model.copy()
                LDA_ours_threshold_07 = weak_model.copy()
                LDA_ours_threshold_08 = weak_model.copy()
                LDA_ours_threshold_09 = weak_model.copy()

                LDA_state_art_labels = weak_model.copy()
                LDA_state_art_soft_labels = weak_model.copy()
                LDA_state_art_probs_01 = weak_model.copy()
                LDA_state_art_probs_02 = weak_model.copy()
                LDA_state_art_probs_03 = weak_model.copy()
                LDA_state_art_probs_04 = weak_model.copy()
                LDA_state_art_probs_05 = weak_model.copy()
                LDA_state_art_probs_06 = weak_model.copy()
                LDA_state_art_probs_07 = weak_model.copy()
                LDA_state_art_probs_08 = weak_model.copy()
                LDA_state_art_probs_09 = weak_model.copy()
                LDA_state_art_probs_10 = weak_model.copy()
                LDA_state_art_threshold_00 = weak_model.copy()
                LDA_state_art_threshold_01 = weak_model.copy()
                LDA_state_art_threshold_02 = weak_model.copy()
                LDA_state_art_threshold_03 = weak_model.copy()
                LDA_state_art_threshold_04 = weak_model.copy()
                LDA_state_art_threshold_05 = weak_model.copy()
                LDA_state_art_threshold_06 = weak_model.copy()
                LDA_state_art_threshold_07 = weak_model.copy()
                LDA_state_art_threshold_08 = weak_model.copy()
                LDA_state_art_threshold_09 = weak_model.copy()

                QDA_ours_labels = weak_model.copy()
                QDA_ours_soft_labels = weak_model.copy()
                QDA_ours_probs_01 = weak_model.copy()
                QDA_ours_probs_02 = weak_model.copy()
                QDA_ours_probs_03 = weak_model.copy()
                QDA_ours_probs_04 = weak_model.copy()
                QDA_ours_probs_05 = weak_model.copy()
                QDA_ours_probs_06 = weak_model.copy()
                QDA_ours_probs_07 = weak_model.copy()
                QDA_ours_probs_08 = weak_model.copy()
                QDA_ours_probs_09 = weak_model.copy()
                QDA_ours_probs_10 = weak_model.copy()
                QDA_ours_threshold_00 = weak_model.copy()
                QDA_ours_threshold_01 = weak_model.copy()
                QDA_ours_threshold_02 = weak_model.copy()
                QDA_ours_threshold_03 = weak_model.copy()
                QDA_ours_threshold_04 = weak_model.copy()
                QDA_ours_threshold_05 = weak_model.copy()
                QDA_ours_threshold_06 = weak_model.copy()
                QDA_ours_threshold_07 = weak_model.copy()
                QDA_ours_threshold_08 = weak_model.copy()
                QDA_ours_threshold_09 = weak_model.copy()

                QDA_state_art_labels = weak_model.copy()
                QDA_state_art_soft_labels = weak_model.copy()
                QDA_state_art_probs_01 = weak_model.copy()
                QDA_state_art_probs_02 = weak_model.copy()
                QDA_state_art_probs_03 = weak_model.copy()
                QDA_state_art_probs_04 = weak_model.copy()
                QDA_state_art_probs_05 = weak_model.copy()
                QDA_state_art_probs_06 = weak_model.copy()
                QDA_state_art_probs_07 = weak_model.copy()
                QDA_state_art_probs_08 = weak_model.copy()
                QDA_state_art_probs_09 = weak_model.copy()
                QDA_state_art_probs_10 = weak_model.copy()
                QDA_state_art_threshold_00 = weak_model.copy()
                QDA_state_art_threshold_01 = weak_model.copy()
                QDA_state_art_threshold_02 = weak_model.copy()
                QDA_state_art_threshold_03 = weak_model.copy()
                QDA_state_art_threshold_04 = weak_model.copy()
                QDA_state_art_threshold_05 = weak_model.copy()
                QDA_state_art_threshold_06 = weak_model.copy()
                QDA_state_art_threshold_07 = weak_model.copy()
                QDA_state_art_threshold_08 = weak_model.copy()
                QDA_state_art_threshold_09 = weak_model.copy()

                # Number of points show in the graphs
                number_points = 5
                reported_points = np.append(
                    np.arange(np.floor(len(permutationUnlabeledGestures) / number_points),
                              len(permutationUnlabeledGestures),
                              np.floor(len(permutationUnlabeledGestures) / number_points)),
                    len(permutationUnlabeledGestures))

                for rand in list(permutationUnlabeledGestures):
                    trainFeatures = dataMatrix[
                                    (dataMatrix[:, allFeatures] == rand[2]) & (
                                            dataMatrix[:, allFeatures + 1] == person) & (
                                            dataMatrix[:, allFeatures + 3] == rand[0]) & (
                                            dataMatrix[:, allFeatures + 2] == rand[1]), 0:allFeatures]
                    trainLabels = dataMatrix[
                        (dataMatrix[:, allFeatures] == rand[2]) & (dataMatrix[:, allFeatures + 1] == person) & (
                                dataMatrix[:, allFeatures + 3] == rand[0]) & (
                                dataMatrix[:, allFeatures + 2] == rand[1]), allFeatures + 2]

                    print('current label', rand[1])
                    true_weight = np.zeros(classes)
                    true_weight[rand[1] - 1] = 1

                    #####Pre-processing
                    trainFeatures = scaler.transform(trainFeatures)
                    if typeDatabase == 'Capgmyo_dbb':
                        trainFeatures = pca_model.transform(trainFeatures)

                    time_incrementalSet = time.time()
                    gesture_mean = DA_classifiers.mean(trainFeatures, np.size(trainFeatures, axis=1))
                    gesture_cov = DA_classifiers.covariance(trainFeatures, np.size(trainFeatures, axis=1), gesture_mean)
                    gesture_N = np.size(trainFeatures, axis=0)
                    results.at[idx, 'time_incrementalSet'] = time.time() - time_incrementalSet

                    if printR:
                        print('person', person, 'day', day, 'exp', seed, 'gesture', numberGestures)

                    ###### Weak model

                    for type_DA in ['LDA', 'QDA']:
                        name = type_DA + '_' + 'weak'
                        results = update_results(results, idx, name, true_weight, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, weak_model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                    ###### Batch learning
                    name = 'batch'
                    accumLabeledGesturesFeatures = np.vstack((accumLabeledGesturesFeatures, trainFeatures))
                    accumLabeledGesturesLabels = np.hstack((accumLabeledGesturesLabels, trainLabels))

                    batch_model, results.at[idx, 'time_update_LDA_' + name], results.at[
                        idx, 'time_update_QDA_' + name], = currentDistributionValues(
                        accumLabeledGesturesFeatures, accumLabeledGesturesLabels, classes,
                        np.size(trainFeatures, axis=1))

                    for type_DA in ['LDA', 'QDA']:
                        name = type_DA + '_' + 'batch'
                        results = update_results(results, idx, name, true_weight, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, batch_model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                    ################################# LDA #########################
                    type_DA = 'LDA'

                    ############################# OURS LDA

                    name = type_DA + '_' + 'ours_soft_labels'
                    LDA_ours_soft_labels, results.at[
                        idx, 'time_update' + '_' + name], results.at[
                        idx, 'time_weight' + '_' + name], weight_vector = models.model_ours_soft_labels(
                        LDA_ours_soft_labels, classes, trainFeatures, labeledGesturesFeatures, labeledGesturesLabels,
                        type_DA, weak_model, gesture_N, gesture_mean, gesture_cov)
                    model = LDA_ours_soft_labels
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)
                    model = LDA_ours_soft_labels
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_labels'
                    LDA_ours_labels, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_labels(
                        LDA_ours_labels, classes, rand[1], type_DA, gesture_N, gesture_mean, gesture_cov)
                    model = LDA_ours_labels
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    #### Probs with our LDA classifier

                    name = type_DA + '_' + 'ours_probs_' + str(0.1)
                    LDA_ours_probs_01, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                        LDA_ours_probs_01, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.1)
                    model = LDA_ours_probs_01
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_probs_' + str(0.2)
                    LDA_ours_probs_02, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                        LDA_ours_probs_02, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.2)
                    model = LDA_ours_probs_02
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_probs_' + str(0.3)
                    LDA_ours_probs_03, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                        LDA_ours_probs_03, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.3)
                    model = LDA_ours_probs_03
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_probs_' + str(0.4)
                    LDA_ours_probs_04, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                        LDA_ours_probs_04, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.4)
                    model = LDA_ours_probs_04
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_probs_' + str(0.5)
                    LDA_ours_probs_05, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                        LDA_ours_probs_05, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.5)
                    model = LDA_ours_probs_05
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_probs_' + str(0.6)
                    LDA_ours_probs_06, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                        LDA_ours_probs_06, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.6)
                    model = LDA_ours_probs_06
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_probs_' + str(0.7)
                    LDA_ours_probs_07, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                        LDA_ours_probs_07, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.7)
                    model = LDA_ours_probs_07
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_probs_' + str(0.8)
                    LDA_ours_probs_08, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                        LDA_ours_probs_08, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.8)
                    model = LDA_ours_probs_08
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_probs_' + str(0.9)
                    LDA_ours_probs_09, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                        LDA_ours_probs_09, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.9)
                    model = LDA_ours_probs_09
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_probs_' + str(1.0)
                    LDA_ours_probs_10, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                        LDA_ours_probs_10, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=1.0)
                    model = LDA_ours_probs_10
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    #### Threshold with our LDA classifier

                    name = type_DA + '_' + 'ours_threshold_' + str(0.0)
                    LDA_ours_threshold_00, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_threshold(
                        LDA_ours_threshold_00, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, threshold=0.0)
                    model = LDA_ours_threshold_00
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_threshold_' + str(0.1)
                    LDA_ours_threshold_01, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_threshold(
                        LDA_ours_threshold_01, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, threshold=0.1)
                    model = LDA_ours_threshold_01
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_threshold_' + str(0.2)
                    LDA_ours_threshold_02, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_threshold(
                        LDA_ours_threshold_02, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, threshold=0.2)
                    model = LDA_ours_threshold_02
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_threshold_' + str(0.3)
                    LDA_ours_threshold_03, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_threshold(
                        LDA_ours_threshold_03, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, threshold=0.3)
                    model = LDA_ours_threshold_03
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_threshold_' + str(0.4)
                    LDA_ours_threshold_04, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_threshold(
                        LDA_ours_threshold_04, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, threshold=0.4)
                    model = LDA_ours_threshold_04
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_threshold_' + str(0.5)
                    LDA_ours_threshold_05, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_threshold(
                        LDA_ours_threshold_05, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, threshold=0.5)
                    model = LDA_ours_threshold_05
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_threshold_' + str(0.6)
                    LDA_ours_threshold_06, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_threshold(
                        LDA_ours_threshold_06, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, threshold=0.6)
                    model = LDA_ours_threshold_06
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_threshold_' + str(0.7)
                    LDA_ours_threshold_07, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_threshold(
                        LDA_ours_threshold_07, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, threshold=0.7)
                    model = LDA_ours_threshold_07
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_threshold_' + str(0.8)
                    LDA_ours_threshold_08, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_threshold(
                        LDA_ours_threshold_08, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, threshold=0.8)
                    model = LDA_ours_threshold_08
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_threshold_' + str(0.9)
                    LDA_ours_threshold_09, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_threshold(
                        LDA_ours_threshold_09, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, threshold=0.9)
                    model = LDA_ours_threshold_09
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    ############################# STATE OF THE ART LDA

                    name = type_DA + '_' + 'state_art_soft_labels'
                    LDA_state_art_soft_labels, results.at[
                        idx, 'time_update' + '_' + name], results.at[
                        idx, 'time_weight' + '_' + name], weight_vector = models.model_state_art_soft_labels(
                        LDA_state_art_soft_labels, classes, trainFeatures, labeledGesturesFeatures,
                        labeledGesturesLabels,
                        type_DA, weak_model, gesture_N, gesture_mean, gesture_cov)
                    model = LDA_state_art_soft_labels
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_labels'
                    LDA_state_art_labels, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_labels(
                        LDA_state_art_labels, classes, rand[1], type_DA, gesture_N, gesture_mean, gesture_cov)
                    model = LDA_state_art_labels
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    #### Probs with state_art LDA classifier

                    name = type_DA + '_' + 'state_art_probs' + str(0.1)
                    LDA_state_art_probs_01, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_probabilities(
                        LDA_state_art_probs_01, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.1)
                    model = LDA_state_art_probs_01
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_probs' + str(0.2)
                    LDA_state_art_probs_02, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_probabilities(
                        LDA_state_art_probs_02, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.2)
                    model = LDA_state_art_probs_02
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_probs' + str(0.3)
                    LDA_state_art_probs_03, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_probabilities(
                        LDA_state_art_probs_03, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.3)
                    model = LDA_state_art_probs_03
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_probs' + str(0.4)
                    LDA_state_art_probs_04, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_probabilities(
                        LDA_state_art_probs_04, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.4)
                    model = LDA_state_art_probs_04
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_probs' + str(0.5)
                    LDA_state_art_probs_05, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_probabilities(
                        LDA_state_art_probs_05, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.5)
                    model = LDA_state_art_probs_05
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_probs' + str(0.6)
                    LDA_state_art_probs_06, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_probabilities(
                        LDA_state_art_probs_06, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.6)
                    model = LDA_state_art_probs_06
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_probs' + str(0.7)
                    LDA_state_art_probs_07, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_probabilities(
                        LDA_state_art_probs_07, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.7)
                    model = LDA_state_art_probs_07
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_probs' + str(0.8)
                    LDA_state_art_probs_08, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_probabilities(
                        LDA_state_art_probs_08, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.8)
                    model = LDA_state_art_probs_08
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_probs' + str(0.9)
                    LDA_state_art_probs_09, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_probabilities(
                        LDA_state_art_probs_09, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.9)
                    model = LDA_state_art_probs_09
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_probs' + str(1.0)
                    LDA_state_art_probs_10, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_probabilities(
                        LDA_state_art_probs_10, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=1.0)
                    model = LDA_state_art_probs_10
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    #### Threshold with our LDA classifier

                    name = type_DA + '_' + 'state_art_threshold' + str(0.0)
                    LDA_state_art_threshold_00, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_threshold(
                        LDA_state_art_threshold_00, classes, trainFeatures, type_DA, weak_model, gesture_N,
                        gesture_mean, gesture_cov, threshold=0.0)
                    model = LDA_state_art_threshold_00
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_threshold' + str(0.1)
                    LDA_state_art_threshold_01, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_threshold(
                        LDA_state_art_threshold_01, classes, trainFeatures, type_DA, weak_model, gesture_N,
                        gesture_mean, gesture_cov, threshold=0.1)
                    model = LDA_state_art_threshold_01
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_threshold' + str(0.2)
                    LDA_state_art_threshold_02, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_threshold(
                        LDA_state_art_threshold_02, classes, trainFeatures, type_DA, weak_model, gesture_N,
                        gesture_mean, gesture_cov, threshold=0.2)
                    model = LDA_state_art_threshold_02
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_threshold' + str(0.3)
                    LDA_state_art_threshold_03, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_threshold(
                        LDA_state_art_threshold_03, classes, trainFeatures, type_DA, weak_model, gesture_N,
                        gesture_mean, gesture_cov, threshold=0.3)
                    model = LDA_state_art_threshold_03
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_threshold' + str(0.4)
                    LDA_state_art_threshold_04, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_threshold(
                        LDA_state_art_threshold_04, classes, trainFeatures, type_DA, weak_model, gesture_N,
                        gesture_mean, gesture_cov, threshold=0.4)
                    model = LDA_state_art_threshold_04
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_threshold' + str(0.5)
                    LDA_state_art_threshold_05, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_threshold(
                        LDA_state_art_threshold_05, classes, trainFeatures, type_DA, weak_model, gesture_N,
                        gesture_mean, gesture_cov, threshold=0.5)
                    model = LDA_state_art_threshold_05
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_threshold' + str(0.6)
                    LDA_state_art_threshold_06, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_threshold(
                        LDA_state_art_threshold_06, classes, trainFeatures, type_DA, weak_model, gesture_N,
                        gesture_mean, gesture_cov, threshold=0.6)
                    model = LDA_state_art_threshold_06
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_threshold' + str(0.7)
                    LDA_state_art_threshold_07, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_threshold(
                        LDA_state_art_threshold_07, classes, trainFeatures, type_DA, weak_model, gesture_N,
                        gesture_mean, gesture_cov, threshold=0.7)
                    model = LDA_state_art_threshold_07
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_threshold' + str(0.8)
                    LDA_state_art_threshold_08, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_threshold(
                        LDA_state_art_threshold_08, classes, trainFeatures, type_DA, weak_model, gesture_N,
                        gesture_mean, gesture_cov, threshold=0.8)
                    model = LDA_state_art_threshold_08
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_threshold' + str(0.9)
                    LDA_state_art_threshold_09, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_threshold(
                        LDA_state_art_threshold_09, classes, trainFeatures, type_DA, weak_model, gesture_N,
                        gesture_mean, gesture_cov, threshold=0.9)
                    model = LDA_state_art_threshold_09
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    ################################################### QDA ########################################
                    type_DA = 'QDA'

                    ############################# OURS QDA

                    name = type_DA + '_' + 'ours_soft_labels'
                    QDA_ours_soft_labels, results.at[
                        idx, 'time_update' + '_' + name], results.at[
                        idx, 'time_weight' + '_' + name], weight_vector = models.model_ours_soft_labels(
                        QDA_ours_soft_labels, classes, trainFeatures, labeledGesturesFeatures, labeledGesturesLabels,
                        type_DA, weak_model, gesture_N, gesture_mean, gesture_cov)
                    model = QDA_ours_soft_labels
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)
                    model = QDA_ours_soft_labels
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_labels'
                    QDA_ours_labels, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_labels(
                        QDA_ours_labels, classes, rand[1], type_DA, gesture_N, gesture_mean, gesture_cov)
                    model = QDA_ours_labels
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    #### Probs with our QDA classifier

                    name = type_DA + '_' + 'ours_probs_' + str(0.1)
                    QDA_ours_probs_01, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                        QDA_ours_probs_01, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.1)
                    model = QDA_ours_probs_01
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_probs_' + str(0.2)
                    QDA_ours_probs_02, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                        QDA_ours_probs_02, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.2)
                    model = QDA_ours_probs_02
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_probs_' + str(0.3)
                    QDA_ours_probs_03, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                        QDA_ours_probs_03, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.3)
                    model = QDA_ours_probs_03
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_probs_' + str(0.4)
                    QDA_ours_probs_04, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                        QDA_ours_probs_04, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.4)
                    model = QDA_ours_probs_04
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_probs_' + str(0.5)
                    QDA_ours_probs_05, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                        QDA_ours_probs_05, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.5)
                    model = QDA_ours_probs_05
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_probs_' + str(0.6)
                    QDA_ours_probs_06, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                        QDA_ours_probs_06, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.6)
                    model = QDA_ours_probs_06
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_probs_' + str(0.7)
                    QDA_ours_probs_07, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                        QDA_ours_probs_07, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.7)
                    model = QDA_ours_probs_07
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_probs_' + str(0.8)
                    QDA_ours_probs_08, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                        QDA_ours_probs_08, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.8)
                    model = QDA_ours_probs_08
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_probs_' + str(0.9)
                    QDA_ours_probs_09, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                        QDA_ours_probs_09, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.9)
                    model = QDA_ours_probs_09
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_probs_' + str(1.0)
                    QDA_ours_probs_10, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                        QDA_ours_probs_10, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=1.0)
                    model = QDA_ours_probs_10
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    #### Threshold with our QDA classifier

                    name = type_DA + '_' + 'ours_threshold_' + str(0.0)
                    QDA_ours_threshold_00, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_threshold(
                        QDA_ours_threshold_00, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, threshold=0.0)
                    model = QDA_ours_threshold_00
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_threshold_' + str(0.1)
                    QDA_ours_threshold_01, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_threshold(
                        QDA_ours_threshold_01, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, threshold=0.1)
                    model = QDA_ours_threshold_01
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_threshold_' + str(0.2)
                    QDA_ours_threshold_02, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_threshold(
                        QDA_ours_threshold_02, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, threshold=0.2)
                    model = QDA_ours_threshold_02
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_threshold_' + str(0.3)
                    QDA_ours_threshold_03, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_threshold(
                        QDA_ours_threshold_03, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, threshold=0.3)
                    model = QDA_ours_threshold_03
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_threshold_' + str(0.4)
                    QDA_ours_threshold_04, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_threshold(
                        QDA_ours_threshold_04, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, threshold=0.4)
                    model = QDA_ours_threshold_04
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_threshold_' + str(0.5)
                    QDA_ours_threshold_05, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_threshold(
                        QDA_ours_threshold_05, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, threshold=0.5)
                    model = QDA_ours_threshold_05
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_threshold_' + str(0.6)
                    QDA_ours_threshold_06, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_threshold(
                        QDA_ours_threshold_06, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, threshold=0.6)
                    model = QDA_ours_threshold_06
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_threshold_' + str(0.7)
                    QDA_ours_threshold_07, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_threshold(
                        QDA_ours_threshold_07, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, threshold=0.7)
                    model = QDA_ours_threshold_07
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_threshold_' + str(0.8)
                    QDA_ours_threshold_08, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_threshold(
                        QDA_ours_threshold_08, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, threshold=0.8)
                    model = QDA_ours_threshold_08
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'ours_threshold_' + str(0.9)
                    QDA_ours_threshold_09, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_ours_threshold(
                        QDA_ours_threshold_09, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, threshold=0.9)
                    model = QDA_ours_threshold_09
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    ############################# STATE OF THE ART QDA

                    name = type_DA + '_' + 'state_art_soft_labels'
                    QDA_state_art_soft_labels, results.at[
                        idx, 'time_update' + '_' + name], results.at[
                        idx, 'time_weight' + '_' + name], weight_vector = models.model_state_art_soft_labels(
                        QDA_state_art_soft_labels, classes, trainFeatures, labeledGesturesFeatures,
                        labeledGesturesLabels,
                        type_DA, weak_model, gesture_N, gesture_mean, gesture_cov)
                    model = QDA_state_art_soft_labels
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_labels'
                    QDA_state_art_labels, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_labels(
                        QDA_state_art_labels, classes, rand[1], type_DA, gesture_N, gesture_mean, gesture_cov)
                    model = QDA_state_art_labels
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    #### Probs with state_art QDA classifier

                    name = type_DA + '_' + 'state_art_probs' + str(0.1)
                    QDA_state_art_probs_01, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_probabilities(
                        QDA_state_art_probs_01, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.1)
                    model = QDA_state_art_probs_01
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_probs' + str(0.2)
                    QDA_state_art_probs_02, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_probabilities(
                        QDA_state_art_probs_02, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.2)
                    model = QDA_state_art_probs_02
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_probs' + str(0.3)
                    QDA_state_art_probs_03, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_probabilities(
                        QDA_state_art_probs_03, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.3)
                    model = QDA_state_art_probs_03
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_probs' + str(0.4)
                    QDA_state_art_probs_04, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_probabilities(
                        QDA_state_art_probs_04, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.4)
                    model = QDA_state_art_probs_04
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_probs' + str(0.5)
                    QDA_state_art_probs_05, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_probabilities(
                        QDA_state_art_probs_05, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.5)
                    model = QDA_state_art_probs_05
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_probs' + str(0.6)
                    QDA_state_art_probs_06, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_probabilities(
                        QDA_state_art_probs_06, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.6)
                    model = QDA_state_art_probs_06
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_probs' + str(0.7)
                    QDA_state_art_probs_07, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_probabilities(
                        QDA_state_art_probs_07, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.7)
                    model = QDA_state_art_probs_07
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_probs' + str(0.8)
                    QDA_state_art_probs_08, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_probabilities(
                        QDA_state_art_probs_08, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.8)
                    model = QDA_state_art_probs_08
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_probs' + str(0.9)
                    QDA_state_art_probs_09, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_probabilities(
                        QDA_state_art_probs_09, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=0.9)
                    model = QDA_state_art_probs_09
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_probs' + str(1.0)
                    QDA_state_art_probs_10, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_probabilities(
                        QDA_state_art_probs_10, classes, trainFeatures, type_DA, weak_model, gesture_N, gesture_mean,
                        gesture_cov, weight=1.0)
                    model = QDA_state_art_probs_10
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    #### Threshold with our QDA classifier

                    name = type_DA + '_' + 'state_art_threshold' + str(0.0)
                    QDA_state_art_threshold_00, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_threshold(
                        QDA_state_art_threshold_00, classes, trainFeatures, type_DA, weak_model, gesture_N,
                        gesture_mean, gesture_cov, threshold=0.0)
                    model = QDA_state_art_threshold_00
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_threshold' + str(0.1)
                    QDA_state_art_threshold_01, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_threshold(
                        QDA_state_art_threshold_01, classes, trainFeatures, type_DA, weak_model, gesture_N,
                        gesture_mean, gesture_cov, threshold=0.1)
                    model = QDA_state_art_threshold_01
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_threshold' + str(0.2)
                    QDA_state_art_threshold_02, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_threshold(
                        QDA_state_art_threshold_02, classes, trainFeatures, type_DA, weak_model, gesture_N,
                        gesture_mean, gesture_cov, threshold=0.2)
                    model = QDA_state_art_threshold_02
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_threshold' + str(0.3)
                    QDA_state_art_threshold_03, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_threshold(
                        QDA_state_art_threshold_03, classes, trainFeatures, type_DA, weak_model, gesture_N,
                        gesture_mean, gesture_cov, threshold=0.3)
                    model = QDA_state_art_threshold_03
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_threshold' + str(0.4)
                    QDA_state_art_threshold_04, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_threshold(
                        QDA_state_art_threshold_04, classes, trainFeatures, type_DA, weak_model, gesture_N,
                        gesture_mean, gesture_cov, threshold=0.4)
                    model = QDA_state_art_threshold_04
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_threshold' + str(0.5)
                    QDA_state_art_threshold_05, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_threshold(
                        QDA_state_art_threshold_05, classes, trainFeatures, type_DA, weak_model, gesture_N,
                        gesture_mean, gesture_cov, threshold=0.5)
                    model = QDA_state_art_threshold_05
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_threshold' + str(0.6)
                    QDA_state_art_threshold_06, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_threshold(
                        QDA_state_art_threshold_06, classes, trainFeatures, type_DA, weak_model, gesture_N,
                        gesture_mean, gesture_cov, threshold=0.6)
                    model = QDA_state_art_threshold_06
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_threshold' + str(0.7)
                    QDA_state_art_threshold_07, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_threshold(
                        QDA_state_art_threshold_07, classes, trainFeatures, type_DA, weak_model, gesture_N,
                        gesture_mean, gesture_cov, threshold=0.7)
                    model = QDA_state_art_threshold_07
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_threshold' + str(0.8)
                    QDA_state_art_threshold_08, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_threshold(
                        QDA_state_art_threshold_08, classes, trainFeatures, type_DA, weak_model, gesture_N,
                        gesture_mean, gesture_cov, threshold=0.8)
                    model = QDA_state_art_threshold_08
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'state_art_threshold' + str(0.9)
                    QDA_state_art_threshold_09, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_state_art_threshold(
                        QDA_state_art_threshold_09, classes, trainFeatures, type_DA, weak_model, gesture_N,
                        gesture_mean, gesture_cov, threshold=0.9)
                    model = QDA_state_art_threshold_09
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    results.at[idx, 'Feature Set'] = featureSet
                    results.at[idx, 'person'] = person
                    results.at[idx, 'exp_time'] = seed
                    results.at[idx, 'shots in traning'] = initialSamples
                    results.at[idx, 'day'] = day
                    results.at[idx, 'shot_class'] = rand
                    results.at[idx, 'number gesture'] = numberGestures

                    numberGestures += 1


def update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures, testGesturesLabels, model,
                   classes, type_DA, reported_points, numberGestures, printR, nameFile):
    results.at[idx, 'error_1_weight' + '_' + name] = models.errorWeights_type1(
        weight_vector, true_weight)
    results.at[idx, 'error_2_weight' + '_' + name] = models.errorWeights_type2(
        weight_vector, true_weight)
    if np.any(reported_points == numberGestures):
        if type_DA == 'LDA':
            results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                idx, 'recall_' + name] = DA_classifiers.accuracyModelLDA(
                testGesturesFeatures, testGesturesLabels, model, classes)
        elif type_DA == 'QDA':
            results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                idx, 'recall_' + name] = DA_classifiers.accuracyModelQDA(
                testGesturesFeatures, testGesturesLabels, model, classes)
        if printR:
            for metric in ['acc']:
                for type_DA in ['LDA', 'QDA']:
                    for classifier in ['weak_model', 'batch_model']:
                        name = metric + '_' + type_DA + '_' + classifier
                        print(name, results.loc[idx, name])

            for type_model in ['ours', 'state_art']:
                for metric in ['acc']:
                    for type_DA in ['LDA', 'QDA']:
                        for classifier in ['soft_labels', 'labels']:
                            name = metric + '_' + type_DA + '_' + type_model + '_' + classifier
                            print(name, results.loc[idx, name])

        if nameFile is not None:
            results.to_csv(nameFile)

    idx += 1

    return results


def update_model(name, results, idx, method, model, classes, trainFeatures, labeledGesturesFeatures,
                 labeledGesturesLabels, type_DA, weak_model, gesture_N, gesture_mean, gesture_cov, true_weight):
    w = np.zeros(classes)
    exec(
        model + ", results.at[idx, 'time_update' + '_' + name],results.at[idx, 'time_weight' + '_' + name], " +
        "w = models." + method + "(" + model + ", classes, trainFeatures, " +
        "labeledGesturesFeatures, labeledGesturesLabels,type_DA, weak_model, gesture_N, gesture_mean, gesture_cov)")

    results.at[idx, 'error_1_weight' + '_' + name] = models.errorWeights_type1(w, true_weight)
    results.at[idx, 'error_2_weight' + '_' + name] = models.errorWeights_type2(w, true_weight)


def getGestureSet(dataMatrix, allFeatures, setGestures, person):
    Features = []
    Labels = []
    for Lgesture in setGestures:
        Features += list(dataMatrix[
                         (dataMatrix[:, allFeatures + 1] == person) &
                         (dataMatrix[:, allFeatures] == Lgesture[2]) &
                         (dataMatrix[:, allFeatures + 3] == Lgesture[0]) &
                         (dataMatrix[:, allFeatures + 2] == Lgesture[1]), 0:allFeatures])
        Labels += list(dataMatrix[
                           (dataMatrix[:, allFeatures + 1] == person) &
                           (dataMatrix[:, allFeatures] == Lgesture[2]) &
                           (dataMatrix[:, allFeatures + 3] == Lgesture[0]) &
                           (dataMatrix[:, allFeatures + 2] == Lgesture[1]), allFeatures + 2].T)

    return np.array(Features), np.array(Labels)


def currentDistributionValues(trainFeatures, trainLabels, classes, allFeatures):
    t = time.time()
    currentValues = pd.DataFrame(
        columns=['cov', 'mean', 'class', 'N', 'N_cov', 'LDAcov', 'N_LDA'])
    trainLabelsAux = trainLabels[np.newaxis]
    Matrix = np.hstack((trainFeatures, trainLabelsAux.T))

    for cla in range(classes):
        X = Matrix[np.where((Matrix[:, allFeatures] == cla + 1)), 0:allFeatures][0]
        currentValues.at[cla, 'mean'] = DA_classifiers.mean(X, allFeatures)
        currentValues.at[cla, 'cov'] = DA_classifiers.covariance(X, allFeatures, currentValues.loc[cla, 'mean'])
        currentValues.at[cla, 'class'] = cla + 1
        N = np.size(X, axis=0)
        currentValues.at[cla, 'N'] = N
    timeQDA = time.time() - t
    currentValues.at[0, 'N_total'] = np.size(Matrix[:, 0:allFeatures], axis=0)
    currentValues.at[0, 'LDAcov'] = DA_classifiers.LDA_Cov_weights(currentValues)
    timeLDA = time.time() - t
    return currentValues, timeLDA, timeQDA


def systematicSampling(trainFeatures, trainLabels, numSamples, classes):
    x = []
    y = []
    for cla in range(classes):
        xAux = trainFeatures[trainLabels == cla + 1]
        initialRandom = np.random.randint(len(xAux))
        if len(xAux) > numSamples:
            interval = int(np.ceil(len(xAux) / numSamples))
            for _ in range(numSamples):
                x.append(xAux[initialRandom, :])
                y.append(cla + 1)
                initialRandom = (initialRandom + interval) % len(xAux)
        else:
            for _ in range(len(xAux)):
                x.append(xAux[initialRandom, :])
                y.append(cla + 1)
                initialRandom = (initialRandom + 1) % len(xAux)
            print('numSamples greater than samples in training, class: ', cla + 1, numSamples, len(xAux))
    return np.array(x), np.array(y)
