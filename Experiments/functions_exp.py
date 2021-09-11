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
    numSamples = 20

    if Database == 'EPN_120':
        classes = 5
        CH = 8
        numberGesturesTotal = 50
        days = 1
        numberGesturesTest = 25
    elif Database == 'Nina5':
        classes = 18
        CH = 16
        numberGesturesTotal = 6
        days = 1
        numberGesturesTest = 4
    elif Database == 'Cote':
        classes = 7
        CH = 8
        numberGesturesTotal = 12
        days = 1
        numberGesturesTest = 9
    elif Database == 'Capgmyo_dbb':
        classes = 8
        CH = 128
        numberGesturesTotal = 10
        days = 2
        numberGesturesTest = 7
    elif Database == 'LongTerm3DC':
        classes = 11
        CH = 10
        numberGesturesTotal = 3
        days = 3
        numberGesturesTest = 2

    if featureSet == 1:
        # Setting variables
        Feature1 = 'logvar'

        numberFeatures = 1
        allFeatures = numberFeatures * CH
        # Getting Data
        logvarMatrix = np.load(path + 'ExtractedData/' + Database + '/' + Feature1 + segment + personFile + '.npy')

        dataMatrix = logvarMatrix.copy()

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

        dataMatrix = np.hstack((mavMatrix[:, :CH], wlMatrix[:, :CH], zcMatrix[:, :CH], sscMatrix[:, :]))

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

        dataMatrix = np.hstack((lscaleMatrix[:, :CH], mflMatrix[:, :CH], msrMatrix[:, :CH], wampMatrix[:, :]))

    return dataMatrix, classes, numberGesturesTotal, allFeatures, numSamples, days, numberGesturesTest


def evaluation(featureSet, nameFile, startPerson, endPerson, printR, initialSamples, initialTime, finalTime,
               typeDatabase, all_models):
    scaler = preprocessing.MinMaxScaler()

    results = pd.DataFrame(
        columns=['Feature Set', 'person', 'exp_time', 'shots in traning', 'shot_class', 'day', 'number gesture'])

    for metric in ['precision', 'recall']:
        for type_DA in ['LDA', 'QDA']:
            results[metric + '_' + type_DA + '_' + 'weak'] = ''
            results[metric + '_' + type_DA + '_' + 'batch_weighted_soft_labels'] = ''
            results[metric + '_' + type_DA + '_' + 'batch_weighted_labels'] = ''
            results[metric + '_' + type_DA + '_' + 'batch_traditional_soft_labels'] = ''
            results[metric + '_' + type_DA + '_' + 'batch_traditional_labels'] = ''

            results[metric + '_' + type_DA + '_ours_soft_labels'] = ''
            results[metric + '_' + type_DA + '_ours_labels'] = ''
            results[metric + '_' + type_DA + '_traditional_COV_soft_labels'] = ''
            results[metric + '_' + type_DA + '__traditional_COV_labels'] = ''
            for weight in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                results[metric + '_' + type_DA + '_Nigam_' + str(weight)] = ''
            for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                results[metric + '_' + type_DA + '_thresholding_' + str(threshold)] = ''

    idx = 0

    for person in range(startPerson, endPerson + 1):
        # Upload Data

        dataMatrix, classes, numberGesturesTotal, allFeatures, numSamples, days, numberGesturesTest = uploadDatabases(
            typeDatabase, person, featureSet)

        for seed in range(initialTime, finalTime + 1):

            for day in range(1, days + 1):

                print('p', person, 's', seed, 'd', day)
                numberGestures = 1
                np.random.seed(person + seed * 1000 + day * 10000)  # to get a different seed per each person and day

                labeledGestures = []
                unlabeledGestures = []
                testGestures = []
                for cla in range(1, classes + 1):
                    repetitions = np.random.choice(numberGesturesTotal, numberGesturesTotal, replace=False) + 1
                    labeledGestures += [[shot, cla, day] for shot in repetitions[:initialSamples]]
                    unlabeledGestures += [[shot, cla, day] for shot in repetitions[initialSamples:numberGesturesTest]]
                    testGestures += [[shot, cla, day] for shot in repetitions[numberGesturesTest:]]

                permutationUnlabeledGestures = np.random.permutation(unlabeledGestures)
                labeledGesturesFeatures, labeledGesturesLabels = getGestureSet(dataMatrix, allFeatures,
                                                                               labeledGestures, person)
                labeledGesturesFeatures = scaler.fit_transform(labeledGesturesFeatures)

                testGesturesFeatures, testGesturesLabels = getGestureSet(dataMatrix, allFeatures, testGestures,
                                                                         person)
                testGesturesFeatures = scaler.transform(testGesturesFeatures)

                if typeDatabase == 'Capgmyo_dbb':  # dimension reduction using PCA for Capgmyo_dbb that has 128 ch
                    pca_model = PCA(n_components=0.99, svd_solver='full')
                    labeledGesturesFeatures = pca_model.fit_transform(labeledGesturesFeatures)
                    testGesturesFeatures = pca_model.transform(testGesturesFeatures)
                    print('features', np.size(labeledGesturesFeatures, axis=1))

                initial_model, _, _ = models.initial_model(
                    labeledGesturesFeatures, labeledGesturesLabels, classes, np.size(labeledGesturesFeatures, axis=1))



                labeledGesturesFeatures, labeledGesturesLabels = systematicSampling(
                    labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)

                LDA_ours_labels = initial_model.copy()
                LDA_ours_soft_labels = initial_model.copy()
                LDA_batch_weighted_labels = initial_model.copy()
                LDA_batch_weighted_soft_labels = initial_model.copy()
                LDA_batch_traditional_labels = initial_model.copy()
                LDA_batch_traditional_soft_labels = initial_model.copy()
                LDA_Nigam_01 = initial_model.copy()
                LDA_Nigam_02 = initial_model.copy()
                LDA_Nigam_03 = initial_model.copy()
                LDA_Nigam_04 = initial_model.copy()
                LDA_Nigam_05 = initial_model.copy()
                LDA_Nigam_06 = initial_model.copy()
                LDA_Nigam_07 = initial_model.copy()
                LDA_Nigam_08 = initial_model.copy()
                LDA_Nigam_09 = initial_model.copy()
                LDA_Nigam_10 = initial_model.copy()
                LDA_thresholding_00 = initial_model.copy()
                LDA_thresholding_01 = initial_model.copy()
                LDA_thresholding_02 = initial_model.copy()
                LDA_thresholding_03 = initial_model.copy()
                LDA_thresholding_04 = initial_model.copy()
                LDA_thresholding_05 = initial_model.copy()
                LDA_thresholding_06 = initial_model.copy()
                LDA_thresholding_07 = initial_model.copy()
                LDA_thresholding_08 = initial_model.copy()
                LDA_thresholding_09 = initial_model.copy()

                QDA_ours_labels = initial_model.copy()
                QDA_ours_soft_labels = initial_model.copy()
                QDA_batch_weighted_labels = initial_model.copy()
                QDA_batch_weighted_soft_labels = initial_model.copy()
                QDA_batch_traditional_labels = initial_model.copy()
                QDA_batch_traditional_soft_labels = initial_model.copy()
                QDA_Nigam_01 = initial_model.copy()
                QDA_Nigam_02 = initial_model.copy()
                QDA_Nigam_03 = initial_model.copy()
                QDA_Nigam_04 = initial_model.copy()
                QDA_Nigam_05 = initial_model.copy()
                QDA_Nigam_06 = initial_model.copy()
                QDA_Nigam_07 = initial_model.copy()
                QDA_Nigam_08 = initial_model.copy()
                QDA_Nigam_09 = initial_model.copy()
                QDA_Nigam_10 = initial_model.copy()
                QDA_thresholding_00 = initial_model.copy()
                QDA_thresholding_01 = initial_model.copy()
                QDA_thresholding_02 = initial_model.copy()
                QDA_thresholding_03 = initial_model.copy()
                QDA_thresholding_04 = initial_model.copy()
                QDA_thresholding_05 = initial_model.copy()
                QDA_thresholding_06 = initial_model.copy()
                QDA_thresholding_07 = initial_model.copy()
                QDA_thresholding_08 = initial_model.copy()
                QDA_thresholding_09 = initial_model.copy()



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

                    # print('current label', rand[1])
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

                    print('person', person, 'day', day, 'exp', seed, 'gesture', numberGestures)

                    ###### Weak model

                    for type_DA in ['LDA', 'QDA']:
                        name = type_DA + '_' + 'weak'
                        results = update_results(results, idx, name, true_weight, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, initial_model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)


                    ################################# LDA #########################
                    type_DA = 'LDA'

                    # OURS LDA

                    name = type_DA + '_' + 'ours_soft_labels'
                    LDA_ours_soft_labels, results.at[
                        idx, 'time_update' + '_' + name], results.at[
                        idx, 'time_weight' + '_' + name], weight_vector = models.model_ours_soft_labels(
                        LDA_ours_soft_labels, classes, trainFeatures, labeledGesturesFeatures, labeledGesturesLabels,
                        type_DA, initial_model, gesture_N, gesture_mean, gesture_cov)
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

                    # Batch classifiers LDA

                    name = type_DA + '_' + 'batch_weighted_soft_labels'
                    LDA_batch_weighted_soft_labels, results.at[
                        idx, 'time_update' + '_' + name], results.at[
                        idx, 'time_weight' + '_' + name], weight_vector = models.model_batch_weighted_soft_labels(
                        LDA_batch_weighted_soft_labels, classes, trainFeatures, labeledGesturesFeatures, labeledGesturesLabels,
                        type_DA, initial_model, gesture_N, gesture_mean, gesture_cov)
                    model = LDA_ours_soft_labels
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'batch_weighted_labels'
                    LDA_batch_weighted_labels, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_batch_weighted_labels(
                        LDA_batch_weighted_labels, classes, rand[1], type_DA, gesture_N, gesture_mean, gesture_cov)
                    model = LDA_ours_labels
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'batch_traditional_soft_labels'
                    LDA_batch_traditional_soft_labels, results.at[
                        idx, 'time_update' + '_' + name], results.at[
                        idx, 'time_weight' + '_' + name], weight_vector = models.model_batch_traditional_soft_labels(
                        LDA_batch_traditional_soft_labels, classes, trainFeatures, labeledGesturesFeatures,
                        labeledGesturesLabels,
                        type_DA, initial_model, gesture_N, gesture_mean, gesture_cov)
                    model = LDA_ours_soft_labels
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'batch_traditional_labels'
                    LDA_batch_traditional_labels, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_batch_traditional_labels(
                        LDA_batch_traditional_labels, classes, rand[1], type_DA, gesture_N, gesture_mean, gesture_cov)
                    model = LDA_ours_labels
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)


                    if all_models == True:
                        #### Probs with our LDA classifier
                        name = type_DA + '_' + 'Nigam_' + str(0.1)
                        LDA_Nigam_01, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                            LDA_Nigam_01, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, weight=0.1)
                        model = LDA_Nigam_01
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'Nigam_' + str(0.2)
                        LDA_Nigam_02, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                            LDA_Nigam_02, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, weight=0.2)
                        model = LDA_Nigam_02
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'Nigam_' + str(0.3)
                        LDA_Nigam_03, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                            LDA_Nigam_03, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, weight=0.3)
                        model = LDA_Nigam_03
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'Nigam_' + str(0.4)
                        LDA_Nigam_04, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                            LDA_Nigam_04, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, weight=0.4)
                        model = LDA_Nigam_04
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'Nigam_' + str(0.5)
                        LDA_Nigam_05, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                            LDA_Nigam_05, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, weight=0.5)
                        model = LDA_Nigam_05
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'Nigam_' + str(0.6)
                        LDA_Nigam_06, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                            LDA_Nigam_06, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, weight=0.6)
                        model = LDA_Nigam_06
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'Nigam_' + str(0.7)
                        LDA_Nigam_07, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                            LDA_Nigam_07, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, weight=0.7)
                        model = LDA_Nigam_07
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'Nigam_' + str(0.8)
                        LDA_Nigam_08, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                            LDA_Nigam_08, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, weight=0.8)
                        model = LDA_Nigam_08
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'Nigam_' + str(0.9)
                        LDA_Nigam_09, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                            LDA_Nigam_09, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, weight=0.9)
                        model = LDA_Nigam_09
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'Nigam_' + str(1.0)
                        LDA_Nigam_10, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                            LDA_Nigam_10, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, weight=1.0)
                        model = LDA_Nigam_10
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        #### Threshold with our LDA classifier

                        name = type_DA + '_' + 'thresholding_' + str(0.0)
                        LDA_thresholding_00, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_thresholding(
                            LDA_thresholding_00, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, threshold=0.0)
                        model = LDA_thresholding_00
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'thresholding_' + str(0.1)
                        LDA_thresholding_01, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_thresholding(
                            LDA_thresholding_01, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, threshold=0.1)
                        model = LDA_thresholding_01
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'thresholding_' + str(0.2)
                        LDA_thresholding_02, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_thresholding(
                            LDA_thresholding_02, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, threshold=0.2)
                        model = LDA_thresholding_02
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'thresholding_' + str(0.3)
                        LDA_thresholding_03, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_thresholding(
                            LDA_thresholding_03, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, threshold=0.3)
                        model = LDA_thresholding_03
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'thresholding_' + str(0.4)
                        LDA_thresholding_04, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_thresholding(
                            LDA_thresholding_04, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, threshold=0.4)
                        model = LDA_thresholding_04
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'thresholding_' + str(0.5)
                        LDA_thresholding_05, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_thresholding(
                            LDA_thresholding_05, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, threshold=0.5)
                        model = LDA_thresholding_05
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'thresholding_' + str(0.6)
                        LDA_thresholding_06, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_thresholding(
                            LDA_thresholding_06, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, threshold=0.6)
                        model = LDA_thresholding_06
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'thresholding_' + str(0.7)
                        LDA_thresholding_07, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_thresholding(
                            LDA_thresholding_07, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, threshold=0.7)
                        model = LDA_thresholding_07
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'thresholding_' + str(0.8)
                        LDA_thresholding_08, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_thresholding(
                            LDA_thresholding_08, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, threshold=0.8)
                        model = LDA_thresholding_08
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'thresholding_' + str(0.9)
                        LDA_thresholding_09, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_thresholding(
                            LDA_thresholding_09, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, threshold=0.9)
                        model = LDA_thresholding_09
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
                        type_DA, initial_model, gesture_N, gesture_mean, gesture_cov)
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

                    # Batch classifiers QDA

                    name = type_DA + '_' + 'batch_weighted_soft_labels'
                    QDA_batch_weighted_soft_labels, results.at[
                        idx, 'time_update' + '_' + name], results.at[
                        idx, 'time_weight' + '_' + name], weight_vector = models.model_batch_weighted_soft_labels(
                        QDA_batch_weighted_soft_labels, classes, trainFeatures, labeledGesturesFeatures,
                        labeledGesturesLabels,
                        type_DA, initial_model, gesture_N, gesture_mean, gesture_cov)
                    model = QDA_ours_soft_labels
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'batch_weighted_labels'
                    QDA_batch_weighted_labels, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_batch_weighted_labels(
                        QDA_batch_weighted_labels, classes, rand[1], type_DA, gesture_N, gesture_mean, gesture_cov)
                    model = QDA_ours_labels
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'batch_traditional_soft_labels'
                    QDA_batch_traditional_soft_labels, results.at[
                        idx, 'time_update' + '_' + name], results.at[
                        idx, 'time_weight' + '_' + name], weight_vector = models.model_batch_traditional_soft_labels(
                        QDA_batch_traditional_soft_labels, classes, trainFeatures, labeledGesturesFeatures,
                        labeledGesturesLabels,
                        type_DA, initial_model, gesture_N, gesture_mean, gesture_cov)
                    model = QDA_ours_soft_labels
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    name = type_DA + '_' + 'batch_traditional_labels'
                    QDA_batch_traditional_labels, results.at[
                        idx, 'time_update' + '_' + name], weight_vector = models.model_batch_traditional_labels(
                        QDA_batch_traditional_labels, classes, rand[1], type_DA, gesture_N, gesture_mean, gesture_cov)
                    model = QDA_ours_labels
                    results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                             testGesturesLabels, model, classes, type_DA, reported_points,
                                             numberGestures, printR, nameFile)

                    if all_models == True:
                        #### Probs with our QDA classifier

                        name = type_DA + '_' + 'Nigam_' + str(0.1)
                        QDA_Nigam_01, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                            QDA_Nigam_01, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, weight=0.1)
                        model = QDA_Nigam_01
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'Nigam_' + str(0.2)
                        QDA_Nigam_02, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                            QDA_Nigam_02, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, weight=0.2)
                        model = QDA_Nigam_02
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'Nigam_' + str(0.3)
                        QDA_Nigam_03, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                            QDA_Nigam_03, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, weight=0.3)
                        model = QDA_Nigam_03
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'Nigam_' + str(0.4)
                        QDA_Nigam_04, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                            QDA_Nigam_04, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, weight=0.4)
                        model = QDA_Nigam_04
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'Nigam_' + str(0.5)
                        QDA_Nigam_05, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                            QDA_Nigam_05, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, weight=0.5)
                        model = QDA_Nigam_05
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'Nigam_' + str(0.6)
                        QDA_Nigam_06, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                            QDA_Nigam_06, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, weight=0.6)
                        model = QDA_Nigam_06
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'Nigam_' + str(0.7)
                        QDA_Nigam_07, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                            QDA_Nigam_07, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, weight=0.7)
                        model = QDA_Nigam_07
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'Nigam_' + str(0.8)
                        QDA_Nigam_08, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                            QDA_Nigam_08, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, weight=0.8)
                        model = QDA_Nigam_08
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'Nigam_' + str(0.9)
                        QDA_Nigam_09, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                            QDA_Nigam_09, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, weight=0.9)
                        model = QDA_Nigam_09
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'Nigam_' + str(1.0)
                        QDA_Nigam_10, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_ours_probabilities(
                            QDA_Nigam_10, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, weight=1.0)
                        model = QDA_Nigam_10
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        #### Threshold with our QDA classifier

                        name = type_DA + '_' + 'thresholding_' + str(0.0)
                        QDA_thresholding_00, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_thresholding(
                            QDA_thresholding_00, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, threshold=0.0)
                        model = QDA_thresholding_00
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'thresholding_' + str(0.1)
                        QDA_thresholding_01, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_thresholding(
                            QDA_thresholding_01, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, threshold=0.1)
                        model = QDA_thresholding_01
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'thresholding_' + str(0.2)
                        QDA_thresholding_02, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_thresholding(
                            QDA_thresholding_02, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, threshold=0.2)
                        model = QDA_thresholding_02
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'thresholding_' + str(0.3)
                        QDA_thresholding_03, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_thresholding(
                            QDA_thresholding_03, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, threshold=0.3)
                        model = QDA_thresholding_03
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'thresholding_' + str(0.4)
                        QDA_thresholding_04, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_thresholding(
                            QDA_thresholding_04, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, threshold=0.4)
                        model = QDA_thresholding_04
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'thresholding_' + str(0.5)
                        QDA_thresholding_05, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_thresholding(
                            QDA_thresholding_05, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, threshold=0.5)
                        model = QDA_thresholding_05
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'thresholding_' + str(0.6)
                        QDA_thresholding_06, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_thresholding(
                            QDA_thresholding_06, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, threshold=0.6)
                        model = QDA_thresholding_06
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'thresholding_' + str(0.7)
                        QDA_thresholding_07, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_thresholding(
                            QDA_thresholding_07, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, threshold=0.7)
                        model = QDA_thresholding_07
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'thresholding_' + str(0.8)
                        QDA_thresholding_08, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_thresholding(
                            QDA_thresholding_08, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, threshold=0.8)
                        model = QDA_thresholding_08
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)

                        name = type_DA + '_' + 'thresholding_' + str(0.9)
                        QDA_thresholding_09, results.at[
                            idx, 'time_update' + '_' + name], weight_vector = models.model_thresholding(
                            QDA_thresholding_09, classes, trainFeatures, type_DA, initial_model, gesture_N, gesture_mean,
                            gesture_cov, threshold=0.9)
                        model = QDA_thresholding_09
                        results = update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures,
                                                 testGesturesLabels, model, classes, type_DA, reported_points,
                                                 numberGestures, printR, nameFile)


                    if reported_points[-1] == numberGestures:

                        results.at[idx, 'Feature Set'] = featureSet
                        results.at[idx, 'person'] = person
                        results.at[idx, 'exp_time'] = seed
                        results.at[idx, 'shots in traning'] = initialSamples
                        results.at[idx, 'day'] = day
                        results.at[idx, 'shot_class'] = rand
                        results.at[idx, 'number gesture'] = numberGestures

                        if nameFile is not None:
                            results.to_csv(nameFile)

                        if printR:

                            for type_DA in ['LDA', 'QDA']:
                                for metric in ['acc']:
                                    for classifier in ['weak', 'batch']:
                                        name = metric + '_' + type_DA + '_' + classifier
                                        print(name, results.loc[idx, name])

                                    for classifier in ['soft_labels', 'labels']:
                                        for type_model in ['ours', 'state_art']:
                                            name = metric + '_' + type_DA + '_' + type_model + '_' + classifier
                                            print(name, results.loc[idx, name])
                        idx += 1
                    numberGestures += 1


def update_results(results, idx, name, weight_vector, true_weight, testGesturesFeatures, testGesturesLabels, model,
                   classes, type_DA, reported_points, numberGestures, printR, nameFile):
    if numberGestures == 1:
        results.at[idx, 'error_1_weight' + '_' + name] = models.errorWeights_type1(
            weight_vector, true_weight)
        results.at[idx, 'error_2_weight' + '_' + name] = models.errorWeights_type2(
            weight_vector, true_weight)
    else:
        results.at[idx, 'error_1_weight' + '_' + name] += models.errorWeights_type1(
            weight_vector, true_weight)
        results.at[idx, 'error_2_weight' + '_' + name] += models.errorWeights_type2(
            weight_vector, true_weight)
    if reported_points[-1] == numberGestures:
        if type_DA == 'LDA':
            results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                idx, 'recall_' + name] = DA_classifiers.accuracyModelLDA(
                testGesturesFeatures, testGesturesLabels, model, classes)
        elif type_DA == 'QDA':
            results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                idx, 'recall_' + name] = DA_classifiers.accuracyModelQDA(
                testGesturesFeatures, testGesturesLabels, model, classes)
    return results



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
