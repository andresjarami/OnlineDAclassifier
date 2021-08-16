import math

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

import DA_Classifiers as DA_Classifiers
import SemiSupervised as SemiSupervised


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
        numSamples = 140  # to holds real-time
        days = 1
        testSamples = 25
    elif Database == 'Nina5':
        classes = 18
        CH = 16
        numberShots = 6
        numSamples = 17  # to holds real-time
        days = 1
        testSamples = 4
    elif Database == 'Cote':
        classes = 7
        CH = 8
        numberShots = 12
        # numSamples = 120  # to holds real-time
        numSamples = 20  # to holds real-time
        days = 1
        testSamples = 9
    elif Database == 'Capgmyo_dbb':
        classes = 8
        CH = 128
        numberShots = 10
        numSamples = 20  # to holds real-time
        days = 2
        testSamples = 7
    elif Database == 'LongTerm3DC':
        classes = 11
        CH = 10
        numberShots = 3
        numSamples = 20  # to holds real-time
        days = 3
        testSamples = 2

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

    return dataMatrix, classes, numberShots, allFeatures, numSamples, days, testSamples


def evaluation_notest(featureSet, nameFile, startPerson, endPerson, printR, initialSamples, initialTime, finalTime,
                      typeDatabase):
    scaler = preprocessing.MinMaxScaler()

    results = pd.DataFrame(
        columns=['Feature Set', 'person', 'exp_time', 'shots in traning', 'shot_class', 'day', 'unlabeled Gesture'])

    for metric in ['w_predicted', 'time', 'precision', 'recall']:
        for typeDA in ['LDA', 'QDA']:
            results[metric + '_' + typeDA + '_' + 'weak'] = ''
            results[metric + '_' + typeDA + '_' + 'incre_proposed'] = ''
            results[metric + '_' + typeDA + '_' + 'incre_labels'] = ''
            results[metric + '_' + typeDA + '_' + 'incre_sequential'] = ''
            results[metric + '_' + typeDA + '_' + 'incre_sequential_probs'] = ''
            results[metric + '_' + typeDA + '_' + 'incre_supervised'] = ''
            for l in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
                results[metric + '_' + typeDA + '_' + 'incre_Nigam_' + str(l)] = ''
            for l in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                results[metric + '_' + typeDA + '_' + 'incre_threshold_' + str(l)] = ''

    idx = 0

    for person in range(startPerson, endPerson + 1):
        # Upload Data
        dataMatrix, classes, numberShots, allFeatures, numSamples, days, testSamples = uploadDatabases(
            typeDatabase, person, featureSet)

        for seed in range(initialTime, finalTime + 1):
            for day in range(1, days + 1):
                numberUnlabeledGestures = 1
                np.random.seed(person + seed * 1000 + day * 10000)  # to get a different seed per each person

                labeledGestures = []
                unlabeledGestures = []
                testGestures = []
                for cla in range(1, classes + 1):
                    repetitions = np.random.choice(numberShots, numberShots, replace=False) + 1
                    labeledGestures += [[shot, cla, day] for shot in repetitions[:initialSamples]]
                    unlabeledGestures += [[shot, cla, day] for shot in repetitions[initialSamples:]]

                permutationUnlabeledGestures = np.random.permutation(unlabeledGestures)
                print(labeledGestures)
                print(unlabeledGestures)
                print(list(permutationUnlabeledGestures))

                labeledGesturesFeatures, labeledGesturesLabels = getGestureSet(dataMatrix, allFeatures,
                                                                               labeledGestures, person)
                labeledGesturesFeatures = scaler.fit_transform(labeledGesturesFeatures)

                if typeDatabase == 'Capgmyo_dbb':
                    pca_model = PCA(n_components=0.99, svd_solver='full')
                    labeledGesturesFeatures = pca_model.fit_transform(labeledGesturesFeatures)
                    print('features', np.size(labeledGesturesFeatures, axis=1))

                weakModel = currentDistributionValues(
                    labeledGesturesFeatures, labeledGesturesLabels, classes, np.size(labeledGesturesFeatures, axis=1),
                    oldData=False)

                labeledGesturesFeatures, labeledGesturesLabels = systematicSampling(
                    labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)
                print('number labeled samples', len(labeledGesturesLabels) / classes)

                LDA_weak = weakModel.copy()
                LDA_incre_proposed = weakModel.copy()
                LDA_incre_proposed2 = weakModel.copy()
                LDA_incre_proposed3 = weakModel.copy()
                LDA_incre_label = weakModel.copy()
                LDA_incre_sequential = weakModel.copy()
                LDA_incre_sequential_probs = weakModel.copy()
                LDA_incre_supervised = weakModel.copy()
                LDA_incre_Nigam_01 = weakModel.copy()
                LDA_incre_Nigam_02 = weakModel.copy()
                LDA_incre_Nigam_04 = weakModel.copy()
                LDA_incre_Nigam_06 = weakModel.copy()
                LDA_incre_Nigam_08 = weakModel.copy()
                LDA_incre_Nigam_10 = weakModel.copy()
                LDA_incre_threshold_02 = weakModel.copy()
                LDA_incre_threshold_03 = weakModel.copy()
                LDA_incre_threshold_04 = weakModel.copy()
                LDA_incre_threshold_05 = weakModel.copy()
                LDA_incre_threshold_06 = weakModel.copy()
                LDA_incre_threshold_07 = weakModel.copy()
                LDA_incre_threshold_08 = weakModel.copy()
                LDA_incre_threshold_09 = weakModel.copy()

                QDA_weak = weakModel.copy()
                QDA_incre_proposed = weakModel.copy()
                QDA_incre_proposed2 = weakModel.copy()
                QDA_incre_proposed3 = weakModel.copy()
                QDA_incre_label = weakModel.copy()
                QDA_incre_sequential = weakModel.copy()
                QDA_incre_sequential_probs = weakModel.copy()
                QDA_incre_supervised = weakModel.copy()
                QDA_incre_Nigam_01 = weakModel.copy()
                QDA_incre_Nigam_02 = weakModel.copy()
                QDA_incre_Nigam_04 = weakModel.copy()
                QDA_incre_Nigam_06 = weakModel.copy()
                QDA_incre_Nigam_08 = weakModel.copy()
                QDA_incre_Nigam_10 = weakModel.copy()
                QDA_incre_threshold_02 = weakModel.copy()
                QDA_incre_threshold_03 = weakModel.copy()
                QDA_incre_threshold_04 = weakModel.copy()
                QDA_incre_threshold_05 = weakModel.copy()
                QDA_incre_threshold_06 = weakModel.copy()
                QDA_incre_threshold_07 = weakModel.copy()
                QDA_incre_threshold_08 = weakModel.copy()
                QDA_incre_threshold_09 = weakModel.copy()

                CM_LDA_weak = np.zeros((classes, classes))
                CM_LDA_incre_proposed = np.zeros((classes, classes))
                CM_LDA_incre_proposed2 = np.zeros((classes, classes))
                CM_LDA_incre_proposed3 = np.zeros((classes, classes))
                CM_LDA_incre_label = np.zeros((classes, classes))
                CM_LDA_incre_sequential = np.zeros((classes, classes))
                CM_LDA_incre_sequential_probs = np.zeros((classes, classes))
                CM_LDA_incre_supervised = np.zeros((classes, classes))
                CM_LDA_incre_Nigam_01 = np.zeros((classes, classes))
                CM_LDA_incre_Nigam_02 = np.zeros((classes, classes))
                CM_LDA_incre_Nigam_04 = np.zeros((classes, classes))
                CM_LDA_incre_Nigam_06 = np.zeros((classes, classes))
                CM_LDA_incre_Nigam_08 = np.zeros((classes, classes))
                CM_LDA_incre_Nigam_10 = np.zeros((classes, classes))
                CM_LDA_incre_threshold_02 = np.zeros((classes, classes))
                CM_LDA_incre_threshold_03 = np.zeros((classes, classes))
                CM_LDA_incre_threshold_04 = np.zeros((classes, classes))
                CM_LDA_incre_threshold_05 = np.zeros((classes, classes))
                CM_LDA_incre_threshold_06 = np.zeros((classes, classes))
                CM_LDA_incre_threshold_07 = np.zeros((classes, classes))
                CM_LDA_incre_threshold_08 = np.zeros((classes, classes))
                CM_LDA_incre_threshold_09 = np.zeros((classes, classes))

                CM_QDA_weak = np.zeros((classes, classes))
                CM_QDA_incre_proposed = np.zeros((classes, classes))
                CM_QDA_incre_proposed2 = np.zeros((classes, classes))
                CM_QDA_incre_proposed3 = np.zeros((classes, classes))
                CM_QDA_incre_label = np.zeros((classes, classes))
                CM_QDA_incre_sequential = np.zeros((classes, classes))
                CM_QDA_incre_sequential_probs = np.zeros((classes, classes))
                CM_QDA_incre_supervised = np.zeros((classes, classes))
                CM_QDA_incre_Nigam_01 = np.zeros((classes, classes))
                CM_QDA_incre_Nigam_02 = np.zeros((classes, classes))
                CM_QDA_incre_Nigam_04 = np.zeros((classes, classes))
                CM_QDA_incre_Nigam_06 = np.zeros((classes, classes))
                CM_QDA_incre_Nigam_08 = np.zeros((classes, classes))
                CM_QDA_incre_Nigam_10 = np.zeros((classes, classes))
                CM_QDA_incre_threshold_02 = np.zeros((classes, classes))
                CM_QDA_incre_threshold_03 = np.zeros((classes, classes))
                CM_QDA_incre_threshold_04 = np.zeros((classes, classes))
                CM_QDA_incre_threshold_05 = np.zeros((classes, classes))
                CM_QDA_incre_threshold_06 = np.zeros((classes, classes))
                CM_QDA_incre_threshold_07 = np.zeros((classes, classes))
                CM_QDA_incre_threshold_08 = np.zeros((classes, classes))
                CM_QDA_incre_threshold_09 = np.zeros((classes, classes))

                # iauxttt = 0
                for rand in list(permutationUnlabeledGestures):
                    trainFeatures = dataMatrix[
                                    (dataMatrix[:, allFeatures] == rand[2]) & (
                                            dataMatrix[:, allFeatures + 1] == person) & (
                                            dataMatrix[:, allFeatures + 3] == rand[0]) & (
                                            dataMatrix[:, allFeatures + 2] == rand[1]), 0:allFeatures]
                    # trainLabels = dataMatrix[
                    #     (dataMatrix[:, allFeatures] == rand[2]) & (dataMatrix[:, allFeatures + 1] == person) & (
                    #             dataMatrix[:, allFeatures + 3] == rand[0]) & (
                    #             dataMatrix[:, allFeatures + 2] == rand[1]), allFeatures + 2]

                    # print('\nreal label',rand[1])

                    #####Pre-processing
                    trainFeatures = scaler.transform(trainFeatures)
                    if typeDatabase == 'Capgmyo_dbb':
                        trainFeatures = pca_model.transform(trainFeatures)

                    # if iauxttt == 0:
                    #     accumLabeledGesturesFeatures = trainFeatures
                    #     accumLabeledGesturesLabels = trainLabels
                    #     iauxttt = 1
                    # else:
                    #     accumLabeledGesturesFeatures = np.vstack((accumLabeledGesturesFeatures, trainFeatures))
                    #     accumLabeledGesturesLabels = np.hstack((accumLabeledGesturesLabels, trainLabels))

                    type_DA = 'LDA'

                    name = type_DA + '_' + 'weak'
                    CM_LDA_weak[rand[1] - 1, :] += SemiSupervised.predicted_labels(trainFeatures, LDA_weak, classes,
                                                                                   type_DA)

                    name = type_DA + '_' + 'incre_proposed'
                    LDA_incre_proposed, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposed(
                        LDA_incre_proposed, classes, trainFeatures, rand[1], labeledGesturesFeatures,
                        labeledGesturesLabels, type_DA, CM_LDA_incre_proposed, weakModel)
                    # print('time LDA', results.at[idx, 'time' + '_' + name])

                    name = type_DA + '_' + 'incre_proposed2'
                    LDA_incre_proposed2, _, _ = SemiSupervised.model_incre_proposed2(
                        LDA_incre_proposed2, classes, trainFeatures, rand[1], labeledGesturesFeatures,
                        labeledGesturesLabels, type_DA, CM_LDA_incre_proposed2, weakModel)

                    name = type_DA + '_' + 'incre_proposed3'
                    LDA_incre_proposed3, _, _ = SemiSupervised.model_incre_proposed3(
                        LDA_incre_proposed3, classes, trainFeatures, rand[1], labeledGesturesFeatures,
                        labeledGesturesLabels, type_DA, CM_LDA_incre_proposed3, weakModel)

                    name = type_DA + '_' + 'incre_labels'
                    LDA_incre_label, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_labels(
                        LDA_incre_label, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_label, weakModel)

                    name = type_DA + '_' + 'incre_sequential'
                    LDA_incre_sequential, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_sequential_labels(
                        LDA_incre_sequential, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_sequential,
                        weakModel)

                    name = type_DA + '_' + 'incre_sequential_probs'
                    LDA_incre_sequential_probs, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_sequential_probs(
                        LDA_incre_sequential_probs, classes, trainFeatures, rand[1], type_DA,
                        CM_LDA_incre_sequential_probs, weakModel)

                    name = type_DA + '_' + 'incre_supervised'
                    LDA_incre_supervised, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_supervised(
                        LDA_incre_supervised, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_supervised)

                    name = type_DA + '_' + 'incre_Nigam_' + str(0.1)
                    LDA_incre_Nigam_01, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        LDA_incre_Nigam_01, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_Nigam_01, weakModel,
                        weight_Nigam=0.1)
                    name = type_DA + '_' + 'incre_Nigam_' + str(0.2)
                    LDA_incre_Nigam_02, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        LDA_incre_Nigam_02, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_Nigam_02, weakModel,
                        weight_Nigam=0.2)
                    name = type_DA + '_' + 'incre_Nigam_' + str(0.4)
                    LDA_incre_Nigam_04, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        LDA_incre_Nigam_04, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_Nigam_04, weakModel,
                        weight_Nigam=0.4)
                    name = type_DA + '_' + 'incre_Nigam_' + str(0.6)
                    LDA_incre_Nigam_06, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        LDA_incre_Nigam_06, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_Nigam_06, weakModel,
                        weight_Nigam=0.6)
                    name = type_DA + '_' + 'incre_Nigam_' + str(0.8)
                    LDA_incre_Nigam_08, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        LDA_incre_Nigam_08, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_Nigam_08, weakModel,
                        weight_Nigam=0.8)

                    name = type_DA + '_' + 'incre_Nigam_' + str(1.0)
                    LDA_incre_Nigam_10, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        LDA_incre_Nigam_10, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_Nigam_10, weakModel,
                        weight_Nigam=1)

                    name = type_DA + '_' + 'incre_threshold_' + str(0.2)
                    LDA_incre_threshold_02, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        LDA_incre_threshold_02, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_02,
                        weakModel, threshold=0.2)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.3)
                    LDA_incre_threshold_03, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        LDA_incre_threshold_03, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_03,
                        weakModel, threshold=0.3)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.4)
                    LDA_incre_threshold_04, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        LDA_incre_threshold_04, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_04,
                        weakModel, threshold=0.4)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.5)
                    LDA_incre_threshold_05, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        LDA_incre_threshold_05, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_05,
                        weakModel, threshold=0.5)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.6)
                    LDA_incre_threshold_06, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        LDA_incre_threshold_06, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_06,
                        weakModel, threshold=0.6)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.7)
                    LDA_incre_threshold_07, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        LDA_incre_threshold_07, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_07,
                        weakModel, threshold=0.7)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.8)
                    LDA_incre_threshold_08, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        LDA_incre_threshold_08, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_08,
                        weakModel, threshold=0.8)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.9)
                    LDA_incre_threshold_09, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        LDA_incre_threshold_09, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_09,
                        weakModel, threshold=0.9)

                    ###################################
                    type_DA = 'QDA'

                    name = type_DA + '_' + 'weak'
                    CM_QDA_weak[rand[1] - 1, :] += SemiSupervised.predicted_labels(trainFeatures, QDA_weak, classes,
                                                                                   type_DA)

                    name = type_DA + '_' + 'incre_proposed'
                    QDA_incre_proposed, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposed(
                        QDA_incre_proposed, classes, trainFeatures, rand[1], labeledGesturesFeatures,
                        labeledGesturesLabels, type_DA, CM_QDA_incre_proposed, weakModel)
                    # print('time QDA', results.at[idx, 'time' + '_' + name])

                    name = type_DA + '_' + 'incre_proposed2'
                    QDA_incre_proposed2, _, _ = SemiSupervised.model_incre_proposed2(
                        QDA_incre_proposed2, classes, trainFeatures, rand[1], labeledGesturesFeatures,
                        labeledGesturesLabels, type_DA, CM_QDA_incre_proposed2, weakModel)

                    name = type_DA + '_' + 'incre_proposed3'
                    QDA_incre_proposed3, _, _ = SemiSupervised.model_incre_proposed3(
                        QDA_incre_proposed3, classes, trainFeatures, rand[1], labeledGesturesFeatures,
                        labeledGesturesLabels, type_DA, CM_QDA_incre_proposed3, weakModel)

                    name = type_DA + '_' + 'incre_labels'
                    QDA_incre_label, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_labels(
                        QDA_incre_label, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_label, weakModel)

                    name = type_DA + '_' + 'incre_sequential'
                    QDA_incre_sequential, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_sequential_labels(
                        QDA_incre_sequential, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_sequential,
                        weakModel)

                    name = type_DA + '_' + 'incre_sequential_probs'
                    QDA_incre_sequential_probs, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_sequential_probs(
                        QDA_incre_sequential_probs, classes, trainFeatures, rand[1], type_DA,
                        CM_QDA_incre_sequential_probs, weakModel)

                    name = type_DA + '_' + 'incre_supervised'
                    QDA_incre_supervised, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_supervised(
                        QDA_incre_supervised, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_supervised)

                    name = type_DA + '_' + 'incre_Nigam_' + str(0.1)
                    QDA_incre_Nigam_01, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        QDA_incre_Nigam_01, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_Nigam_01, weakModel,
                        weight_Nigam=0.1)
                    name = type_DA + '_' + 'incre_Nigam_' + str(0.2)
                    QDA_incre_Nigam_02, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        QDA_incre_Nigam_02, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_Nigam_02, weakModel,
                        weight_Nigam=0.2)
                    name = type_DA + '_' + 'incre_Nigam_' + str(0.4)
                    QDA_incre_Nigam_04, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        QDA_incre_Nigam_04, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_Nigam_04, weakModel,
                        weight_Nigam=0.4)
                    name = type_DA + '_' + 'incre_Nigam_' + str(0.6)
                    QDA_incre_Nigam_06, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        QDA_incre_Nigam_06, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_Nigam_06, weakModel,
                        weight_Nigam=0.6)
                    name = type_DA + '_' + 'incre_Nigam_' + str(0.8)
                    QDA_incre_Nigam_08, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        QDA_incre_Nigam_08, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_Nigam_08, weakModel,
                        weight_Nigam=0.8)

                    name = type_DA + '_' + 'incre_Nigam_' + str(1.0)
                    QDA_incre_Nigam_10, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        QDA_incre_Nigam_10, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_Nigam_10, weakModel,
                        weight_Nigam=1)

                    name = type_DA + '_' + 'incre_threshold_' + str(0.2)
                    QDA_incre_threshold_02, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        QDA_incre_threshold_02, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_02,
                        weakModel, threshold=0.2)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.3)
                    QDA_incre_threshold_03, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        QDA_incre_threshold_03, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_03,
                        weakModel, threshold=0.3)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.4)
                    QDA_incre_threshold_04, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        QDA_incre_threshold_04, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_04,
                        weakModel, threshold=0.4)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.5)
                    QDA_incre_threshold_05, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        QDA_incre_threshold_05, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_05,
                        weakModel, threshold=0.5)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.6)
                    QDA_incre_threshold_06, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        QDA_incre_threshold_06, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_06,
                        weakModel, threshold=0.6)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.7)
                    QDA_incre_threshold_07, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        QDA_incre_threshold_07, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_07,
                        weakModel, threshold=0.7)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.8)
                    QDA_incre_threshold_08, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        QDA_incre_threshold_08, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_08,
                        weakModel, threshold=0.8)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.9)
                    QDA_incre_threshold_09, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        QDA_incre_threshold_09, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_09,
                        weakModel, threshold=0.9)

                    results.at[idx, 'Feature Set'] = featureSet
                    results.at[idx, 'person'] = person
                    results.at[idx, 'exp_time'] = seed
                    results.at[idx, 'shots in traning'] = initialSamples
                    results.at[idx, 'day'] = day
                    results.at[idx, 'shot_class'] = rand
                    results.at[idx, 'unlabeled Gesture'] = numberUnlabeledGestures
                    numberUnlabeledGestures += 1

                    if np.all(permutationUnlabeledGestures[-1] == rand):

                        type_DA = 'LDA'
                        name = type_DA + '_' + 'weak'
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_weak)
                        name = type_DA + '_' + 'incre_proposed'
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_proposed)
                        name = type_DA + '_' + 'incre_proposed2'
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_proposed2)
                        name = type_DA + '_' + 'incre_proposed3'
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_proposed3)
                        name = type_DA + '_' + 'incre_labels'
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_label)
                        name = type_DA + '_' + 'incre_sequential'
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_sequential)
                        name = type_DA + '_' + 'incre_sequential_probs'
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_sequential_probs)
                        name = type_DA + '_' + 'incre_supervised'
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_supervised)
                        name = type_DA + '_' + 'incre_Nigam_' + str(0.1)
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_Nigam_01)
                        name = type_DA + '_' + 'incre_Nigam_' + str(0.2)
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_Nigam_02)
                        name = type_DA + '_' + 'incre_Nigam_' + str(0.4)
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_LDA_incre_Nigam_04)
                        name = type_DA + '_' + 'incre_Nigam_' + str(0.6)
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_LDA_incre_Nigam_06)
                        name = type_DA + '_' + 'incre_Nigam_' + str(0.8)
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_LDA_incre_Nigam_08)
                        name = type_DA + '_' + 'incre_Nigam_' + str(1.0)
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_Nigam_10)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.2)
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_LDA_incre_threshold_02)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.3)
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_LDA_incre_threshold_03)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.4)
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_LDA_incre_threshold_04)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.5)
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_LDA_incre_threshold_05)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.6)
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_LDA_incre_threshold_06)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.7)
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_LDA_incre_threshold_07)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.8)
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_LDA_incre_threshold_08)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.9)
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_LDA_incre_threshold_09)
                        ###################################
                        type_DA = 'QDA'
                        name = type_DA + '_' + 'weak'
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_weak)
                        name = type_DA + '_' + 'incre_proposed'
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_proposed)
                        name = type_DA + '_' + 'incre_proposed2'
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_proposed2)
                        name = type_DA + '_' + 'incre_proposed3'
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_proposed3)
                        name = type_DA + '_' + 'incre_labels'
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_label)
                        name = type_DA + '_' + 'incre_sequential'
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_QDA_incre_sequential)
                        name = type_DA + '_' + 'incre_sequential_probs'
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_QDA_incre_sequential_probs)
                        name = type_DA + '_' + 'incre_supervised'
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_supervised)
                        name = type_DA + '_' + 'incre_Nigam_' + str(0.1)
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_QDA_incre_Nigam_01)
                        name = type_DA + '_' + 'incre_Nigam_' + str(0.2)
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_Nigam_02)
                        name = type_DA + '_' + 'incre_Nigam_' + str(0.4)
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_Nigam_04)
                        name = type_DA + '_' + 'incre_Nigam_' + str(0.6)
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_QDA_incre_Nigam_06)
                        name = type_DA + '_' + 'incre_Nigam_' + str(0.8)
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_QDA_incre_Nigam_08)
                        name = type_DA + '_' + 'incre_Nigam_' + str(1.0)
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_Nigam_10)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.2)
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_threshold_02)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.3)
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_QDA_incre_threshold_03)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.4)
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_QDA_incre_threshold_04)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.5)
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_QDA_incre_threshold_05)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.6)
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_threshold_06)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.7)
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_QDA_incre_threshold_07)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.8)
                        results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_threshold_08)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.9)
                        results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_QDA_incre_threshold_09)

                        if printR:
                            print('person', person, 'day', day, 'exp', seed)
                            for type_DA in ['LDA', 'QDA']:
                                for model in ['weak', 'incre_proposed', 'incre_proposed2', 'incre_proposed3',
                                              'incre_supervised']:
                                    name = type_DA + '_' + model
                                    print('acc_' + name, results.loc[idx, 'acc_' + name])

                        # np.save(
                        #     'confussionMatrix/' + 'LDA_weak' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_weak)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_proposed' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_proposed)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_proposed2' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_proposed2)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_proposed3' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_proposed3)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_label' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_label)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_sequential' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_sequential)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_sequential_probs' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_sequential_probs)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_supervised' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_supervised)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_Nigam_01' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_Nigam_01)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_Nigam_02' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_Nigam_02)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_Nigam_04' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_Nigam_04)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_Nigam_06' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_Nigam_06)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_Nigam_08' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_Nigam_08)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_Nigam_10' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_Nigam_10)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_threshold_02' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_threshold_02)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_threshold_03' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_threshold_03)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_threshold_04' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_threshold_04)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_threshold_05' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_threshold_05)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_threshold_06' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_threshold_06)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_threshold_07' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_threshold_07)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_threshold_08' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_threshold_08)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_threshold_09' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_threshold_09)
                        #
                        # # ###############################QDA##########################################################
                        #
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_weak' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_weak)
                        #
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_proposed' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_proposed)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_proposed2' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_proposed2)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_proposed3' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_proposed3)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_label' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_label)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_sequential' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_sequential)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_sequential_probs' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_sequential_probs)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_supervised' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_supervised)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_Nigam_01' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_Nigam_01)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_Nigam_02' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_Nigam_02)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_Nigam_04' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_Nigam_04)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_Nigam_06' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_Nigam_06)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_Nigam_08' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_Nigam_08)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_Nigam_10' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_Nigam_10)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_threshold_02' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_threshold_02)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_threshold_03' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_threshold_03)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_threshold_04' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_threshold_04)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_threshold_05' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_threshold_05)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_threshold_06' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_threshold_06)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_threshold_07' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_threshold_07)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_threshold_08' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_threshold_08)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_threshold_09' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_threshold_09)

                    if nameFile is not None:
                        results.to_csv(nameFile)

                    idx += 1


def evaluation(featureSet, nameFile, startPerson, endPerson, printR, initialSamples, initialTime, finalTime,
               typeDatabase):
    scaler = preprocessing.MinMaxScaler()

    results = pd.DataFrame(
        columns=['Feature Set', 'person', 'exp_time', 'shots in traning', 'shot_class', 'day', 'unlabeled Gesture'])

    for metric in ['w_predicted', 'time', 'precision', 'recall']:
        for typeDA in ['LDA', 'QDA']:
            results[metric + '_' + typeDA + '_' + 'weak'] = ''
            results[metric + '_' + typeDA + '_' + 'incre_proposed'] = ''
            results[metric + '_' + typeDA + '_' + 'incre_labels'] = ''
            results[metric + '_' + typeDA + '_' + 'incre_sequential'] = ''
            results[metric + '_' + typeDA + '_' + 'incre_sequential_probs'] = ''
            results[metric + '_' + typeDA + '_' + 'incre_supervised'] = ''
            for l in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
                results[metric + '_' + typeDA + '_' + 'incre_Nigam_' + str(l)] = ''
            for l in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                results[metric + '_' + typeDA + '_' + 'incre_threshold_' + str(l)] = ''

    idx = 0

    for person in range(startPerson, endPerson + 1):
        # Upload Data
        dataMatrix, classes, numberShots, allFeatures, numSamples, days, testSamples = uploadDatabases(
            typeDatabase, person, featureSet)

        for seed in range(initialTime, finalTime + 1):
            for day in range(1, days + 1):
                numberUnlabeledGestures = 1
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
                print(labeledGestures)
                print(unlabeledGestures)
                print(list(permutationUnlabeledGestures))
                print(testGestures)

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

                weakModel = currentDistributionValues(
                    labeledGesturesFeatures, labeledGesturesLabels, classes, np.size(labeledGesturesFeatures, axis=1),
                    oldData=False)

                labeledGesturesFeatures, labeledGesturesLabels = systematicSampling(
                    labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)
                print('number labeled samples', len(labeledGesturesLabels) / classes)

                LDA_weak = weakModel.copy()
                LDA_incre_proposed = weakModel.copy()
                LDA_incre_proposed2 = weakModel.copy()
                LDA_incre_proposed3 = weakModel.copy()
                LDA_incre_label = weakModel.copy()
                LDA_incre_sequential = weakModel.copy()
                LDA_incre_sequential_probs = weakModel.copy()
                LDA_incre_supervised = weakModel.copy()
                LDA_incre_Nigam_01 = weakModel.copy()
                LDA_incre_Nigam_02 = weakModel.copy()
                LDA_incre_Nigam_04 = weakModel.copy()
                LDA_incre_Nigam_06 = weakModel.copy()
                LDA_incre_Nigam_08 = weakModel.copy()
                LDA_incre_Nigam_10 = weakModel.copy()
                LDA_incre_threshold_02 = weakModel.copy()
                LDA_incre_threshold_03 = weakModel.copy()
                LDA_incre_threshold_04 = weakModel.copy()
                LDA_incre_threshold_05 = weakModel.copy()
                LDA_incre_threshold_06 = weakModel.copy()
                LDA_incre_threshold_07 = weakModel.copy()
                LDA_incre_threshold_08 = weakModel.copy()
                LDA_incre_threshold_09 = weakModel.copy()

                QDA_weak = weakModel.copy()
                QDA_incre_proposed = weakModel.copy()
                QDA_incre_proposed2 = weakModel.copy()
                QDA_incre_proposed3 = weakModel.copy()
                QDA_incre_label = weakModel.copy()
                QDA_incre_sequential = weakModel.copy()
                QDA_incre_sequential_probs = weakModel.copy()
                QDA_incre_supervised = weakModel.copy()
                QDA_incre_Nigam_01 = weakModel.copy()
                QDA_incre_Nigam_02 = weakModel.copy()
                QDA_incre_Nigam_04 = weakModel.copy()
                QDA_incre_Nigam_06 = weakModel.copy()
                QDA_incre_Nigam_08 = weakModel.copy()
                QDA_incre_Nigam_10 = weakModel.copy()
                QDA_incre_threshold_02 = weakModel.copy()
                QDA_incre_threshold_03 = weakModel.copy()
                QDA_incre_threshold_04 = weakModel.copy()
                QDA_incre_threshold_05 = weakModel.copy()
                QDA_incre_threshold_06 = weakModel.copy()
                QDA_incre_threshold_07 = weakModel.copy()
                QDA_incre_threshold_08 = weakModel.copy()
                QDA_incre_threshold_09 = weakModel.copy()

                CM_LDA_weak = np.zeros((classes, classes))
                CM_LDA_incre_proposed = np.zeros((classes, classes))
                CM_LDA_incre_proposed2 = np.zeros((classes, classes))
                CM_LDA_incre_proposed3 = np.zeros((classes, classes))
                CM_LDA_incre_label = np.zeros((classes, classes))
                CM_LDA_incre_sequential = np.zeros((classes, classes))
                CM_LDA_incre_sequential_probs = np.zeros((classes, classes))
                CM_LDA_incre_supervised = np.zeros((classes, classes))
                CM_LDA_incre_Nigam_01 = np.zeros((classes, classes))
                CM_LDA_incre_Nigam_02 = np.zeros((classes, classes))
                CM_LDA_incre_Nigam_04 = np.zeros((classes, classes))
                CM_LDA_incre_Nigam_06 = np.zeros((classes, classes))
                CM_LDA_incre_Nigam_08 = np.zeros((classes, classes))
                CM_LDA_incre_Nigam_10 = np.zeros((classes, classes))
                CM_LDA_incre_threshold_02 = np.zeros((classes, classes))
                CM_LDA_incre_threshold_03 = np.zeros((classes, classes))
                CM_LDA_incre_threshold_04 = np.zeros((classes, classes))
                CM_LDA_incre_threshold_05 = np.zeros((classes, classes))
                CM_LDA_incre_threshold_06 = np.zeros((classes, classes))
                CM_LDA_incre_threshold_07 = np.zeros((classes, classes))
                CM_LDA_incre_threshold_08 = np.zeros((classes, classes))
                CM_LDA_incre_threshold_09 = np.zeros((classes, classes))

                CM_QDA_weak = np.zeros((classes, classes))
                CM_QDA_incre_proposed = np.zeros((classes, classes))
                CM_QDA_incre_proposed2 = np.zeros((classes, classes))
                CM_QDA_incre_proposed3 = np.zeros((classes, classes))
                CM_QDA_incre_label = np.zeros((classes, classes))
                CM_QDA_incre_sequential = np.zeros((classes, classes))
                CM_QDA_incre_sequential_probs = np.zeros((classes, classes))
                CM_QDA_incre_supervised = np.zeros((classes, classes))
                CM_QDA_incre_Nigam_01 = np.zeros((classes, classes))
                CM_QDA_incre_Nigam_02 = np.zeros((classes, classes))
                CM_QDA_incre_Nigam_04 = np.zeros((classes, classes))
                CM_QDA_incre_Nigam_06 = np.zeros((classes, classes))
                CM_QDA_incre_Nigam_08 = np.zeros((classes, classes))
                CM_QDA_incre_Nigam_10 = np.zeros((classes, classes))
                CM_QDA_incre_threshold_02 = np.zeros((classes, classes))
                CM_QDA_incre_threshold_03 = np.zeros((classes, classes))
                CM_QDA_incre_threshold_04 = np.zeros((classes, classes))
                CM_QDA_incre_threshold_05 = np.zeros((classes, classes))
                CM_QDA_incre_threshold_06 = np.zeros((classes, classes))
                CM_QDA_incre_threshold_07 = np.zeros((classes, classes))
                CM_QDA_incre_threshold_08 = np.zeros((classes, classes))
                CM_QDA_incre_threshold_09 = np.zeros((classes, classes))

                # iauxttt = 0
                for rand in list(permutationUnlabeledGestures):
                    trainFeatures = dataMatrix[
                                    (dataMatrix[:, allFeatures] == rand[2]) & (
                                            dataMatrix[:, allFeatures + 1] == person) & (
                                            dataMatrix[:, allFeatures + 3] == rand[0]) & (
                                            dataMatrix[:, allFeatures + 2] == rand[1]), 0:allFeatures]
                    # trainLabels = dataMatrix[
                    #     (dataMatrix[:, allFeatures] == rand[2]) & (dataMatrix[:, allFeatures + 1] == person) & (
                    #             dataMatrix[:, allFeatures + 3] == rand[0]) & (
                    #             dataMatrix[:, allFeatures + 2] == rand[1]), allFeatures + 2]

                    # print('\nreal label',rand[1])

                    #####Pre-processing
                    trainFeatures = scaler.transform(trainFeatures)
                    if typeDatabase == 'Capgmyo_dbb':
                        trainFeatures = pca_model.transform(trainFeatures)

                    # if iauxttt == 0:
                    #     accumLabeledGesturesFeatures = trainFeatures
                    #     accumLabeledGesturesLabels = trainLabels
                    #     iauxttt = 1
                    # else:
                    #     accumLabeledGesturesFeatures = np.vstack((accumLabeledGesturesFeatures, trainFeatures))
                    #     accumLabeledGesturesLabels = np.hstack((accumLabeledGesturesLabels, trainLabels))

                    type_DA = 'LDA'

                    name = type_DA + '_' + 'weak'
                    CM_LDA_weak[rand[1] - 1, :] += SemiSupervised.predicted_labels(trainFeatures, LDA_weak, classes,
                                                                                   type_DA)

                    name = type_DA + '_' + 'incre_proposed'
                    LDA_incre_proposed, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposed(
                        LDA_incre_proposed, classes, trainFeatures, rand[1], labeledGesturesFeatures,
                        labeledGesturesLabels, type_DA, CM_LDA_incre_proposed, weakModel)
                    # print('time LDA', results.at[idx, 'time' + '_' + name])

                    name = type_DA + '_' + 'incre_proposed2'
                    LDA_incre_proposed2, _, _ = SemiSupervised.model_incre_proposed2(
                        LDA_incre_proposed2, classes, trainFeatures, rand[1], labeledGesturesFeatures,
                        labeledGesturesLabels, type_DA, CM_LDA_incre_proposed2, weakModel)

                    name = type_DA + '_' + 'incre_proposed3'
                    LDA_incre_proposed3, _, _ = SemiSupervised.model_incre_proposed3(
                        LDA_incre_proposed3, classes, trainFeatures, rand[1], labeledGesturesFeatures,
                        labeledGesturesLabels, type_DA, CM_LDA_incre_proposed3, weakModel)

                    name = type_DA + '_' + 'incre_labels'
                    LDA_incre_label, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_labels(
                        LDA_incre_label, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_label, weakModel)

                    name = type_DA + '_' + 'incre_sequential'
                    LDA_incre_sequential, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_sequential_labels(
                        LDA_incre_sequential, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_sequential,
                        weakModel)

                    name = type_DA + '_' + 'incre_sequential_probs'
                    LDA_incre_sequential_probs, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_sequential_probs(
                        LDA_incre_sequential_probs, classes, trainFeatures, rand[1], type_DA,
                        CM_LDA_incre_sequential_probs, weakModel)

                    name = type_DA + '_' + 'incre_supervised'
                    LDA_incre_supervised, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_supervised(
                        LDA_incre_supervised, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_supervised)

                    name = type_DA + '_' + 'incre_Nigam_' + str(0.1)
                    LDA_incre_Nigam_01, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        LDA_incre_Nigam_01, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_Nigam_01, weakModel,
                        weight_Nigam=0.1)
                    name = type_DA + '_' + 'incre_Nigam_' + str(0.2)
                    LDA_incre_Nigam_02, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        LDA_incre_Nigam_02, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_Nigam_02, weakModel,
                        weight_Nigam=0.2)
                    name = type_DA + '_' + 'incre_Nigam_' + str(0.4)
                    LDA_incre_Nigam_04, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        LDA_incre_Nigam_04, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_Nigam_04, weakModel,
                        weight_Nigam=0.4)
                    name = type_DA + '_' + 'incre_Nigam_' + str(0.6)
                    LDA_incre_Nigam_06, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        LDA_incre_Nigam_06, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_Nigam_06, weakModel,
                        weight_Nigam=0.6)
                    name = type_DA + '_' + 'incre_Nigam_' + str(0.8)
                    LDA_incre_Nigam_08, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        LDA_incre_Nigam_08, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_Nigam_08, weakModel,
                        weight_Nigam=0.8)

                    name = type_DA + '_' + 'incre_Nigam_' + str(1.0)
                    LDA_incre_Nigam_10, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        LDA_incre_Nigam_10, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_Nigam_10, weakModel,
                        weight_Nigam=1)

                    name = type_DA + '_' + 'incre_threshold_' + str(0.2)
                    LDA_incre_threshold_02, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        LDA_incre_threshold_02, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_02,
                        weakModel, threshold=0.2)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.3)
                    LDA_incre_threshold_03, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        LDA_incre_threshold_03, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_03,
                        weakModel, threshold=0.3)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.4)
                    LDA_incre_threshold_04, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        LDA_incre_threshold_04, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_04,
                        weakModel, threshold=0.4)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.5)
                    LDA_incre_threshold_05, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        LDA_incre_threshold_05, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_05,
                        weakModel, threshold=0.5)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.6)
                    LDA_incre_threshold_06, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        LDA_incre_threshold_06, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_06,
                        weakModel, threshold=0.6)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.7)
                    LDA_incre_threshold_07, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        LDA_incre_threshold_07, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_07,
                        weakModel, threshold=0.7)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.8)
                    LDA_incre_threshold_08, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        LDA_incre_threshold_08, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_08,
                        weakModel, threshold=0.8)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.9)
                    LDA_incre_threshold_09, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        LDA_incre_threshold_09, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_09,
                        weakModel, threshold=0.9)

                    ###################################
                    type_DA = 'QDA'

                    name = type_DA + '_' + 'weak'
                    CM_QDA_weak[rand[1] - 1, :] += SemiSupervised.predicted_labels(trainFeatures, QDA_weak, classes,
                                                                                   type_DA)

                    name = type_DA + '_' + 'incre_proposed'
                    QDA_incre_proposed, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposed(
                        QDA_incre_proposed, classes, trainFeatures, rand[1], labeledGesturesFeatures,
                        labeledGesturesLabels, type_DA, CM_QDA_incre_proposed, weakModel)
                    # print('time QDA', results.at[idx, 'time' + '_' + name])

                    name = type_DA + '_' + 'incre_proposed2'
                    QDA_incre_proposed2, _, _ = SemiSupervised.model_incre_proposed2(
                        QDA_incre_proposed2, classes, trainFeatures, rand[1], labeledGesturesFeatures,
                        labeledGesturesLabels, type_DA, CM_QDA_incre_proposed2, weakModel)

                    name = type_DA + '_' + 'incre_proposed3'
                    QDA_incre_proposed3, _, _ = SemiSupervised.model_incre_proposed3(
                        QDA_incre_proposed3, classes, trainFeatures, rand[1], labeledGesturesFeatures,
                        labeledGesturesLabels, type_DA, CM_QDA_incre_proposed3, weakModel)

                    name = type_DA + '_' + 'incre_labels'
                    QDA_incre_label, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_labels(
                        QDA_incre_label, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_label, weakModel)

                    name = type_DA + '_' + 'incre_sequential'
                    QDA_incre_sequential, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_sequential_labels(
                        QDA_incre_sequential, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_sequential,
                        weakModel)

                    name = type_DA + '_' + 'incre_sequential_probs'
                    QDA_incre_sequential_probs, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_sequential_probs(
                        QDA_incre_sequential_probs, classes, trainFeatures, rand[1], type_DA,
                        CM_QDA_incre_sequential_probs, weakModel)

                    name = type_DA + '_' + 'incre_supervised'
                    QDA_incre_supervised, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_supervised(
                        QDA_incre_supervised, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_supervised)

                    name = type_DA + '_' + 'incre_Nigam_' + str(0.1)
                    QDA_incre_Nigam_01, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        QDA_incre_Nigam_01, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_Nigam_01, weakModel,
                        weight_Nigam=0.1)
                    name = type_DA + '_' + 'incre_Nigam_' + str(0.2)
                    QDA_incre_Nigam_02, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        QDA_incre_Nigam_02, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_Nigam_02, weakModel,
                        weight_Nigam=0.2)
                    name = type_DA + '_' + 'incre_Nigam_' + str(0.4)
                    QDA_incre_Nigam_04, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        QDA_incre_Nigam_04, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_Nigam_04, weakModel,
                        weight_Nigam=0.4)
                    name = type_DA + '_' + 'incre_Nigam_' + str(0.6)
                    QDA_incre_Nigam_06, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        QDA_incre_Nigam_06, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_Nigam_06, weakModel,
                        weight_Nigam=0.6)
                    name = type_DA + '_' + 'incre_Nigam_' + str(0.8)
                    QDA_incre_Nigam_08, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        QDA_incre_Nigam_08, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_Nigam_08, weakModel,
                        weight_Nigam=0.8)

                    name = type_DA + '_' + 'incre_Nigam_' + str(1.0)
                    QDA_incre_Nigam_10, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
                        QDA_incre_Nigam_10, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_Nigam_10, weakModel,
                        weight_Nigam=1)

                    name = type_DA + '_' + 'incre_threshold_' + str(0.2)
                    QDA_incre_threshold_02, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        QDA_incre_threshold_02, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_02,
                        weakModel, threshold=0.2)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.3)
                    QDA_incre_threshold_03, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        QDA_incre_threshold_03, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_03,
                        weakModel, threshold=0.3)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.4)
                    QDA_incre_threshold_04, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        QDA_incre_threshold_04, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_04,
                        weakModel, threshold=0.4)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.5)
                    QDA_incre_threshold_05, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        QDA_incre_threshold_05, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_05,
                        weakModel, threshold=0.5)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.6)
                    QDA_incre_threshold_06, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        QDA_incre_threshold_06, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_06,
                        weakModel, threshold=0.6)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.7)
                    QDA_incre_threshold_07, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        QDA_incre_threshold_07, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_07,
                        weakModel, threshold=0.7)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.8)
                    QDA_incre_threshold_08, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        QDA_incre_threshold_08, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_08,
                        weakModel, threshold=0.8)
                    name = type_DA + '_' + 'incre_threshold_' + str(0.9)
                    QDA_incre_threshold_09, results.at[idx, 'time' + '_' + name], results.at[
                        idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
                        QDA_incre_threshold_09, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_09,
                        weakModel, threshold=0.9)

                    results.at[idx, 'Feature Set'] = featureSet
                    results.at[idx, 'person'] = person
                    results.at[idx, 'exp_time'] = seed
                    results.at[idx, 'shots in traning'] = initialSamples
                    results.at[idx, 'day'] = day
                    results.at[idx, 'shot_class'] = rand
                    results.at[idx, 'unlabeled Gesture'] = numberUnlabeledGestures
                    numberUnlabeledGestures += 1

                    # if np.all(permutationUnlabeledGestures[-1] == rand):
                    #
                    #     type_DA = 'LDA'
                    #     name = type_DA + '_' + 'weak'
                    #     results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_LDA_weak)
                    #     name = type_DA + '_' + 'incre_proposed'
                    #     results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_LDA_incre_proposed)
                    #     name = type_DA + '_' + 'incre_proposed2'
                    #     results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_LDA_incre_proposed2)
                    #     name = type_DA + '_' + 'incre_proposed3'
                    #     results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_LDA_incre_proposed3)
                    #
                    #     name = type_DA + '_' + 'incre_sequential'
                    #     results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_sequential)
                    #     name = type_DA + '_' + 'incre_sequential_probs'
                    #     results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_LDA_incre_sequential_probs)
                    #
                    #     ###################################
                    #     type_DA = 'QDA'
                    #     name = type_DA + '_' + 'weak'
                    #     results.at[idx, 'acc_' + name]=accuracyPerGesture(CM_QDA_weak)
                    #     name = type_DA + '_' + 'incre_proposed'
                    #     results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_proposed)
                    #     name = type_DA + '_' + 'incre_proposed2'
                    #     results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_proposed2)
                    #     name = type_DA + '_' + 'incre_proposed3'
                    #     results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_QDA_incre_proposed3)
                    #
                    #     name = type_DA + '_' + 'incre_sequential'
                    #     results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_sequential)
                    #     name = type_DA + '_' + 'incre_sequential_probs'
                    #     results.at[idx, 'acc_' + name]= accuracyPerGesture(CM_QDA_incre_sequential_probs)
                    #
                    #
                    #     if printR:
                    #         print('person', person, 'day', day)
                    #         for type_DA in ['LDA', 'QDA']:
                    #             # for model in ['weak', 'incre_proposed', 'incre_labels', 'incre_sequential',
                    #             #               'incre_sequential_probs', 'incre_supervised',
                    #             #               'incre_Nigam_' + str(1.0)]:
                    #             for model in ['weak', 'incre_proposed', 'incre_proposed2', 'incre_proposed3',
                    #                           'incre_sequential',
                    #                           'incre_sequential_probs']:
                    #                 name = type_DA + '_' + model
                    #                 print('acc_' + name, results.loc[idx, 'acc_' + name])

                    if np.all(permutationUnlabeledGestures[-1] == rand):

                        type_DA = 'LDA'
                        name = type_DA + '_' + 'weak'
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_weak, classes)
                        name = type_DA + '_' + 'incre_proposed'
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_proposed, classes)
                        name = type_DA + '_' + 'incre_proposed2'
                        results.at[idx, 'acc_' + name], _, _ = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_proposed2, classes)
                        name = type_DA + '_' + 'incre_proposed3'
                        results.at[idx, 'acc_' + name], _, _ = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_proposed3, classes)
                        name = type_DA + '_' + 'incre_labels'
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_label, classes)
                        name = type_DA + '_' + 'incre_sequential'
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_sequential, classes)
                        name = type_DA + '_' + 'incre_sequential_probs'
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_sequential_probs, classes)
                        name = type_DA + '_' + 'incre_supervised'
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_supervised, classes)
                        name = type_DA + '_' + 'incre_Nigam_' + str(0.1)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_Nigam_01, classes)
                        name = type_DA + '_' + 'incre_Nigam_' + str(0.2)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_Nigam_02, classes)
                        name = type_DA + '_' + 'incre_Nigam_' + str(0.4)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_Nigam_04, classes)
                        name = type_DA + '_' + 'incre_Nigam_' + str(0.6)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_Nigam_06, classes)
                        name = type_DA + '_' + 'incre_Nigam_' + str(0.8)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_Nigam_08, classes)
                        name = type_DA + '_' + 'incre_Nigam_' + str(1.0)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_Nigam_10, classes)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.2)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_threshold_02, classes)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.3)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_threshold_03, classes)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.4)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_threshold_04, classes)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.5)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_threshold_05, classes)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.6)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_threshold_06, classes)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.7)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_threshold_07, classes)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.8)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_threshold_08, classes)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.9)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                            testGesturesFeatures, testGesturesLabels, LDA_incre_threshold_09, classes)
                        ###################################
                        type_DA = 'QDA'
                        name = type_DA + '_' + 'weak'
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_weak, classes)
                        name = type_DA + '_' + 'incre_proposed'
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_proposed, classes)
                        name = type_DA + '_' + 'incre_proposed2'
                        results.at[idx, 'acc_' + name], _, _ = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_proposed2, classes)
                        name = type_DA + '_' + 'incre_proposed3'
                        results.at[idx, 'acc_' + name], _, _ = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_proposed3, classes)
                        name = type_DA + '_' + 'incre_labels'
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_label, classes)
                        name = type_DA + '_' + 'incre_sequential'
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_sequential, classes)
                        name = type_DA + '_' + 'incre_sequential_probs'
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_sequential_probs, classes)
                        name = type_DA + '_' + 'incre_supervised'
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_supervised, classes)
                        name = type_DA + '_' + 'incre_Nigam_' + str(0.1)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_Nigam_01, classes)
                        name = type_DA + '_' + 'incre_Nigam_' + str(0.2)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_Nigam_02, classes)
                        name = type_DA + '_' + 'incre_Nigam_' + str(0.4)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_Nigam_04, classes)
                        name = type_DA + '_' + 'incre_Nigam_' + str(0.6)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_Nigam_06, classes)
                        name = type_DA + '_' + 'incre_Nigam_' + str(0.8)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_Nigam_08, classes)
                        name = type_DA + '_' + 'incre_Nigam_' + str(1.0)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_Nigam_10, classes)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.2)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_threshold_02, classes)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.3)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_threshold_03, classes)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.4)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_threshold_04, classes)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.5)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_threshold_05, classes)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.6)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_threshold_06, classes)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.7)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_threshold_07, classes)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.8)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_threshold_08, classes)
                        name = type_DA + '_' + 'incre_threshold_' + str(0.9)
                        results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                            idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                            testGesturesFeatures, testGesturesLabels, QDA_incre_threshold_09, classes)

                        if printR:
                            print('person', person, 'day', day, 'exp', seed)
                            for type_DA in ['LDA', 'QDA']:
                                for model in ['weak', 'incre_proposed', 'incre_proposed2', 'incre_proposed3']:
                                    name = type_DA + '_' + model
                                    print('acc_' + name, results.loc[idx, 'acc_' + name])

                        # np.save(
                        #     'confussionMatrix/' + 'LDA_weak' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_weak)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_proposed' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_proposed)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_proposed2' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_proposed2)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_proposed3' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_proposed3)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_label' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_label)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_sequential' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_sequential)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_sequential_probs' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_sequential_probs)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_supervised' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_supervised)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_Nigam_01' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_Nigam_01)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_Nigam_02' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_Nigam_02)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_Nigam_04' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_Nigam_04)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_Nigam_06' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_Nigam_06)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_Nigam_08' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_Nigam_08)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_Nigam_10' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_Nigam_10)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_threshold_02' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_threshold_02)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_threshold_03' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_threshold_03)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_threshold_04' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_threshold_04)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_threshold_05' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_threshold_05)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_threshold_06' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_threshold_06)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_threshold_07' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_threshold_07)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_threshold_08' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_threshold_08)
                        # np.save(
                        #     'confussionMatrix/' + 'LDA_incre_threshold_09' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_LDA_incre_threshold_09)
                        #
                        # # ###############################QDA##########################################################
                        #
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_weak' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_weak)
                        #
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_proposed' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_proposed)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_proposed2' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_proposed2)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_proposed3' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_proposed3)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_label' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_label)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_sequential' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_sequential)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_sequential_probs' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_sequential_probs)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_supervised' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_supervised)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_Nigam_01' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_Nigam_01)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_Nigam_02' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_Nigam_02)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_Nigam_04' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_Nigam_04)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_Nigam_06' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_Nigam_06)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_Nigam_08' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_Nigam_08)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_Nigam_10' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_Nigam_10)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_threshold_02' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_threshold_02)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_threshold_03' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_threshold_03)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_threshold_04' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_threshold_04)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_threshold_05' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_threshold_05)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_threshold_06' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_threshold_06)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_threshold_07' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_threshold_07)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_threshold_08' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_threshold_08)
                        # np.save(
                        #     'confussionMatrix/' + 'QDA_incre_threshold_09' + '_' + typeDatabase + '_featureSet_' + str(
                        #         featureSet) + '_person_' + str(person) + '_initialSamples_' + str(
                        #         initialSamples) + '_seed_' + str(
                        #         seed) + '.npy', CM_QDA_incre_threshold_09)

                    if nameFile is not None:
                        results.to_csv(nameFile)

                    idx += 1


def evaluation_Supervised(featureSet, nameFile, startPerson, endPerson, printR, initialSamples, initialTime, finalTime,
                          typeDatabase):
    scaler = preprocessing.MinMaxScaler()

    results = pd.DataFrame(
        columns=['Feature Set', 'person', 'exp_time', 'shots in traning', 'shot_class', 'day', 'unlabeled Gesture'])

    for metric in ['precision', 'recall']:
        for typeDA in ['LDA', 'QDA']:
            results[metric + '_' + typeDA + '_' + 'weak1'] = ''
            results[metric + '_' + typeDA + '_' + 'weak2'] = ''
            results[metric + '_' + typeDA + '_' + 'weak3'] = ''
            results[metric + '_' + typeDA + '_' + 'weak4'] = ''

    idx = 0

    for person in range(startPerson, endPerson + 1):
        # Upload Data
        dataMatrix, classes, numberShots, allFeatures, numSamples, days, testSamples = uploadDatabases(
            typeDatabase, person, featureSet)

        for seed in range(initialTime, finalTime + 1):
            for day in range(1, days + 1):
                np.random.seed(person + seed * 1000 + day * 10000)  # to get a different seed per each person
                labeledGestures = []
                testGestures = []
                for cla in range(1, classes + 1):
                    repetitions = np.random.choice(numberShots, numberShots, replace=False) + 1
                    labeledGestures += [[shot, cla, day] for shot in repetitions[:testSamples]]
                    testGestures += [[shot, cla, day] for shot in repetitions[testSamples:]]

                print(labeledGestures)
                print(testGestures)

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
                    data_pre = np.hstack((pca_model.transform(
                        scaler.transform(dataMatrix[:, :allFeatures])), dataMatrix[:, allFeatures:]))
                    print('features', np.size(labeledGesturesFeatures, axis=1))
                    weakModel2 = currentDistributionValues2(data_pre, np.size(labeledGesturesFeatures, axis=1),
                                                            labeledGestures, person, classes)
                    weakModel3 = currentDistributionValues3(data_pre, np.size(labeledGesturesFeatures, axis=1),
                                                            labeledGestures, person, classes)
                    weakModel4 = currentDistributionValues4(data_pre, np.size(labeledGesturesFeatures, axis=1),
                                                            labeledGestures, person, classes)
                else:
                    data_pre = np.hstack((scaler.transform(dataMatrix[:, :allFeatures]), dataMatrix[:, allFeatures:]))
                    weakModel2 = currentDistributionValues2(data_pre, np.size(labeledGesturesFeatures, axis=1),
                                                            labeledGestures, person, classes)
                    weakModel3 = currentDistributionValues3(data_pre, np.size(labeledGesturesFeatures, axis=1),
                                                            labeledGestures, person, classes)
                    weakModel4 = currentDistributionValues4(data_pre, np.size(labeledGesturesFeatures, axis=1),
                                                            labeledGestures, person, classes)

                weakModel1 = currentDistributionValues(
                    labeledGesturesFeatures, labeledGesturesLabels, classes, np.size(labeledGesturesFeatures, axis=1),
                    oldData=False)

                results.at[idx, 'Feature Set'] = featureSet
                results.at[idx, 'person'] = person
                results.at[idx, 'exp_time'] = seed
                results.at[idx, 'shots in traning'] = initialSamples
                results.at[idx, 'day'] = day

                type_DA = 'LDA'
                name = type_DA + '_' + 'weak1'
                results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                    idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                    testGesturesFeatures, testGesturesLabels, weakModel1, classes)
                name = type_DA + '_' + 'weak2'
                results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                    idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                    testGesturesFeatures, testGesturesLabels, weakModel2, classes)
                name = type_DA + '_' + 'weak3'
                results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                    idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                    testGesturesFeatures, testGesturesLabels, weakModel3, classes)
                name = type_DA + '_' + 'weak4'
                results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                    idx, 'recall_' + name] = DA_Classifiers.accuracyModelLDA(
                    testGesturesFeatures, testGesturesLabels, weakModel4, classes)

                ###################################
                type_DA = 'QDA'
                name = type_DA + '_' + 'weak1'
                results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                    idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                    testGesturesFeatures, testGesturesLabels, weakModel1, classes)
                name = type_DA + '_' + 'weak2'
                results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                    idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                    testGesturesFeatures, testGesturesLabels, weakModel2, classes)
                name = type_DA + '_' + 'weak3'
                results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                    idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                    testGesturesFeatures, testGesturesLabels, weakModel3, classes)
                name = type_DA + '_' + 'weak4'
                results.at[idx, 'acc_' + name], results.at[idx, 'precision_' + name], results.at[
                    idx, 'recall_' + name] = DA_Classifiers.accuracyModelQDA(
                    testGesturesFeatures, testGesturesLabels, weakModel4, classes)

                if printR:
                    print('person', person, 'day', day, 'exp', seed)
                    for type_DA in ['LDA', 'QDA']:
                        for model in ['weak1', 'weak2', 'weak3', 'weak4']:
                            name = type_DA + '_' + model
                            print('acc_' + name, results.loc[idx, 'acc_' + name])

                if nameFile is not None:
                    results.to_csv(nameFile)

                idx += 1


def accuracyPerGesture(CM):
    # classes = len(CM)
    # auxColumns = []
    # for cla in range(classes):
    #     auxColumns.append('CL' + str(cla + 1))
    # confMatrix = CM[auxColumns].values
    return np.diagonal(CM).sum() / np.sum(CM)


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


def currentDistributionValues4(dataMatrix, allFeatures, setGestures, person, classes):
    currentValues = pd.DataFrame(
        columns=['cov', 'mean', 'class', 'N', 'N_cov', 'LDAcov', 'Total_LDAcov', 'N_LDA'])
    for cla in range(classes):
        currentValues.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
        currentValues.at[cla, 'mean'] = np.zeros(allFeatures)

    N_Total = np.zeros(classes)
    count = np.zeros(classes)
    for Lgesture in setGestures:
        Features = list(dataMatrix[
                        (dataMatrix[:, allFeatures + 1] == person) &
                        (dataMatrix[:, allFeatures] == Lgesture[2]) &
                        (dataMatrix[:, allFeatures + 3] == Lgesture[0]) &
                        (dataMatrix[:, allFeatures + 2] == Lgesture[1]), 0:allFeatures])
        Labels = list(dataMatrix[
                          (dataMatrix[:, allFeatures + 1] == person) &
                          (dataMatrix[:, allFeatures] == Lgesture[2]) &
                          (dataMatrix[:, allFeatures + 3] == Lgesture[0]) &
                          (dataMatrix[:, allFeatures + 2] == Lgesture[1]), allFeatures + 2].T)
        N = len(Labels)
        N_Total[Lgesture[1] - 1] += N
        currentValues.at[Lgesture[1] - 1, 'cov'] += np.cov(Features, rowvar=False) * (N - 1)
        currentValues.at[Lgesture[1] - 1, 'mean'] += np.sum(Features, axis=0)
        count[Lgesture[1] - 1] += 1

    LDA_cov = np.zeros((allFeatures, allFeatures))
    for cla in range(classes):
        LDA_cov += currentValues.loc[cla, 'cov']
        currentValues.at[cla, 'cov'] = currentValues.loc[cla, 'cov'] / (N_Total[cla] - count[cla])
        currentValues.at[cla, 'mean'] = currentValues.loc[cla, 'mean'] / N_Total[cla]
        currentValues.at[cla, 'class'] = cla + 1
        currentValues.at[cla, 'N'] = N_Total[cla]
        currentValues.at[cla, 'N_cov'] = N_Total[cla] - 1

    currentValues.at[0, 'LDAcov'] = LDA_cov / (np.sum(N_Total) - np.sum(count))
    return currentValues


def currentDistributionValues3(dataMatrix, allFeatures, setGestures, person, classes):
    currentValues = pd.DataFrame(
        columns=['cov', 'mean', 'class', 'N', 'N_cov', 'LDAcov', 'Total_LDAcov', 'N_LDA'])
    for cla in range(classes):
        currentValues.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
        currentValues.at[cla, 'mean'] = np.zeros(allFeatures)
    N_Total = np.zeros(classes)
    count = np.zeros(classes)
    for Lgesture in setGestures:
        Features = list(dataMatrix[
                        (dataMatrix[:, allFeatures + 1] == person) &
                        (dataMatrix[:, allFeatures] == Lgesture[2]) &
                        (dataMatrix[:, allFeatures + 3] == Lgesture[0]) &
                        (dataMatrix[:, allFeatures + 2] == Lgesture[1]), 0:allFeatures])
        Labels = list(dataMatrix[
                          (dataMatrix[:, allFeatures + 1] == person) &
                          (dataMatrix[:, allFeatures] == Lgesture[2]) &
                          (dataMatrix[:, allFeatures + 3] == Lgesture[0]) &
                          (dataMatrix[:, allFeatures + 2] == Lgesture[1]), allFeatures + 2].T)
        N = len(Labels)
        N_Total[Lgesture[1] - 1] += N
        currentValues.at[Lgesture[1] - 1, 'cov'] += np.cov(Features, rowvar=False) * (N - 1)
        currentValues.at[Lgesture[1] - 1, 'mean'] += np.sum(Features, axis=0)
        count[Lgesture[1] - 1] += 1
    for cla in range(classes):
        currentValues.at[cla, 'cov'] = currentValues.loc[cla, 'cov'] / (N_Total[cla] - 1)
        currentValues.at[cla, 'mean'] = currentValues.loc[cla, 'mean'] / N_Total[cla]
        currentValues.at[cla, 'class'] = cla + 1
        currentValues.at[cla, 'N'] = N_Total[cla]
        currentValues.at[cla, 'N_cov'] = N_Total[cla] - 1

    currentValues.at[0, 'LDAcov'] = DA_Classifiers.LDA_Cov_weights(currentValues)
    return currentValues


def currentDistributionValues2(dataMatrix, allFeatures, setGestures, person, classes):
    currentValues = pd.DataFrame(
        columns=['cov', 'mean', 'class', 'N', 'N_cov', 'LDAcov', 'Total_LDAcov', 'N_LDA'])
    for cla in range(classes):
        currentValues.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
        currentValues.at[cla, 'mean'] = np.zeros(allFeatures)
    N_Total = np.zeros(classes)
    count = np.zeros(classes)
    for Lgesture in setGestures:
        Features = list(dataMatrix[
                        (dataMatrix[:, allFeatures + 1] == person) &
                        (dataMatrix[:, allFeatures] == Lgesture[2]) &
                        (dataMatrix[:, allFeatures + 3] == Lgesture[0]) &
                        (dataMatrix[:, allFeatures + 2] == Lgesture[1]), 0:allFeatures])
        Labels = list(dataMatrix[
                          (dataMatrix[:, allFeatures + 1] == person) &
                          (dataMatrix[:, allFeatures] == Lgesture[2]) &
                          (dataMatrix[:, allFeatures + 3] == Lgesture[0]) &
                          (dataMatrix[:, allFeatures + 2] == Lgesture[1]), allFeatures + 2].T)
        N = len(Labels)
        N_Total[Lgesture[1] - 1] += N
        currentValues.at[Lgesture[1] - 1, 'cov'] += np.cov(Features, rowvar=False) * (N - 1)
        currentValues.at[Lgesture[1] - 1, 'mean'] += np.sum(Features, axis=0)
        count[Lgesture[1] - 1] += 1

    for cla in range(classes):
        currentValues.at[cla, 'cov'] = currentValues.loc[cla, 'cov'] / (N_Total[cla] - count[cla])
        currentValues.at[cla, 'mean'] = currentValues.loc[cla, 'mean'] / N_Total[cla]
        currentValues.at[cla, 'class'] = cla + 1
        currentValues.at[cla, 'N'] = N_Total[cla]
        currentValues.at[cla, 'N_cov'] = N_Total[cla] - 1

    currentValues.at[0, 'LDAcov'] = DA_Classifiers.LDA_Cov_weights(currentValues)
    return currentValues


def currentDistributionValues(trainFeatures, trainLabels, classes, allFeatures, oldData):
    currentValues = pd.DataFrame(
        columns=['cov', 'mean', 'class', 'N', 'N_cov', 'LDAcov', 'Total_LDAcov', 'N_LDA'])
    trainLabelsAux = trainLabels[np.newaxis]
    Matrix = np.hstack((trainFeatures, trainLabelsAux.T))

    for cla in range(classes):
        X = Matrix[np.where((Matrix[:, allFeatures] == cla + 1)), 0:allFeatures][0]
        currentValues.at[cla, 'cov'] = np.cov(X, rowvar=False)
        currentValues.at[cla, 'mean'] = np.mean(X, axis=0)
        currentValues.at[cla, 'class'] = cla + 1
        N = np.size(X, axis=0)
        currentValues.at[cla, 'N'] = N
        currentValues.at[cla, 'N_cov'] = N - 1

    currentValues.at[0, 'N_LDA'] = np.size(Matrix[:, 0:allFeatures], axis=0) - classes
    currentValues.at[0, 'LDAcov'] = DA_Classifiers.LDA_Cov_weights(currentValues)
    return currentValues


def covTotal_class(x, mean):
    cov_data = np.zeros((np.size(x, axis=1), np.size(x, axis=1)))
    for data_i in x:
        x_mean = np.resize(data_i - mean, (len(data_i), 1))
        cov_data += np.dot(x_mean, x_mean.T.conj())
    return cov_data


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

# def evaluation(featureSet, nameFile, startPerson, endPerson, printR, shotStart, initialTime, finalTime, typeDatabase):
#     scaler = preprocessing.MinMaxScaler()
#
#     results = pd.DataFrame(
#         columns=['Feature Set', 'person', 'exp_time', 'shots in traning', 'shot_class', 'day', 'unlabeled Gesture'])
#
#     for metric in ['w_predicted', 'time']:
#         for typeDA in ['LDA', 'QDA']:
#             results[metric + '_' + typeDA + '_' + 'incre_proposed'] = ''
#             results[metric + '_' + typeDA + '_' + 'incre_labels'] = ''
#             results[metric + '_' + typeDA + '_' + 'incre_sequential'] = ''
#             results[metric + '_' + typeDA + '_' + 'incre_supervised'] = ''
#             for l in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
#                 results[metric + '_' + typeDA + '_' + 'incre_Nigam_' + str(l)] = ''
#             for l in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#                 results[metric + '_' + typeDA + '_' + 'incre_threshold_' + str(l)] = ''
#
#     idx = 0
#
#     for person in range(startPerson, endPerson + 1):
#         # Upload Data
#         dataMatrix, classes, numberShots, allFeatures, numSamples, days = uploadDatabases(
#             typeDatabase, person, featureSet)
#
#         if person >= 20 and typeDatabase == 'Cote':
#             numberShots = 12
#
#         for seed in range(initialTime, finalTime + 1):
#             np.random.seed(seed * 1000 + person)  # to get a different seed per each person
#             unlabeledGestures = []
#             labeledGestures = []
#             day = 1
#             for cla in range(1, classes + 1):
#                 repetitions = np.random.choice(numberShots, numberShots, replace=False) + 1
#                 unlabeledGestures += [[shot, cla, day] for shot in repetitions[shotStart:]]
#                 labeledGestures += [[shot, cla, day] for shot in repetitions[:shotStart]]
#
#             print(labeledGestures)
#             print(unlabeledGestures)
#             labeledGesturesFeatures = []
#             labeledGesturesLabels = []
#
#             for Lgesture in labeledGestures:
#                 labeledGesturesFeatures += list(dataMatrix[
#                                                 (dataMatrix[:, allFeatures + 1] == person) &
#                                                 (dataMatrix[:, allFeatures] == Lgesture[2]) &
#                                                 (dataMatrix[:, allFeatures + 3] == Lgesture[0]) &
#                                                 (dataMatrix[:, allFeatures + 2] == Lgesture[1]), 0:allFeatures])
#                 labeledGesturesLabels += list(dataMatrix[
#                                                   (dataMatrix[:, allFeatures + 1] == person) &
#                                                   (dataMatrix[:, allFeatures] == Lgesture[2]) &
#                                                   (dataMatrix[:, allFeatures + 3] == Lgesture[0]) &
#                                                   (dataMatrix[:, allFeatures + 2] == Lgesture[1]), allFeatures + 2].T)
#
#             labeledGesturesFeatures = np.array(labeledGesturesFeatures)
#             labeledGesturesLabels = np.array(labeledGesturesLabels)
#
#             labeledGesturesFeatures = scaler.fit_transform(labeledGesturesFeatures)
#
#             if typeDatabase == 'Capgmyo_dbb':
#                 pca_model = PCA(n_components=0.999, svd_solver='full')
#                 # pca_model = IncrementalPCA(n_components=10)
#                 pca_model.fit(labeledGesturesFeatures)
#                 labeledGesturesFeatures = pca_model.transform(labeledGesturesFeatures)
#                 print('features', np.size(labeledGesturesFeatures, axis=1))
#
#             weakModel = currentDistributionValues(
#                 labeledGesturesFeatures, labeledGesturesLabels, classes, np.size(labeledGesturesFeatures, axis=1),
#                 shotStart)
#
#             labeledGesturesFeatures, labeledGesturesLabels = systematicSampling(
#                 labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)
#             print('number samples', len(labeledGesturesLabels) / classes)
#
#             LDA_weak = weakModel.copy()
#             LDA_incre_proposed = weakModel.copy()
#             LDA_incre_label = weakModel.copy()
#             LDA_incre_sequential = weakModel.copy()
#             LDA_incre_sequential_probs = weakModel.copy()
#             LDA_incre_supervised = weakModel.copy()
#             LDA_incre_Nigam_01 = weakModel.copy()
#             LDA_incre_Nigam_02 = weakModel.copy()
#             LDA_incre_Nigam_04 = weakModel.copy()
#             LDA_incre_Nigam_06 = weakModel.copy()
#             LDA_incre_Nigam_08 = weakModel.copy()
#             LDA_incre_Nigam_10 = weakModel.copy()
#             LDA_incre_threshold_02 = weakModel.copy()
#             LDA_incre_threshold_03 = weakModel.copy()
#             LDA_incre_threshold_04 = weakModel.copy()
#             LDA_incre_threshold_05 = weakModel.copy()
#             LDA_incre_threshold_06 = weakModel.copy()
#             LDA_incre_threshold_07 = weakModel.copy()
#             LDA_incre_threshold_08 = weakModel.copy()
#             LDA_incre_threshold_09 = weakModel.copy()
#
#             QDA_weak = weakModel.copy()
#             QDA_incre_proposed = weakModel.copy()
#             QDA_incre_label = weakModel.copy()
#             QDA_incre_sequential = weakModel.copy()
#             QDA_incre_sequential_probs = weakModel.copy()
#             QDA_incre_supervised = weakModel.copy()
#             QDA_incre_Nigam_01 = weakModel.copy()
#             QDA_incre_Nigam_02 = weakModel.copy()
#             QDA_incre_Nigam_04 = weakModel.copy()
#             QDA_incre_Nigam_06 = weakModel.copy()
#             QDA_incre_Nigam_08 = weakModel.copy()
#             QDA_incre_Nigam_10 = weakModel.copy()
#             QDA_incre_threshold_02 = weakModel.copy()
#             QDA_incre_threshold_03 = weakModel.copy()
#             QDA_incre_threshold_04 = weakModel.copy()
#             QDA_incre_threshold_05 = weakModel.copy()
#             QDA_incre_threshold_06 = weakModel.copy()
#             QDA_incre_threshold_07 = weakModel.copy()
#             QDA_incre_threshold_08 = weakModel.copy()
#             QDA_incre_threshold_09 = weakModel.copy()
#
#             CM_LDA_weak = np.zeros((classes, classes))
#             CM_LDA_incre_proposed = np.zeros((classes, classes))
#             CM_LDA_incre_label = np.zeros((classes, classes))
#             CM_LDA_incre_sequential = np.zeros((classes, classes))
#             CM_LDA_incre_sequential_probs = np.zeros((classes, classes))
#             CM_LDA_incre_supervised = np.zeros((classes, classes))
#             CM_LDA_incre_Nigam_01 = np.zeros((classes, classes))
#             CM_LDA_incre_Nigam_02 = np.zeros((classes, classes))
#             CM_LDA_incre_Nigam_04 = np.zeros((classes, classes))
#             CM_LDA_incre_Nigam_06 = np.zeros((classes, classes))
#             CM_LDA_incre_Nigam_08 = np.zeros((classes, classes))
#             CM_LDA_incre_Nigam_10 = np.zeros((classes, classes))
#             CM_LDA_incre_threshold_02 = np.zeros((classes, classes))
#             CM_LDA_incre_threshold_03 = np.zeros((classes, classes))
#             CM_LDA_incre_threshold_04 = np.zeros((classes, classes))
#             CM_LDA_incre_threshold_05 = np.zeros((classes, classes))
#             CM_LDA_incre_threshold_06 = np.zeros((classes, classes))
#             CM_LDA_incre_threshold_07 = np.zeros((classes, classes))
#             CM_LDA_incre_threshold_08 = np.zeros((classes, classes))
#             CM_LDA_incre_threshold_09 = np.zeros((classes, classes))
#
#             CM_QDA_weak = np.zeros((classes, classes))
#             CM_QDA_incre_proposed = np.zeros((classes, classes))
#             CM_QDA_incre_label = np.zeros((classes, classes))
#             CM_QDA_incre_sequential = np.zeros((classes, classes))
#             CM_QDA_incre_sequential_probs = np.zeros((classes, classes))
#             CM_QDA_incre_supervised = np.zeros((classes, classes))
#             CM_QDA_incre_Nigam_01 = np.zeros((classes, classes))
#             CM_QDA_incre_Nigam_02 = np.zeros((classes, classes))
#             CM_QDA_incre_Nigam_04 = np.zeros((classes, classes))
#             CM_QDA_incre_Nigam_06 = np.zeros((classes, classes))
#             CM_QDA_incre_Nigam_08 = np.zeros((classes, classes))
#             CM_QDA_incre_Nigam_10 = np.zeros((classes, classes))
#             CM_QDA_incre_threshold_02 = np.zeros((classes, classes))
#             CM_QDA_incre_threshold_03 = np.zeros((classes, classes))
#             CM_QDA_incre_threshold_04 = np.zeros((classes, classes))
#             CM_QDA_incre_threshold_05 = np.zeros((classes, classes))
#             CM_QDA_incre_threshold_06 = np.zeros((classes, classes))
#             CM_QDA_incre_threshold_07 = np.zeros((classes, classes))
#             CM_QDA_incre_threshold_08 = np.zeros((classes, classes))
#             CM_QDA_incre_threshold_09 = np.zeros((classes, classes))
#
#             numberUnlabeledGestures = 1
#
#             for day in range(1, days + 1):
#                 if day == 1:
#                     permutationUnlabeledGestures = np.random.permutation(unlabeledGestures)
#                     print(list(permutationUnlabeledGestures))
#                 else:
#                     unlabeledGestures = []
#                     for cla in range(1, classes + 1):
#                         repetitions = np.random.choice(numberShots, numberShots, replace=False) + 1
#                         unlabeledGestures += [[shot, cla, day] for shot in repetitions]
#                     permutationUnlabeledGestures = np.random.permutation(unlabeledGestures)
#                     print(list(permutationUnlabeledGestures))
#
#                 # iauxttt = 0
#                 for rand in list(permutationUnlabeledGestures):
#                     trainFeatures = dataMatrix[
#                                     (dataMatrix[:, allFeatures] == rand[2]) & (
#                                             dataMatrix[:, allFeatures + 1] == person) & (
#                                             dataMatrix[:, allFeatures + 3] == rand[0]) & (
#                                             dataMatrix[:, allFeatures + 2] == rand[1]), 0:allFeatures]
#                     trainLabels = dataMatrix[
#                         (dataMatrix[:, allFeatures] == rand[2]) & (dataMatrix[:, allFeatures + 1] == person) & (
#                                 dataMatrix[:, allFeatures + 3] == rand[0]) & (
#                                 dataMatrix[:, allFeatures + 2] == rand[1]), allFeatures + 2]
#
#                     # print('\nreal label',rand[1])
#
#                     #####Pre-processing
#                     trainFeatures = scaler.transform(trainFeatures)
#                     if typeDatabase == 'Capgmyo_dbb':
#                         # pca_model.fit(trainFeatures)
#                         trainFeatures = pca_model.transform(trainFeatures)
#                         # print('features test', np.size(trainFeatures, axis=1))
#
#                     # if iauxttt == 0:
#                     #     accumLabeledGesturesFeatures = trainFeatures
#                     #     accumLabeledGesturesLabels = trainLabels
#                     #     iauxttt = 1
#                     # else:
#                     #     accumLabeledGesturesFeatures = np.vstack((accumLabeledGesturesFeatures, trainFeatures))
#                     #     accumLabeledGesturesLabels = np.hstack((accumLabeledGesturesLabels, trainLabels))
#
#                     type_DA = 'LDA'
#
#                     name = type_DA + '_' + 'weak'
#                     CM_LDA_weak[rand[1] - 1, :] += SemiSupervised.predicted_labels(trainFeatures, LDA_weak, classes,
#                                                                                    type_DA)
#
#                     name = type_DA + '_' + 'incre_proposed'
#                     LDA_incre_proposed, results.at[idx, 'time' + '_' + name], results.at[
#                         idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposed(
#                         LDA_incre_proposed, classes, trainFeatures, rand[1], labeledGesturesFeatures,
#                         labeledGesturesLabels, type_DA, CM_LDA_incre_proposed)
#                     print('time LDA', results.at[idx, 'time' + '_' + name])
#
#                     name = type_DA + '_' + 'incre_labels'
#                     LDA_incre_label, results.at[idx, 'time' + '_' + name], results.at[
#                         idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_labels(
#                         LDA_incre_label, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_label)
#
#                     name = type_DA + '_' + 'incre_sequential'
#                     LDA_incre_sequential, results.at[idx, 'time' + '_' + name], results.at[
#                         idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_sequential_labels(
#                         LDA_incre_sequential, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_sequential)
#
#                     name = type_DA + '_' + 'incre_sequential_probs'
#                     LDA_incre_sequential_probs, results.at[idx, 'time' + '_' + name], results.at[
#                         idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_sequential_probs(
#                         LDA_incre_sequential_probs, classes, trainFeatures, rand[1], type_DA,
#                         CM_LDA_incre_sequential_probs)
#
#                     name = type_DA + '_' + 'incre_supervised'
#                     LDA_incre_supervised, results.at[idx, 'time' + '_' + name], results.at[
#                         idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_supervised(
#                         LDA_incre_supervised, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_supervised)
#
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(0.1)
#                     # LDA_incre_Nigam_01, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                     #     LDA_incre_Nigam_01, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_Nigam_01, weight_Nigam=0.1)
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(0.2)
#                     # LDA_incre_Nigam_02, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                     #     LDA_incre_Nigam_02, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_Nigam_02, weight_Nigam=0.2)
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(0.4)
#                     # LDA_incre_Nigam_04, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                     #     LDA_incre_Nigam_04, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_Nigam_04, weight_Nigam=0.4)
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(0.6)
#                     # LDA_incre_Nigam_06, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                     #     LDA_incre_Nigam_06, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_Nigam_06, weight_Nigam=0.6)
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(0.8)
#                     # LDA_incre_Nigam_08, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                     #     LDA_incre_Nigam_08, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_Nigam_08, weight_Nigam=0.8)
#
#                     name = type_DA + '_' + 'incre_Nigam_' + str(1.0)
#                     LDA_incre_Nigam_10, results.at[idx, 'time' + '_' + name], results.at[
#                         idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                         LDA_incre_Nigam_10, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_Nigam_10,
#                         weight_Nigam=1)
#
#                     #
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.2)
#                     # LDA_incre_threshold_02, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                     #     LDA_incre_threshold_02, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_02,
#                     #     threshold=0.2)
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.3)
#                     # LDA_incre_threshold_03, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                     #     LDA_incre_threshold_03, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_03,
#                     #     threshold=0.3)
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.4)
#                     # LDA_incre_threshold_04, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                     #     LDA_incre_threshold_04, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_04,
#                     #     threshold=0.4)
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.5)
#                     # LDA_incre_threshold_05, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                     #     LDA_incre_threshold_05, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_05,
#                     #     threshold=0.5)
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.6)
#                     # LDA_incre_threshold_06, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                     #     LDA_incre_threshold_06, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_06,
#                     #     threshold=0.6)
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.7)
#                     # LDA_incre_threshold_07, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                     #     LDA_incre_threshold_07, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_07,
#                     #     threshold=0.7)
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.8)
#                     # LDA_incre_threshold_08, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                     #     LDA_incre_threshold_08, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_08,
#                     #     threshold=0.8)
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.9)
#                     # LDA_incre_threshold_09, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                     #     LDA_incre_threshold_09, classes, trainFeatures, rand[1], type_DA, CM_LDA_incre_threshold_09,
#                     #     threshold=0.9)
#
#                     ###################################
#                     type_DA = 'QDA'
#
#                     name = type_DA + '_' + 'weak'
#                     CM_QDA_weak[rand[1] - 1, :] += SemiSupervised.predicted_labels(trainFeatures, QDA_weak, classes,
#                                                                                    type_DA)
#                     name = type_DA + '_' + 'incre_proposed'
#                     QDA_incre_proposed, results.at[idx, 'time' + '_' + name], results.at[
#                         idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposed(
#                         QDA_incre_proposed, classes, trainFeatures, rand[1], labeledGesturesFeatures,
#                         labeledGesturesLabels, type_DA, CM_QDA_incre_proposed)
#                     print('time QDA', results.at[idx, 'time' + '_' + name])
#
#                     name = type_DA + '_' + 'incre_labels'
#                     QDA_incre_label, results.at[idx, 'time' + '_' + name], results.at[
#                         idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_labels(
#                         QDA_incre_label, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_label)
#
#                     name = type_DA + '_' + 'incre_sequential'
#                     QDA_incre_sequential, results.at[idx, 'time' + '_' + name], results.at[
#                         idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_sequential_labels(
#                         QDA_incre_sequential, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_sequential)
#
#                     name = type_DA + '_' + 'incre_sequential_probs'
#                     QDA_incre_sequential_probs, results.at[idx, 'time' + '_' + name], results.at[
#                         idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_sequential_probs(
#                         QDA_incre_sequential_probs, classes, trainFeatures, rand[1], type_DA,
#                         CM_QDA_incre_sequential_probs)
#
#                     name = type_DA + '_' + 'incre_supervised'
#                     QDA_incre_supervised, results.at[idx, 'time' + '_' + name], results.at[
#                         idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_supervised(
#                         QDA_incre_supervised, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_supervised)
#
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(0.1)
#                     # QDA_incre_Nigam_01, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                     #     QDA_incre_Nigam_01, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_Nigam_01, weight_Nigam=0.1)
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(0.2)
#                     # QDA_incre_Nigam_02, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                     #     QDA_incre_Nigam_02, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_Nigam_02, weight_Nigam=0.2)
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(0.4)
#                     # QDA_incre_Nigam_04, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                     #     QDA_incre_Nigam_04, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_Nigam_04, weight_Nigam=0.4)
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(0.6)
#                     # QDA_incre_Nigam_06, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                     #     QDA_incre_Nigam_06, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_Nigam_06, weight_Nigam=0.6)
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(0.8)
#                     # QDA_incre_Nigam_08, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                     #     QDA_incre_Nigam_08, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_Nigam_08, weight_Nigam=0.8)
#
#                     name = type_DA + '_' + 'incre_Nigam_' + str(1.0)
#                     QDA_incre_Nigam_10, results.at[idx, 'time' + '_' + name], results.at[
#                         idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                         QDA_incre_Nigam_10, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_Nigam_10,
#                         weight_Nigam=1)
#
#                     #
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.2)
#                     # QDA_incre_threshold_02, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                     #     QDA_incre_threshold_02, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_02,
#                     #     threshold=0.2)
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.3)
#                     # QDA_incre_threshold_03, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                     #     QDA_incre_threshold_03, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_03,
#                     #     threshold=0.3)
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.4)
#                     # QDA_incre_threshold_04, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                     #     QDA_incre_threshold_04, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_04,
#                     #     threshold=0.4)
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.5)
#                     # QDA_incre_threshold_05, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                     #     QDA_incre_threshold_05, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_05,
#                     #     threshold=0.5)
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.6)
#                     # QDA_incre_threshold_06, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                     #     QDA_incre_threshold_06, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_06,
#                     #     threshold=0.6)
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.7)
#                     # QDA_incre_threshold_07, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                     #     QDA_incre_threshold_07, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_07,
#                     #     threshold=0.7)
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.8)
#                     # QDA_incre_threshold_08, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                     #     QDA_incre_threshold_08, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_08,
#                     #     threshold=0.8)
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.9)
#                     # QDA_incre_threshold_09, results.at[idx, 'time' + '_' + name], results.at[
#                     #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                     #     QDA_incre_threshold_09, classes, trainFeatures, rand[1], type_DA, CM_QDA_incre_threshold_09,
#                     #     threshold=0.9)
#
#                     results.at[idx, 'Feature Set'] = featureSet
#                     results.at[idx, 'person'] = person
#                     results.at[idx, 'exp_time'] = seed
#                     results.at[idx, 'shots in traning'] = shotStart
#                     results.at[idx, 'day'] = day
#                     results.at[idx, 'shot_class'] = rand
#                     results.at[idx, 'unlabeled Gesture'] = numberUnlabeledGestures
#                     numberUnlabeledGestures += 1
#                     if nameFile is not None:
#                         results.to_csv(nameFile)
#
#                     # np.save(
#                     #     'confussionMatrix/' + 'LDA_weak' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_LDA_weak)
#                     # np.save(
#                     #     'confussionMatrix/' + 'LDA_incre_proposed' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_LDA_incre_proposed)
#                     # np.save(
#                     #     'confussionMatrix/' + 'LDA_incre_label' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_LDA_incre_label)
#                     # np.save(
#                     #     'confussionMatrix/' + 'LDA_incre_sequential' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_LDA_incre_sequential)
#                     # np.save(
#                     #     'confussionMatrix/' + 'LDA_incre_sequential_probs' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_LDA_incre_sequential_probs)
#                     # np.save(
#                     #     'confussionMatrix/' + 'LDA_incre_supervised' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_LDA_incre_supervised)
#                     # np.save(
#                     #     'confussionMatrix/' + 'LDA_incre_Nigam_01' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_LDA_incre_Nigam_01)
#                     # np.save(
#                     #     'confussionMatrix/' + 'LDA_incre_Nigam_02' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_LDA_incre_Nigam_02)
#                     # np.save(
#                     #     'confussionMatrix/' + 'LDA_incre_Nigam_04' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_LDA_incre_Nigam_04)
#                     # np.save(
#                     #     'confussionMatrix/' + 'LDA_incre_Nigam_06' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_LDA_incre_Nigam_06)
#                     # np.save(
#                     #     'confussionMatrix/' + 'LDA_incre_Nigam_08' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_LDA_incre_Nigam_08)
#                     # np.save(
#                     #     'confussionMatrix/' + 'LDA_incre_Nigam_10' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_LDA_incre_Nigam_10)
#                     # np.save(
#                     #     'confussionMatrix/' + 'LDA_incre_threshold_02' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_LDA_incre_threshold_02)
#                     # np.save(
#                     #     'confussionMatrix/' + 'LDA_incre_threshold_03' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_LDA_incre_threshold_03)
#                     # np.save(
#                     #     'confussionMatrix/' + 'LDA_incre_threshold_04' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_LDA_incre_threshold_04)
#                     # np.save(
#                     #     'confussionMatrix/' + 'LDA_incre_threshold_05' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_LDA_incre_threshold_05)
#                     # np.save(
#                     #     'confussionMatrix/' + 'LDA_incre_threshold_06' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_LDA_incre_threshold_06)
#                     # np.save(
#                     #     'confussionMatrix/' + 'LDA_incre_threshold_07' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_LDA_incre_threshold_07)
#                     # np.save(
#                     #     'confussionMatrix/' + 'LDA_incre_threshold_08' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_LDA_incre_threshold_08)
#                     # np.save(
#                     #     'confussionMatrix/' + 'LDA_incre_threshold_09' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_LDA_incre_threshold_09)
#
#                     ###############################QDA##########################################################
#
#                     # np.save(
#                     #     'confussionMatrix/' + 'QDA_weak' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_QDA_weak)
#                     #
#                     # np.save(
#                     #     'confussionMatrix/' + 'QDA_incre_proposed' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_QDA_incre_proposed)
#                     # np.save(
#                     #     'confussionMatrix/' + 'QDA_incre_label' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_QDA_incre_label)
#                     # np.save(
#                     #     'confussionMatrix/' + 'QDA_incre_sequential' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_QDA_incre_sequential)
#                     # np.save(
#                     #     'confussionMatrix/' + 'QDA_incre_sequential_probs' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_QDA_incre_sequential_probs)
#                     # np.save(
#                     #     'confussionMatrix/' + 'QDA_incre_supervised' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_QDA_incre_supervised)
#                     # np.save(
#                     #     'confussionMatrix/' + 'QDA_incre_Nigam_01' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_QDA_incre_Nigam_01)
#                     # np.save(
#                     #     'confussionMatrix/' + 'QDA_incre_Nigam_02' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_QDA_incre_Nigam_02)
#                     # np.save(
#                     #     'confussionMatrix/' + 'QDA_incre_Nigam_04' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_QDA_incre_Nigam_04)
#                     # np.save(
#                     #     'confussionMatrix/' + 'QDA_incre_Nigam_06' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_QDA_incre_Nigam_06)
#                     # np.save(
#                     #     'confussionMatrix/' + 'QDA_incre_Nigam_08' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_QDA_incre_Nigam_08)
#                     # np.save(
#                     #     'confussionMatrix/' + 'QDA_incre_Nigam_10' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_QDA_incre_Nigam_10)
#                     # np.save(
#                     #     'confussionMatrix/' + 'QDA_incre_threshold_02' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_QDA_incre_threshold_02)
#                     # np.save(
#                     #     'confussionMatrix/' + 'QDA_incre_threshold_03' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_QDA_incre_threshold_03)
#                     # np.save(
#                     #     'confussionMatrix/' + 'QDA_incre_threshold_04' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_QDA_incre_threshold_04)
#                     # np.save(
#                     #     'confussionMatrix/' + 'QDA_incre_threshold_05' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_QDA_incre_threshold_05)
#                     # np.save(
#                     #     'confussionMatrix/' + 'QDA_incre_threshold_06' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_QDA_incre_threshold_06)
#                     # np.save(
#                     #     'confussionMatrix/' + 'QDA_incre_threshold_07' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_QDA_incre_threshold_07)
#                     # np.save(
#                     #     'confussionMatrix/' + 'QDA_incre_threshold_08' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_QDA_incre_threshold_08)
#                     # np.save(
#                     #     'confussionMatrix/' + 'QDA_incre_threshold_09' + '_' + typeDatabase + '_featureSet_' + str(
#                     #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                     #         seed) + '.npy', CM_QDA_incre_threshold_09)
#
#                     if np.all(permutationUnlabeledGestures[-1] == rand):
#                         type_DA = 'LDA'
#                         name = type_DA + '_' + 'weak'
#                         results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_weak)
#                         name = type_DA + '_' + 'incre_proposed'
#                         results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_proposed)
#                         name = type_DA + '_' + 'incre_labels'
#                         results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_label)
#                         name = type_DA + '_' + 'incre_sequential'
#                         results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_sequential)
#                         name = type_DA + '_' + 'incre_sequential_probs'
#                         results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_sequential_probs)
#                         name = type_DA + '_' + 'incre_supervised'
#                         results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_supervised)
#                         # name = type_DA + '_' + 'incre_Nigam_' + str(0.1)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_Nigam_01)
#                         # name = type_DA + '_' + 'incre_Nigam_' + str(0.2)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_Nigam_02)
#                         # name = type_DA + '_' + 'incre_Nigam_' + str(0.4)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_Nigam_04)
#                         # name = type_DA + '_' + 'incre_Nigam_' + str(0.6)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_Nigam_06)
#                         # name = type_DA + '_' + 'incre_Nigam_' + str(0.8)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_Nigam_08)
#                         name = type_DA + '_' + 'incre_Nigam_' + str(1.0)
#                         results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_Nigam_10)
#                         # name = type_DA + '_' + 'incre_threshold_' + str(0.2)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_threshold_02)
#                         # name = type_DA + '_' + 'incre_threshold_' + str(0.3)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_threshold_03)
#                         # name = type_DA + '_' + 'incre_threshold_' + str(0.4)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_threshold_04)
#                         # name = type_DA + '_' + 'incre_threshold_' + str(0.5)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_threshold_05)
#                         # name = type_DA + '_' + 'incre_threshold_' + str(0.6)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_threshold_06)
#                         # name = type_DA + '_' + 'incre_threshold_' + str(0.7)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_threshold_07)
#                         # name = type_DA + '_' + 'incre_threshold_' + str(0.8)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_threshold_08)
#                         # name = type_DA + '_' + 'incre_threshold_' + str(0.9)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_LDA_incre_threshold_09)
#                         ###################################
#                         type_DA = 'QDA'
#                         name = type_DA + '_' + 'weak'
#                         results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_weak)
#                         name = type_DA + '_' + 'incre_proposed'
#                         results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_proposed)
#                         name = type_DA + '_' + 'incre_labels'
#                         results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_label)
#                         name = type_DA + '_' + 'incre_sequential'
#                         results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_sequential)
#                         name = type_DA + '_' + 'incre_sequential_probs'
#                         results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_sequential_probs)
#                         name = type_DA + '_' + 'incre_supervised'
#                         results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_supervised)
#                         # name = type_DA + '_' + 'incre_Nigam_' + str(0.1)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_Nigam_01)
#                         # name = type_DA + '_' + 'incre_Nigam_' + str(0.2)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_Nigam_02)
#                         # name = type_DA + '_' + 'incre_Nigam_' + str(0.4)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_Nigam_04)
#                         # name = type_DA + '_' + 'incre_Nigam_' + str(0.6)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_Nigam_06)
#                         # name = type_DA + '_' + 'incre_Nigam_' + str(0.8)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_Nigam_08)
#                         name = type_DA + '_' + 'incre_Nigam_' + str(1.0)
#                         results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_Nigam_10)
#                         # name = type_DA + '_' + 'incre_threshold_' + str(0.2)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_threshold_02)
#                         # name = type_DA + '_' + 'incre_threshold_' + str(0.3)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_threshold_03)
#                         # name = type_DA + '_' + 'incre_threshold_' + str(0.4)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_threshold_04)
#                         # name = type_DA + '_' + 'incre_threshold_' + str(0.5)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_threshold_05)
#                         # name = type_DA + '_' + 'incre_threshold_' + str(0.6)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_threshold_06)
#                         # name = type_DA + '_' + 'incre_threshold_' + str(0.7)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_threshold_07)
#                         # name = type_DA + '_' + 'incre_threshold_' + str(0.8)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_threshold_08)
#                         # name = type_DA + '_' + 'incre_threshold_' + str(0.9)
#                         # results.at[idx, 'acc_' + name] = accuracyPerGesture(CM_QDA_incre_threshold_09)
#
#                         if printR:
#                             print('person', person)
#                             for type_DA in ['LDA', 'QDA']:
#                                 for model in ['weak', 'incre_proposed', 'incre_labels', 'incre_sequential',
#                                               'incre_sequential_probs', 'incre_supervised',
#                                               'incre_Nigam_' + str(1.0)]:
#                                     name = type_DA + '_' + model
#                                     print('acc_' + name, results.loc[idx, 'acc_' + name])
#
#                             # print('LDA')
#                             # print(CM_LDA_incre_proposed)
#                             # print(CM_LDA_weak)
#                             # print(CM_LDA_incre_supervised)
#                             # print('QDA')
#                             # print(CM_QDA_incre_proposed)
#                             # print(CM_QDA_weak)
#                             # print(CM_QDA_incre_supervised)
#
#                     idx += 1


# def evaluation(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson, endPerson,
#                allFeatures, printR, shotStart, initialTime, finalTime, typeDatabase):
#     scaler = preprocessing.MinMaxScaler()
#     if typeDatabase == 'EPN':
#         evaluatedGestures = np.array(
#             [1, 5, 10, 15, 20, 40, 60, 80, 100, 120, numberShots * classes - shotStart * classes])
#         numSamples = 250
#     elif typeDatabase == 'Nina5':
#         evaluatedGestures = np.array([1, 5, 10, 15, 20, 30, 40, 50, 54, numberShots * classes - shotStart * classes])
#         numSamples = 20
#     elif typeDatabase == 'Cote':
#         evaluatedGestures = np.array([1, 5, 10, 15, 20, numberShots * classes - shotStart * classes])
#         numSamples = 120
#
#     results = pd.DataFrame(
#         columns=['Feature Set', 'person', 'exp_time', 'shots in traning', 'shot_class', 'unlabeled Gesture'])
#
#     for metric in ['precision', 'recall', 'w_predicted', 'time']:
#         for typeDA in ['LDA', 'QDA']:
#             if metric == 'precision' or metric == 'recall':
#                 results[metric + '_' + typeDA + '_' + 'weak'] = ''
#                 results[metric + '_' + typeDA + '_' + 'accum'] = ''
#                 # results[metric + '_' + typeDA + '_' + 'adapted'] = ''
#             results[metric + '_' + typeDA + '_' + 'incre_proposed'] = ''
#             results[metric + '_' + typeDA + '_' + 'incre_labels'] = ''
#             results[metric + '_' + typeDA + '_' + 'incre_sequential'] = ''
#             results[metric + '_' + typeDA + '_' + 'incre_supervised'] = ''
#             for l in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
#                 results[metric + '_' + typeDA + '_' + 'incre_Nigam_' + str(l)] = ''
#             for l in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#                 results[metric + '_' + typeDA + '_' + 'incre_threshold_' + str(l)] = ''
#
#     idx = 0
#
#     for person in range(startPerson, endPerson + 1):
#
#         for seed in range(initialTime, finalTime + 1):
#             testFeatures = \
#                 dataMatrix[(dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 1), :allFeatures]
#             testLabels = dataMatrix[
#                 (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 1), allFeatures + 2].T
#
#             np.random.seed(seed)
#             unlabeledGestures = []
#             labeledGestures = []
#             for cla in range(1, classes + 1):
#                 repetitions = np.random.choice(numberShots, numberShots, replace=False) + 1
#                 unlabeledGestures += [[shot, cla] for shot in repetitions[shotStart:]]
#                 labeledGestures += [[shot, cla] for shot in repetitions[:shotStart]]
#             # print(labeledGestures)
#             # print(unlabeledGestures)
#             permutationUnlabeledGestures = np.random.permutation(unlabeledGestures)
#
#             # ###########
#             # # print(permutationUnlabeledGestures)
#             # permutationUnlabeledGestures = permutationUnlabeledGestures[0:2]
#             # print(permutationUnlabeledGestures)
#             # #########
#
#             labeledGesturesFeatures = []
#             labeledGesturesLabels = []
#             numberLabeledGestureList = []
#             probLabeledGestures = []
#             numberLabeledGestures = 1
#             for Lgesture in labeledGestures:
#                 labeledGesturesFeatures += list(dataMatrix[
#                                                 (dataMatrix[:, allFeatures + 1] == person) &
#                                                 (dataMatrix[:, allFeatures] == 0) &
#                                                 (dataMatrix[:, allFeatures + 3] == Lgesture[0]) &
#                                                 (dataMatrix[:, allFeatures + 2] == Lgesture[1]), 0:allFeatures])
#                 labeledGesturesLabels += list(dataMatrix[
#                                                   (dataMatrix[:, allFeatures + 1] == person) &
#                                                   (dataMatrix[:, allFeatures] == 0) &
#                                                   (dataMatrix[:, allFeatures + 3] == Lgesture[0]) &
#                                                   (dataMatrix[:, allFeatures + 2] == Lgesture[1]), allFeatures + 2].T)
#                 numberSegments = len(dataMatrix[(dataMatrix[:, allFeatures + 1] == person) &
#                                                 (dataMatrix[:, allFeatures] == 0) &
#                                                 (dataMatrix[:, allFeatures + 3] == Lgesture[0]) &
#                                                 (dataMatrix[:, allFeatures + 2] == Lgesture[
#                                                     1]), allFeatures + 2].T)
#                 numberLabeledGestureList += list(np.ones(numberSegments) * numberLabeledGestures)
#                 auxProb = np.zeros((numberSegments, classes))
#                 auxProb[:, Lgesture[1] - 1] = np.ones(numberSegments)
#                 probLabeledGestures += list(auxProb)
#                 numberLabeledGestures += 1
#
#             labeledGesturesFeatures = np.array(labeledGesturesFeatures)
#             labeledGesturesLabels = np.array(labeledGesturesLabels)
#
#             labeledGesturesFeatures = scaler.fit_transform(labeledGesturesFeatures)
#             testFeatures = scaler.transform(testFeatures)
#
#             accumLabeledGesturesFeatures = labeledGesturesFeatures
#             accumLabeledGesturesLabels = labeledGesturesLabels
#             try:
#                 # adaptedModelLDA = pd.read_pickle(
#                 #     'pretrainedModels/adaptedModelLDA_' + typeDatabase + '_featureSet_' + str(
#                 #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                 #         seed) + '.pkl')
#                 # adaptedModelQDA = pd.read_pickle(
#                 #     'pretrainedModels/adaptedModelQDA_' + typeDatabase + '_featureSet_' + str(
#                 #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                 #         seed) + '.pkl')
#
#                 weakModel = pd.read_pickle(
#                     'pretrainedModels/weakModel_' + typeDatabase + '_featureSet_' + str(featureSet) + '_person_' + str(
#                         person) + '_shotStart_' + str(shotStart) + '_seed_' + str(seed) + '.pkl')
#
#                 print('Found the models')
#             except:
#                 print('Did not find the models')
#
#                 weakModel = currentDistributionValues(
#                     labeledGesturesFeatures, labeledGesturesLabels, classes, allFeatures, shotStart)
#
#                 weakModel.to_pickle(
#                     'pretrainedModels/weakModel_' + typeDatabase + '_featureSet_' + str(featureSet) + '_person_' + str(
#                         person) + '_shotStart_' + str(shotStart) + '_seed_' + str(seed) + '.pkl')
#
#                 # adaptive model
#                 # dataPK, _ = preprocessingPK(dataMatrix, allFeatures, scaler)
#                 # preTrainedDataMatrix = PKModels(dataPK, classes, peoplePriorK, person, allFeatures)
#                 #
#                 # k = 1 - (np.log(shotStart) / np.log(numberShots + 1))
#                 # step = 1
#                 #
#                 # adaptedModelLDA, _, _, _, _, _ = adaptive.OurModel(
#                 #     weakModel, preTrainedDataMatrix, classes, allFeatures, labeledGesturesFeatures,
#                 #     labeledGesturesLabels, step, 'LDA', k, shotStart)
#                 # adaptedModelQDA, _, _, _, _, _ = adaptive.OurModel(
#                 #     weakModel, preTrainedDataMatrix, classes, allFeatures, labeledGesturesFeatures,
#                 #     labeledGesturesLabels, step, 'QDA', k, shotStart)
#                 #
#                 # adaptedModelLDA.to_pickle(
#                 #     'pretrainedModels/adaptedModelLDA_' + typeDatabase + '_featureSet_' + str(
#                 #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                 #         seed) + '.pkl')
#                 # adaptedModelQDA.to_pickle(
#                 #     'pretrainedModels/adaptedModelQDA_' + typeDatabase + '_featureSet_' + str(
#                 #         featureSet) + '_person_' + str(person) + '_shotStart_' + str(shotStart) + '_seed_' + str(
#                 #         seed) + '.pkl')
#
#             labeledGesturesFeatures, labeledGesturesLabels = systematicSampling(
#                 labeledGesturesFeatures, labeledGesturesLabels, numSamples, classes)
#             print('number samples', len(labeledGesturesLabels) / classes)
#
#             numberUnlabeledGestures = 0
#
#             name = 'LDA' + '_' + 'weak'
#             results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                 idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelLDA(
#                 testFeatures, testLabels, weakModel, classes)
#             print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#
#             name = 'QDA' + '_' + 'weak'
#             results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                 idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelQDA(
#                 testFeatures, testLabels, weakModel, classes)
#             print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#
#             # name = 'LDA' + '_' + 'adapted'
#             # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#             #     idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelLDA(
#             #     testFeatures, testLabels, adaptedModelLDA, classes)
#             # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#             #
#             # name = 'QDA' + '_' + 'adapted'
#             # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#             #     idx, 'recall' + '_' + name], _ = DA_Classifiers.accuracyModelQDA(
#             #     testFeatures, testLabels, adaptedModelQDA, classes)
#             # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#
#             LDA_incre_proposed = weakModel.copy()
#             LDA_incre_label = weakModel.copy()
#             LDA_incre_sequential = weakModel.copy()
#             LDA_incre_supervised = weakModel.copy()
#             LDA_incre_Nigam_01 = weakModel.copy()
#             LDA_incre_Nigam_02 = weakModel.copy()
#             LDA_incre_Nigam_04 = weakModel.copy()
#             LDA_incre_Nigam_06 = weakModel.copy()
#             LDA_incre_Nigam_08 = weakModel.copy()
#             LDA_incre_Nigam_10 = weakModel.copy()
#             LDA_incre_threshold_02 = weakModel.copy()
#             LDA_incre_threshold_03 = weakModel.copy()
#             LDA_incre_threshold_04 = weakModel.copy()
#             LDA_incre_threshold_05 = weakModel.copy()
#             LDA_incre_threshold_06 = weakModel.copy()
#             LDA_incre_threshold_07 = weakModel.copy()
#             LDA_incre_threshold_08 = weakModel.copy()
#             LDA_incre_threshold_09 = weakModel.copy()
#
#             QDA_incre_proposed = weakModel.copy()
#             QDA_incre_label = weakModel.copy()
#             QDA_incre_sequential = weakModel.copy()
#             QDA_incre_supervised = weakModel.copy()
#             QDA_incre_Nigam_01 = weakModel.copy()
#             QDA_incre_Nigam_02 = weakModel.copy()
#             QDA_incre_Nigam_04 = weakModel.copy()
#             QDA_incre_Nigam_06 = weakModel.copy()
#             QDA_incre_Nigam_08 = weakModel.copy()
#             QDA_incre_Nigam_10 = weakModel.copy()
#             QDA_incre_threshold_02 = weakModel.copy()
#             QDA_incre_threshold_03 = weakModel.copy()
#             QDA_incre_threshold_04 = weakModel.copy()
#             QDA_incre_threshold_05 = weakModel.copy()
#             QDA_incre_threshold_06 = weakModel.copy()
#             QDA_incre_threshold_07 = weakModel.copy()
#             QDA_incre_threshold_08 = weakModel.copy()
#             QDA_incre_threshold_09 = weakModel.copy()
#
#             # LDA_incre_proposed_adapt = adaptedModelLDA.copy()
#             # QDA_incre_proposed_adapt = adaptedModelQDA.copy()
#
#             for rand in list(permutationUnlabeledGestures):
#                 trainFeatures = dataMatrix[
#                                 (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 0) &
#                                 (dataMatrix[:, allFeatures + 3] == rand[0]) & (
#                                         dataMatrix[:, allFeatures + 2] == rand[1]), 0:allFeatures]
#                 trainLabels = dataMatrix[
#                     (dataMatrix[:, allFeatures + 1] == person) & (dataMatrix[:, allFeatures] == 0) &
#                     (dataMatrix[:, allFeatures + 3] == rand[0]) & (
#                             dataMatrix[:, allFeatures + 2] == rand[1]), allFeatures + 2]
#
#                 # print('\nreal label',rand[1])
#
#                 trainFeatures = scaler.transform(trainFeatures)
#
#                 accumLabeledGesturesFeatures = np.vstack((accumLabeledGesturesFeatures, trainFeatures))
#                 accumLabeledGesturesLabels = np.hstack((accumLabeledGesturesLabels, trainLabels))
#
#                 numberUnlabeledGestures += 1
#
#                 type_DA = 'LDA'
#
#                 # name = type_DA + '_' + 'incre_proposed_adapt'
#                 # LDA_incre_proposed_adapt, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposedProbMSDA(
#                 #     LDA_incre_proposed_adapt, classes, trainFeatures, labeledGesturesFeatures, labeledGesturesLabels,
#                 #     'LDA')
#
#                 name = type_DA + '_' + 'incre_proposed'
#                 LDA_incre_proposed, results.at[idx, 'time' + '_' + name], results.at[
#                     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposed(
#                     LDA_incre_proposed, classes, trainFeatures, labeledGesturesFeatures, labeledGesturesLabels,
#                     'LDA')
#                 print('time LDA', results.at[idx, 'time' + '_' + name])
#
#                 name = type_DA + '_' + 'incre_labels'
#                 LDA_incre_label, results.at[idx, 'time' + '_' + name], results.at[
#                     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_labels(
#                     LDA_incre_label, classes, trainFeatures, 'LDA')
#
#                 # name = type_DA + '_' + 'incre_sequential'
#                 # LDA_incre_sequential, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_sequential_labels(
#                 #     LDA_incre_sequential, classes, trainFeatures, 'LDA')
#
#                 name = type_DA + '_' + 'incre_supervised'
#                 LDA_incre_supervised, results.at[idx, 'time' + '_' + name], results.at[
#                     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_supervised(
#                     LDA_incre_supervised, classes, trainFeatures, 'LDA', rand[1])
#
#                 # name = type_DA + '_' + 'incre_Nigam_' + str(0.1)
#                 # LDA_incre_Nigam_01, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                 #     LDA_incre_Nigam_01, classes, trainFeatures, 'LDA', weight_Nigam=0.1)
#                 # name = type_DA + '_' + 'incre_Nigam_' + str(0.2)
#                 # LDA_incre_Nigam_02, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                 #     LDA_incre_Nigam_02, classes, trainFeatures, 'LDA', weight_Nigam=0.2)
#                 # name = type_DA + '_' + 'incre_Nigam_' + str(0.4)
#                 # LDA_incre_Nigam_04, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                 #     LDA_incre_Nigam_04, classes, trainFeatures, 'LDA', weight_Nigam=0.4)
#                 # name = type_DA + '_' + 'incre_Nigam_' + str(0.6)
#                 # LDA_incre_Nigam_06, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                 #     LDA_incre_Nigam_06, classes, trainFeatures, 'LDA', weight_Nigam=0.6)
#                 # name = type_DA + '_' + 'incre_Nigam_' + str(0.8)
#                 # LDA_incre_Nigam_08, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                 #     LDA_incre_Nigam_08, classes, trainFeatures, 'LDA', weight_Nigam=0.8)
#                 # name = type_DA + '_' + 'incre_Nigam_' + str(1.0)
#                 # LDA_incre_Nigam_10, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                 #     LDA_incre_Nigam_10, classes, trainFeatures, 'LDA', weight_Nigam=1)
#                 #
#                 # name = type_DA + '_' + 'incre_threshold_' + str(0.2)
#                 # LDA_incre_threshold_02, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                 #     LDA_incre_threshold_02, classes, trainFeatures, 'LDA', threshold=0.2)
#                 # name = type_DA + '_' + 'incre_threshold_' + str(0.3)
#                 # LDA_incre_threshold_03, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                 #     LDA_incre_threshold_03, classes, trainFeatures, 'LDA', threshold=0.3)
#                 # name = type_DA + '_' + 'incre_threshold_' + str(0.4)
#                 # LDA_incre_threshold_04, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                 #     LDA_incre_threshold_04, classes, trainFeatures, 'LDA', threshold=0.4)
#                 # name = type_DA + '_' + 'incre_threshold_' + str(0.5)
#                 # LDA_incre_threshold_05, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                 #     LDA_incre_threshold_05, classes, trainFeatures, 'LDA', threshold=0.5)
#                 # name = type_DA + '_' + 'incre_threshold_' + str(0.6)
#                 # LDA_incre_threshold_06, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                 #     LDA_incre_threshold_06, classes, trainFeatures, 'LDA', threshold=0.6)
#                 # name = type_DA + '_' + 'incre_threshold_' + str(0.7)
#                 # LDA_incre_threshold_07, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                 #     LDA_incre_threshold_07, classes, trainFeatures, 'LDA', threshold=0.7)
#                 # name = type_DA + '_' + 'incre_threshold_' + str(0.8)
#                 # LDA_incre_threshold_08, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                 #     LDA_incre_threshold_08, classes, trainFeatures, 'LDA', threshold=0.8)
#                 # name = type_DA + '_' + 'incre_threshold_' + str(0.9)
#                 # LDA_incre_threshold_09, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                 #     LDA_incre_threshold_09, classes, trainFeatures, 'LDA', threshold=0.9)
#
#                 ###################################
#                 type_DA = 'QDA'
#
#                 # name = type_DA + '_' + 'incre_proposed_adapt'
#                 # QDA_incre_proposed_adapt, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposedProbMSDA(
#                 #     QDA_incre_proposed_adapt, classes, trainFeatures, labeledGesturesFeatures, labeledGesturesLabels,
#                 #     'QDA')
#
#                 name = type_DA + '_' + 'incre_proposed'
#                 QDA_incre_proposed, results.at[idx, 'time' + '_' + name], results.at[
#                     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_proposed(
#                     QDA_incre_proposed, classes, trainFeatures, labeledGesturesFeatures, labeledGesturesLabels,
#                     'QDA')
#                 print('time QDA', results.at[idx, 'time' + '_' + name])
#
#                 name = type_DA + '_' + 'incre_labels'
#                 QDA_incre_label, results.at[idx, 'time' + '_' + name], results.at[
#                     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_labels(
#                     QDA_incre_label, classes, trainFeatures, 'QDA')
#
#                 # name = type_DA + '_' + 'incre_sequential'
#                 # QDA_incre_sequential, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_sequential_labels(
#                 #     QDA_incre_sequential, classes, trainFeatures, 'QDA')
#
#                 name = type_DA + '_' + 'incre_supervised'
#                 QDA_incre_supervised, results.at[idx, 'time' + '_' + name], results.at[
#                     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_supervised(
#                     QDA_incre_supervised, classes, trainFeatures, 'QDA', rand[1])
#
#                 # name = type_DA + '_' + 'incre_Nigam_' + str(0.1)
#                 # QDA_incre_Nigam_01, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                 #     QDA_incre_Nigam_01, classes, trainFeatures, 'QDA', weight_Nigam=0.1)
#                 # name = type_DA + '_' + 'incre_Nigam_' + str(0.2)
#                 # QDA_incre_Nigam_02, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                 #     QDA_incre_Nigam_02, classes, trainFeatures, 'QDA', weight_Nigam=0.2)
#                 # name = type_DA + '_' + 'incre_Nigam_' + str(0.4)
#                 # QDA_incre_Nigam_04, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                 #     QDA_incre_Nigam_04, classes, trainFeatures, 'QDA', weight_Nigam=0.4)
#                 # name = type_DA + '_' + 'incre_Nigam_' + str(0.6)
#                 # QDA_incre_Nigam_06, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                 #     QDA_incre_Nigam_06, classes, trainFeatures, 'QDA', weight_Nigam=0.6)
#                 # name = type_DA + '_' + 'incre_Nigam_' + str(0.8)
#                 # QDA_incre_Nigam_08, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                 #     QDA_incre_Nigam_08, classes, trainFeatures, 'QDA', weight_Nigam=0.8)
#                 # name = type_DA + '_' + 'incre_Nigam_' + str(1.0)
#                 # QDA_incre_Nigam_10, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_weight_Nigam(
#                 #     QDA_incre_Nigam_10, classes, trainFeatures, 'QDA', weight_Nigam=1)
#                 #
#                 # name = type_DA + '_' + 'incre_threshold_' + str(0.2)
#                 # QDA_incre_threshold_02, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                 #     QDA_incre_threshold_02, classes, trainFeatures, 'QDA', threshold=0.2)
#                 # name = type_DA + '_' + 'incre_threshold_' + str(0.3)
#                 # QDA_incre_threshold_03, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                 #     QDA_incre_threshold_03, classes, trainFeatures, 'QDA', threshold=0.3)
#                 # name = type_DA + '_' + 'incre_threshold_' + str(0.4)
#                 # QDA_incre_threshold_04, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                 #     QDA_incre_threshold_04, classes, trainFeatures, 'QDA', threshold=0.4)
#                 # name = type_DA + '_' + 'incre_threshold_' + str(0.5)
#                 # QDA_incre_threshold_05, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                 #     QDA_incre_threshold_05, classes, trainFeatures, 'QDA', threshold=0.5)
#                 # name = type_DA + '_' + 'incre_threshold_' + str(0.6)
#                 # QDA_incre_threshold_06, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                 #     QDA_incre_threshold_06, classes, trainFeatures, 'QDA', threshold=0.6)
#                 # name = type_DA + '_' + 'incre_threshold_' + str(0.7)
#                 # QDA_incre_threshold_07, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                 #     QDA_incre_threshold_07, classes, trainFeatures, 'QDA', threshold=0.7)
#                 # name = type_DA + '_' + 'incre_threshold_' + str(0.8)
#                 # QDA_incre_threshold_08, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                 #     QDA_incre_threshold_08, classes, trainFeatures, 'QDA', threshold=0.8)
#                 # name = type_DA + '_' + 'incre_threshold_' + str(0.9)
#                 # QDA_incre_threshold_09, results.at[idx, 'time' + '_' + name], results.at[
#                 #     idx, 'w_predicted' + '_' + name] = SemiSupervised.model_incre_label_threshold(
#                 #     QDA_incre_threshold_09, classes, trainFeatures, 'QDA', threshold=0.9)
#
#                 results.at[idx, 'Feature Set'] = featureSet
#                 results.at[idx, 'person'] = person
#                 results.at[idx, 'exp_time'] = seed
#                 results.at[idx, 'shots in traning'] = shotStart
#                 results.at[idx, 'shot_class'] = rand
#                 results.at[idx, 'unlabeled Gesture'] = numberUnlabeledGestures
#
#                 idx += 1
#                 if np.any(evaluatedGestures == np.ones(len(evaluatedGestures)) * (numberUnlabeledGestures)):
#                     idx -= 1
#
#                     accumModel = currentDistributionValues(
#                         accumLabeledGesturesFeatures, accumLabeledGesturesLabels, classes, allFeatures, shotStart)
#
#                     type_DA = 'LDA'
#
#                     name = type_DA + '_' + 'accum'
#                     results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                         idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelLDA(
#                         testFeatures, testLabels, accumModel, classes)
#                     print('\n' + name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#
#                     # name = type_DA + '_' + 'incre_proposed_adapt'
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name]= DA_Classifiers.accuracyModelLDA(
#                     #     testFeatures, testLabels, LDA_incre_proposed_adapt, classes)
#                     # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#
#                     name = type_DA + '_' + 'incre_proposed'
#                     results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                         idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelLDA(
#                         testFeatures, testLabels, LDA_incre_proposed, classes)
#                     print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#
#                     name = type_DA + '_' + 'incre_labels'
#                     results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                         idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelLDA(
#                         testFeatures, testLabels, LDA_incre_label, classes)
#                     print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#
#                     # name = type_DA + '_' + 'incre_sequential'
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelLDA(
#                     #     testFeatures, testLabels, LDA_incre_sequential, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#
#                     name = type_DA + '_' + 'incre_supervised'
#                     results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                         idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelLDA(
#                         testFeatures, testLabels, LDA_incre_supervised, classes)
#                     print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(0.1)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelLDA(
#                     #     testFeatures, testLabels, LDA_incre_Nigam_01, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(0.2)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelLDA(
#                     #     testFeatures, testLabels, LDA_incre_Nigam_02, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(0.4)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelLDA(
#                     #     testFeatures, testLabels, LDA_incre_Nigam_04, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(0.6)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelLDA(
#                     #     testFeatures, testLabels, LDA_incre_Nigam_06, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(0.8)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelLDA(
#                     #     testFeatures, testLabels, LDA_incre_Nigam_08, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(1.0)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelLDA(
#                     #     testFeatures, testLabels, LDA_incre_Nigam_10, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.2)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelLDA(
#                     #     testFeatures, testLabels, LDA_incre_threshold_02, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.3)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelLDA(
#                     #     testFeatures, testLabels, LDA_incre_threshold_03, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.4)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelLDA(
#                     #     testFeatures, testLabels, LDA_incre_threshold_04, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.5)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelLDA(
#                     #     testFeatures, testLabels, LDA_incre_threshold_05, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.6)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelLDA(
#                     #     testFeatures, testLabels, LDA_incre_threshold_06, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.7)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelLDA(
#                     #     testFeatures, testLabels, LDA_incre_threshold_07, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.8)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelLDA(
#                     #     testFeatures, testLabels, LDA_incre_threshold_08, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.9)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelLDA(
#                     #     testFeatures, testLabels, LDA_incre_threshold_09, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#
#                     ###QDA
#                     type_DA = 'QDA'
#
#                     name = type_DA + '_' + 'accum'
#                     results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                         idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelQDA(
#                         testFeatures, testLabels, accumModel, classes)
#                     print('\n' + name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#
#                     # name = type_DA + '_' + 'incre_proposed_adapt'
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name]= DA_Classifiers.accuracyModelQDA(
#                     #     testFeatures, testLabels, QDA_incre_proposed_adapt, classes)
#                     # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#
#                     name = type_DA + '_' + 'incre_proposed'
#                     results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                         idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelQDA(
#                         testFeatures, testLabels, QDA_incre_proposed, classes)
#                     print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#
#                     name = type_DA + '_' + 'incre_labels'
#                     results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                         idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelQDA(
#                         testFeatures, testLabels, QDA_incre_label, classes)
#                     print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#
#                     # name = type_DA + '_' + 'incre_sequential'
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelQDA(
#                     #     testFeatures, testLabels, QDA_incre_sequential, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#
#                     name = type_DA + '_' + 'incre_supervised'
#                     results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                         idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelQDA(
#                         testFeatures, testLabels, QDA_incre_supervised, classes)
#                     print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(0.1)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelQDA(
#                     #     testFeatures, testLabels, QDA_incre_Nigam_01, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(0.2)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelQDA(
#                     #     testFeatures, testLabels, QDA_incre_Nigam_02, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(0.4)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelQDA(
#                     #     testFeatures, testLabels, QDA_incre_Nigam_04, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(0.6)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelQDA(
#                     #     testFeatures, testLabels, QDA_incre_Nigam_06, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(0.8)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelQDA(
#                     #     testFeatures, testLabels, QDA_incre_Nigam_08, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_Nigam_' + str(1.0)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelQDA(
#                     #     testFeatures, testLabels, QDA_incre_Nigam_10, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.2)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelQDA(
#                     #     testFeatures, testLabels, QDA_incre_threshold_02, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.3)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelQDA(
#                     #     testFeatures, testLabels, QDA_incre_threshold_03, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.4)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelQDA(
#                     #     testFeatures, testLabels, QDA_incre_threshold_04, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.5)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelQDA(
#                     #     testFeatures, testLabels, QDA_incre_threshold_05, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.6)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelQDA(
#                     #     testFeatures, testLabels, QDA_incre_threshold_06, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.7)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelQDA(
#                     #     testFeatures, testLabels, QDA_incre_threshold_07, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.8)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelQDA(
#                     #     testFeatures, testLabels, QDA_incre_threshold_08, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#                     #
#                     # name = type_DA + '_' + 'incre_threshold_' + str(0.9)
#                     # results.at[idx, name], results.at[idx, 'precision' + '_' + name], results.at[
#                     #     idx, 'recall' + '_' + name] = DA_Classifiers.accuracyModelQDA(
#                     #     testFeatures, testLabels, QDA_incre_threshold_09, classes)
#                     # # print(name + ' ' + str(numberUnlabeledGestures), results.at[idx, name])
#
#                     idx += 1
#                     results.to_csv(nameFile)
