import DA_BasedAdaptiveModels as adaptive
import DA_Classifiers as DA_Classifiers
import numpy as np
import pandas as pd
import math
from sklearn import preprocessing


# Upload Databases

def uploadDatabases(Database, featureSet=1):
    # Setting general variables
    path = '../DA-basedAdaptationTechnique/FewShotLearningEMG/'
    CH = 8

    if Database == 'EPN':
        carpet = 'ExtractedDataCollectedData'
        classes = 5
        peoplePriorK = 30
        peopleTest = 30
        combinationSet = list(range(1, 26))
        numberShots = 25
    elif Database == 'Nina5':
        carpet = 'ExtractedDataNinaDB5'
        classes = 18
        peoplePriorK = 10
        peopleTest = 10
        combinationSet = list(range(1, 5))
        numberShots = 4
    elif Database == 'Cote':
        carpet = 'ExtractedDataCoteAllard'
        classes = 7
        peoplePriorK = 19
        peopleTest = 17
        combinationSet = list(range(1, 5))
        numberShots = 4

    if featureSet == 1:
        # Setting variables
        Feature1 = 'logvarMatrix'
        segment = ''
        numberFeatures = 1
        allFeatures = numberFeatures * CH
        # Getting Data
        logvarMatrix = np.genfromtxt(path + 'ExtractedData/' + carpet + '/' + Feature1 + segment + '.csv',
                                     delimiter=',')
        if Database == 'EPN':
            dataMatrix = logvarMatrix[:, 0:]
            labelsDataMatrix = dataMatrix[:, allFeatures + 2]
        elif Database == 'Nina5':
            dataMatrix = logvarMatrix[:, 8:]
            labelsDataMatrix = dataMatrix[:, allFeatures + 1]
        elif Database == 'Cote':
            dataMatrix = logvarMatrix[:, 0:]
            dataMatrix[:, allFeatures + 1] = dataMatrix[:, allFeatures + 1] + 1
            dataMatrix[:, allFeatures + 3] = dataMatrix[:, allFeatures + 3] + 1
            labelsDataMatrix = dataMatrix[:, allFeatures + 3]



    elif featureSet == 2:
        # Setting variables
        Feature1 = 'mavMatrix'
        Feature2 = 'wlMatrix'
        Feature3 = 'zcMatrix'
        Feature4 = 'sscMatrix'
        segment = ''
        numberFeatures = 4
        allFeatures = numberFeatures * CH
        mavMatrix = np.genfromtxt(path + 'ExtractedData/' + carpet + '/' + Feature1 + segment + '.csv', delimiter=',')
        wlMatrix = np.genfromtxt(path + 'ExtractedData/' + carpet + '/' + Feature2 + segment + '.csv', delimiter=',')
        zcMatrix = np.genfromtxt(path + 'ExtractedData/' + carpet + '/' + Feature3 + segment + '.csv', delimiter=',')
        sscMatrix = np.genfromtxt(path + 'ExtractedData/' + carpet + '/' + Feature4 + segment + '.csv', delimiter=',')
        if Database == 'EPN':
            dataMatrix = np.hstack((mavMatrix[:, 0:CH], wlMatrix[:, 0:CH], zcMatrix[:, 0:CH], sscMatrix[:, 0:]))
            labelsDataMatrix = dataMatrix[:, allFeatures + 2]

        elif Database == 'Nina5':
            dataMatrix = np.hstack(
                (mavMatrix[:, 8:CH * 2], wlMatrix[:, 8:CH * 2], zcMatrix[:, 8:CH * 2], sscMatrix[:, 8:]))
            labelsDataMatrix = dataMatrix[:, allFeatures + 1]

        elif Database == 'Cote':
            dataMatrix = np.hstack((mavMatrix[:, 0:CH], wlMatrix[:, 0:CH], zcMatrix[:, 0:CH], sscMatrix[:, 0:]))
            dataMatrix[:, allFeatures + 1] = dataMatrix[:, allFeatures + 1] + 1
            dataMatrix[:, allFeatures + 3] = dataMatrix[:, allFeatures + 3] + 1
            labelsDataMatrix = dataMatrix[:, allFeatures + 3]


    elif featureSet == 3:
        # Setting variables
        Feature1 = 'lscaleMatrix'
        Feature2 = 'mflMatrix'
        Feature3 = 'msrMatrix'
        Feature4 = 'wampMatrix'
        segment = ''
        numberFeatures = 4
        allFeatures = numberFeatures * CH
        # Getting Data
        lscaleMatrix = np.genfromtxt(path + 'ExtractedData/' + carpet + '/' + Feature1 + segment + '.csv',
                                     delimiter=',')
        mflMatrix = np.genfromtxt(path + 'ExtractedData/' + carpet + '/' + Feature2 + segment + '.csv', delimiter=',')
        msrMatrix = np.genfromtxt(path + 'ExtractedData/' + carpet + '/' + Feature3 + segment + '.csv', delimiter=',')
        wampMatrix = np.genfromtxt(path + 'ExtractedData/' + carpet + '/' + Feature4 + segment + '.csv', delimiter=',')

        if Database == 'EPN':
            dataMatrix = np.hstack((lscaleMatrix[:, 0:CH], mflMatrix[:, 0:CH], msrMatrix[:, 0:CH], wampMatrix[:, 0:]))
            labelsDataMatrix = dataMatrix[:, allFeatures + 2]

        elif Database == 'Nina5':
            dataMatrix = np.hstack(
                (lscaleMatrix[:, 8:CH * 2], mflMatrix[:, 8:CH * 2], msrMatrix[:, 8:CH * 2], wampMatrix[:, 8:]))
            labelsDataMatrix = dataMatrix[:, allFeatures + 1]

        elif Database == 'Cote':
            dataMatrix = np.hstack((lscaleMatrix[:, 0:CH], mflMatrix[:, 0:CH], msrMatrix[:, 0:CH], wampMatrix[:, 0:]))
            dataMatrix[:, allFeatures + 1] = dataMatrix[:, allFeatures + 1] + 1
            dataMatrix[:, allFeatures + 3] = dataMatrix[:, allFeatures + 3] + 1
            labelsDataMatrix = dataMatrix[:, allFeatures + 3]

    return dataMatrix, numberFeatures, CH, classes, peoplePriorK, peopleTest, numberShots, combinationSet, allFeatures, labelsDataMatrix


# EVALUATION OVER THE THREE DATABASES


# def evaluation(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson, endPerson,
#                allFeatures,
#                typeDatabase, printR):
#     scaler = preprocessing.MinMaxScaler()
#
#     if typeDatabase == 'EPN':
#         evaluationEPN(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson, endPerson,
#                       allFeatures, printR, scaler)
#     elif typeDatabase == 'Nina5':
#         evaluationNina5(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson, endPerson,
#                         allFeatures, printR, scaler)
#     elif typeDatabase == 'Cote':
#         evaluationCote(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson, endPerson,
#                        allFeatures, printR, scaler)


def evaluation(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson, endPerson,
               allFeatures, typeDatabase, printR,shotStart):
    scaler = preprocessing.MinMaxScaler()
    results = pd.DataFrame(
        columns=['person', 'subset', '# shots', 'Feature Set', 'wTargetMeanQDA', 'wTargetCovQDA'])
    idx = 0
    typeData = 0
    # shotStart = 3
    shotStart+=1
    initialShots = list(range(1, shotStart))

    fromPeople = peoplePriorK + startPerson
    toPeople = peoplePriorK + endPerson + 1
    if typeDatabase == 'Nina5':
        fromPeople = startPerson
        toPeople = endPerson + 1
    elif typeDatabase == 'Cote':
        trainFeaturesGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData)), 0:allFeatures][0]
        trainLabelsGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData)), allFeatures + 3][0]
    elif typeDatabase == 'EPN':
        trainFeaturesGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures + 1] <= peoplePriorK)), 0:allFeatures][0]
        trainLabelsGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures + 1] <= peoplePriorK)), allFeatures + 2][0]

    for person in range(fromPeople, toPeople):

        if typeDatabase == 'Nina5':

            trainFeaturesGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] != person)), 0:allFeatures][0]
            trainLabelsGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] != person)), allFeatures + 1][0]

            testFeatures = \
                dataMatrix[np.where((dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 2] >= 5)),
                0:allFeatures][0]
            testLabels = dataMatrix[
                np.where(
                    (dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 2] >= 5)), allFeatures + 1][
                0].T

            oneShotFeatures = np.empty((0, allFeatures))
            oneShotLabels = []
            for shot in initialShots:
                oneShotFeatures = np.vstack((oneShotFeatures, dataMatrix[np.where(
                    (dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 2] == shot)),
                                                              0:allFeatures][0]))
                oneShotLabels = np.hstack((oneShotLabels, dataMatrix[np.where(
                    (dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 2] == shot))
                , allFeatures + 1][0].T))

            evaluatedPerson = person

        elif typeDatabase == 'Cote':
            typeData = 1
            carpet = 2
            testFeatures = \
                dataMatrix[
                np.where((dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person)
                         * (dataMatrix[:, allFeatures + 2] == carpet)), 0:allFeatures][0]
            testLabels = \
                dataMatrix[
                    np.where((dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person)
                             * (dataMatrix[:, allFeatures + 2] == carpet)), allFeatures + 3][0]
            carpet = 1

            oneShotFeatures = np.empty((0, allFeatures))
            oneShotLabels = []
            for shot in initialShots:
                oneShotFeatures = np.vstack((oneShotFeatures, dataMatrix[np.where(
                    (dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person) * (
                            dataMatrix[:, allFeatures + 2] == carpet)
                    * (dataMatrix[:, allFeatures + 4] == shot)), 0:allFeatures][0]))
                oneShotLabels = np.hstack((oneShotLabels, dataMatrix[np.where(
                    (dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person)
                    * (dataMatrix[:, allFeatures + 2] == carpet) * (
                            dataMatrix[:, allFeatures + 4] == shot)), allFeatures + 3][0].T))

            evaluatedPerson = None

        elif typeDatabase == 'EPN':
            typeData = 1
            testFeatures = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                    dataMatrix[:, allFeatures + 1] == person)), 0:allFeatures][0]
            testLabels = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                    dataMatrix[:, allFeatures + 1] == person)), allFeatures + 2][0]
            typeData = 0

            oneShotFeatures = np.empty((0, allFeatures))
            oneShotLabels = []
            for shot in initialShots:
                oneShotFeatures = np.vstack(
                    (oneShotFeatures, dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                            dataMatrix[:, allFeatures + 1] == person) * (dataMatrix[:, allFeatures + 3] == shot)),
                                      0:allFeatures][0]))
                oneShotLabels = np.hstack((oneShotLabels, dataMatrix[
                    np.where((dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person) * (
                            dataMatrix[:, allFeatures + 3] == shot)), allFeatures + 2][0].T))

            evaluatedPerson = None

        oneShotFeatures = scaler.fit_transform(oneShotFeatures)
        oneShotModel = currentDistributionValues(oneShotFeatures, oneShotLabels, classes, allFeatures)
        oneShotModelLDA = oneShotModel.copy()
        oneShotModelQDA = oneShotModel.copy()

        dataPK, allFeaturesPK = preprocessingPK(dataMatrix, allFeatures, scaler)

        preTrainedDataMatrix = PKModels(dataPK, classes, peoplePriorK, evaluatedPerson, allFeatures, typeDatabase)

        # propOneShotModelModel, _, _, _, _, _ = adaptive.OurModel(
        #     oneShotModel, preTrainedDataMatrix, classes, allFeatures, oneShotFeatures, oneShotLabels, step=1,
        #     typeModel='QDA', k=1)

        for shot in range(shotStart, numberShots + 1):

            subset = list(range(shotStart, shot + 1))

            trainFeatures = np.empty((0, allFeatures))
            trainLabels = []
            trainRep = []
            for auxIndex in subset:
                if typeDatabase == 'Nina5':
                    trainFeatures = np.vstack((trainFeatures, dataMatrix[np.where(
                        (dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 2] == auxIndex)),
                                                              0:allFeatures][0]))
                    trainLabels = np.hstack((trainLabels, dataMatrix[np.where(
                        (dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 2] == auxIndex))
                    , allFeatures + 1][0].T))
                elif typeDatabase == 'Cote':
                    trainFeatures = np.vstack((trainFeatures, dataMatrix[np.where(
                        (dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person) * (
                                dataMatrix[:, allFeatures + 2] == carpet)
                        * (dataMatrix[:, allFeatures + 4] == auxIndex)), 0:allFeatures][0]))
                    trainLabels = np.hstack((trainLabels, dataMatrix[np.where(
                        (dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person)
                        * (dataMatrix[:, allFeatures + 2] == carpet) * (
                                dataMatrix[:, allFeatures + 4] == auxIndex)), allFeatures + 3][0].T))
                elif typeDatabase == 'EPN':
                    trainFeatures = np.vstack(
                        (trainFeatures, dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                                dataMatrix[:, allFeatures + 1] == person) * (
                                                                    dataMatrix[:, allFeatures + 3] == auxIndex)),
                                        0:allFeatures][0]))
                    trainLabels = np.hstack(
                        (trainLabels, dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                                dataMatrix[:, allFeatures + 1] == person) * (dataMatrix[:,
                                                                             allFeatures + 3] == auxIndex)), allFeatures + 2][
                            0].T))

                    trainRep = np.hstack(
                        (trainRep, dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                                dataMatrix[:, allFeatures + 1] == person) * (dataMatrix[:,
                                                                             allFeatures + 3] == auxIndex)), allFeatures + 3][
                            0].T))

            trainFeaturesGen = np.vstack((trainFeaturesGenPre, trainFeatures))
            trainLabelsGen = np.hstack((trainLabelsGenPre, trainLabels))

            k = 1 - (np.log(shot) / np.log(numberShots + 1))

            trainFeatures = scaler.transform(trainFeatures)
            trainFeaturesGen = scaler.transform(trainFeaturesGen)
            testFeaturesTransform = scaler.transform(testFeatures)

            currentValues = currentDistributionValues(trainFeatures, trainLabels, classes, allFeaturesPK)
            pkValues = currentDistributionValues(trainFeaturesGen, trainLabelsGen, classes, allFeaturesPK)
            # results, idx = resultsDataframe(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels,
            #                                 classes, allFeaturesPK, results, testFeaturesTransform, testLabels, idx,
            #                                 person, subset, featureSet, nameFile, printR, k, pkValues)

            results, idx, oneShotModelLDA, oneShotModelQDA = resultsDataframeUnsupervised(
                currentValues, preTrainedDataMatrix, trainFeatures, trainLabels, classes, allFeaturesPK, results,
                testFeaturesTransform, testLabels, idx, person, subset, featureSet, nameFile, printR, k, pkValues,
                oneShotFeatures, oneShotLabels, trainRep, oneShotModelLDA, oneShotModelQDA, oneShotModel, typeDatabase)

    return results


def PKModels(dataMatrix, classes, peoplePriorK, evaluatedPerson, allFeatures, typeDatabase):
    preTrainedDataMatrix = pd.DataFrame(columns=['mean', 'cov', 'class', 'person'])
    indx = 0
    typeData = 0

    people = list(range(1, peoplePriorK + 1))
    if evaluatedPerson != None:
        del people[evaluatedPerson - 1]
    for cl in range(1, classes + 1):

        for person in people:

            if typeDatabase == 'Nina':
                preTrainedDataMatrix.at[indx, 'cov'] = np.cov(
                    dataMatrix[
                    np.where((dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 1] == cl)),
                    0:allFeatures][0], rowvar=False)
                preTrainedDataMatrix.at[indx, 'mean'] = np.mean(
                    dataMatrix[
                    np.where((dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 1] == cl)),
                    0:allFeatures][0], axis=0)

            elif typeDatabase == 'Cote':
                preTrainedDataMatrix.at[indx, 'cov'] = np.cov(
                    dataMatrix[
                    np.where((dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person) * (
                            dataMatrix[:, allFeatures + 3] == cl)), 0:allFeatures][0], rowvar=False)
                preTrainedDataMatrix.at[indx, 'mean'] = np.mean(
                    dataMatrix[
                    np.where((dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person) * (
                            dataMatrix[:, allFeatures + 3] == cl)), 0:allFeatures][0], axis=0)

            elif typeDatabase == 'EPN':
                preTrainedDataMatrix.at[indx, 'cov'] = np.cov(
                    dataMatrix[
                    np.where((dataMatrix[:, allFeatures + 1] == person) * (dataMatrix[:, allFeatures + 2] == cl)),
                    0:allFeatures][0], rowvar=False)
                preTrainedDataMatrix.at[indx, 'mean'] = np.mean(
                    dataMatrix[
                    np.where((dataMatrix[:, allFeatures + 1] == person) * (dataMatrix[:, allFeatures + 2] == cl)),
                    0:allFeatures][0], axis=0)
            preTrainedDataMatrix.at[indx, 'class'] = cl
            preTrainedDataMatrix.at[indx, 'person'] = person
            indx += 1

    return preTrainedDataMatrix


# Unsupervised

def resultsDataframeUnsupervised(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels, classes, allFeatures,
                                 results, testFeatures, testLabels, idx, person, subset, featureSet, nameFile, printR,
                                 k, pkValues, oneShotFeatures, oneShotLabels, trainRep, ldaUnsupervisedModel,
                                 qdaUnsupervisedModel, oneShotModel, typeDatabase):
    # Amount of Training data
    # minSamplesClass = 20
    # step = math.ceil(np.shape(trainLabels)[0] / (classes * minSamplesClass))
    # print('samples: ', np.shape(trainLabels)[0], 'step: ', step)

    '''
    liuModel = adaptive.LiuModel(currentValues, preTrainedDataMatrix, classes, allFeatures)
    vidovicModelL, vidovicModelQ = adaptive.VidovicModel(currentValues, preTrainedDataMatrix, classes, allFeatures)

    propModel, _, results.at[idx, 'wTargetMeanQDA'], _, results.at[idx, 'wTargetCovQDA'], results.at[
        idx, 'tPropQDA'] = adaptive.OurModel(
        currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, step, 'QDA', k)
    '''

    step = 1
    numSamples = 10
    oneShotFeatures, oneShotLabels = adaptive.subsetTraining(oneShotFeatures, oneShotLabels, numSamples, classes)

    lenOneShot = len(oneShotLabels)
    oneModelLDA = ldaUnsupervisedModel.copy()
    oneModelQDA = qdaUnsupervisedModel.copy()

    unsuperMatrixLDA = DA_Classifiers.scoreModelLDA_ClassificationUnsupervised(trainFeatures, ldaUnsupervisedModel,
                                                                               classes, trainLabels)

    unsuperMatrixQDA = DA_Classifiers.scoreModelQDA_ClassificationUnsupervised(trainFeatures, qdaUnsupervisedModel,
                                                                               classes, trainLabels)

    ldaUnsupervisedModel, _, _, _, _, _ = adaptive.OurModelUnsupervisedAllProb(
        ldaUnsupervisedModel, unsuperMatrixLDA, classes, allFeatures, oneShotFeatures, oneShotLabels, oneShotModel,
        step, 'QDA', 1, typeDatabase)

    qdaUnsupervisedModel, _, _, _, _, _ = adaptive.OurModelUnsupervisedAllProb(
        qdaUnsupervisedModel, unsuperMatrixQDA, classes, allFeatures, oneShotFeatures, oneShotLabels, oneShotModel,
        step, 'QDA', 1, typeDatabase)

    # propModelQDA, wMean, _, wCov, _, _ = adaptive.OurModelNok(
    #     currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, step, 'QDA', k)
    # w = (wMean + wCov) / 2
    # # w=np.ones(classes)
    # print('weight current:', w)
    # propModelLDA = propModelQDA.copy()
    # propModelLDA, _, results.at[idx, 'wTargetMeanLDA'], _, results.at[idx, 'wTargetCovLDA'], results.at[
    #     idx, 'tPropLDA'] = adaptive.OurModelFewClass(currentValues, preTrainedDataMatrix, classes, allFeatures,
    #                                                  trainFeatures, trainLabels, step, 'LDA', k)
    # propModelQDA, _, results.at[idx, 'wTargetMeanQDA'], _, results.at[idx, 'wTargetCovQDA'], results.at[
    #     idx, 'tPropQDA'] = adaptive.OurModelFewClass(currentValues, preTrainedDataMatrix, classes, allFeatures,
    #                                                  trainFeatures, trainLabels, step, 'QDA', k)

    # propModelLDA, _, _, _, _, results.at[idx, 'tPropLDA'] = adaptive.OurModelModified(
    #     currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, 'LDA', 30)
    # propModelQDA, _, _, _, _, results.at[idx, 'tPropQDA'] = adaptive.OurModelModified(
    #     currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, 'QDA', 30)

    # modelPKLDA = adaptive.weightsMCC(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures,
    #                                  trainLabels, 'LDA', 30)
    # modelPKQDA = adaptive.weightsMCC(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures,
    #                                  trainLabels, 'QDA', 30)
    #
    # propModelLDA = adaptive.joinModels(currentValues, classes, trainFeatures, trainLabels, 'LDA', modelPKLDA)
    # propModelQDA = adaptive.joinModels(currentValues, classes, trainFeatures, trainLabels, 'QDA', modelPKQDA)

    # GeneticModelLDA, results.at[idx, 'timeTrainingPropLDA'], wL1, wL2 = adaptive.GeneticModel(
    #     currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, 'LDA', propModelLDA)
    # GeneticModelQDA, results.at[idx, 'timeTrainingPropQDA'], wQ1, wQ2 = adaptive.GeneticModel(
    #     currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, 'QDA', propModelQDA)

    results.at[idx, 'person'] = person
    results.at[idx, 'subset'] = subset
    results.at[idx, '# shots'] = np.size(subset)
    results.at[idx, 'Feature Set'] = featureSet
    # LDA results
    print('indLDA')
    results.at[idx, 'AccLDAInd'], _, results.at[idx, 'tIndL'], _ = DA_Classifiers.accuracyModelLDAconfusionMatrix(
        testFeatures, testLabels, oneModelLDA, classes)

    print('propLDA')

    # results.at[idx, 'AccLDAProp'], _, results.at[idx, 'tPropL'], _ = DA_Classifiers.accuracyModelLDAMix(
    #     testFeatures, testLabels, currentValues, propModelLDA, classes, w, 1 - w)
    results.at[idx, 'AccLDAProp'], _, results.at[idx, 'tPropL'], _ = DA_Classifiers.accuracyModelLDAconfusionMatrix(
        testFeatures, testLabels, ldaUnsupervisedModel, classes)

    print('indQDA')
    results.at[idx, 'AccQDAInd'], _, results.at[idx, 'tIndQ'], _ = DA_Classifiers.accuracyModelQDAconfusionMatrix(
        testFeatures, testLabels, oneModelQDA, classes)

    print('propQDA')
    results.at[idx, 'AccQDAProp'], _, results.at[idx, 'tPropQ'], _ = DA_Classifiers.accuracyModelQDAconfusionMatrix(
        testFeatures, testLabels, qdaUnsupervisedModel, classes)

    # results.at[idx, 'AccQDAProp'], _, results.at[idx, 'tPropQ'], _ = DA_Classifiers.accuracyModelQDAconfusionMatrix(
    #     testFeatures, testLabels, propModelQDA, classes)

    # print('modelLDA')
    # results.at[idx, 'AccLDAGenetic'], _, _, _ = DA_Classifiers.accuracyModelLDAconfusionMatrix(
    #     testFeatures, testLabels, GeneticModelLDA, classes)
    #
    # results.at[idx, 'AccLDAMulti'], results.at[idx, 'tGenL'] = DA_Classifiers.accuracyModelLDA(testFeatures, testLabels,
    #                                                                                            pkValues,
    #                                                                                            classes)
    # print('mixLDA')
    # results.at[idx, 'AccLDAMix'], results.at[idx, 'tMixL'] = DA_Classifiers.accuracyModelLDAMix(
    #     testFeatures, testLabels, currentValues, GeneticModelLDA, classes, wL1, wL2)
    #
    # print('mixQDA')
    # results.at[idx, 'AccQDAMix'], results.at[idx, 'tMixQ'] = DA_Classifiers.accuracyModelQDAMix(
    #     testFeatures, testLabels, currentValues, GeneticModelQDA, classes, wQ1, wQ2)
    '''
    # results.at[idx, 'AccLDAProp'], results.at[idx, 'tCLPropL'] = DA_Classifiers.accuracyModelLDA(testFeatures,
    #                                                                                              testLabels, propModel,
    #                                                                                              classes)
    # results.at[idx, 'AccLDALiu'], _ = DA_Classifiers.accuracyModelLDA(testFeatures, testLabels, liuModel, classes)
    # results.at[idx, 'AccLDAVidovic'], _ = DA_Classifiers.accuracyModelLDA(testFeatures, testLabels, vidovicModelL,
    #                                                                       classes)

    ## QDA results


    results.at[idx, 'AccQDAInd'], results.at[idx, 'tIndQ'] = DA_Classifiers.accuracyModelQDA(testFeatures, testLabels,
                                                                                             currentValues,
                                                                                             classes)


    results.at[idx, 'AccQDAMulti'], results.at[idx, 'tGenQ'] = DA_Classifiers.accuracyModelQDA(testFeatures, testLabels,
                                                                                               pkValues,
                                                                                               classes)

    results.at[idx, 'AccQDAProp'], results.at[idx, 'tCLPropQ'] = DA_Classifiers.accuracyModelQDA(testFeatures,
                                                                                                 testLabels, propModel,
                                                                                                 classes)
    results.at[idx, 'AccQDALiu'], _ = DA_Classifiers.accuracyModelQDA(testFeatures, testLabels, liuModel, classes)
    results.at[idx, 'AccQDAVidovic'], _ = DA_Classifiers.accuracyModelQDA(testFeatures, testLabels, vidovicModelQ,
                                                                          classes)

    '''

    if nameFile is not None:
        results.to_csv(nameFile)
    if printR:
        print(featureSet)
        print('Results: person= ', person, ' shot set= ', subset)
        print(results.loc[idx])

    idx += 1

    return results, idx, ldaUnsupervisedModel, qdaUnsupervisedModel


# Auxiliar functions of the evaluation
def resultsDataframe(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels, classes, allFeatures,
                     results, testFeatures, testLabels, idx, person, subset, featureSet, nameFile, printR, k, pkValues):
    # Amount of Training data
    minSamplesClass = 20
    step = math.ceil(np.shape(trainLabels)[0] / (classes * minSamplesClass))
    print('samples: ', np.shape(trainLabels)[0], 'step: ', step)

    '''
    liuModel = adaptive.LiuModel(currentValues, preTrainedDataMatrix, classes, allFeatures)
    vidovicModelL, vidovicModelQ = adaptive.VidovicModel(currentValues, preTrainedDataMatrix, classes, allFeatures)
    
    propModel, _, results.at[idx, 'wTargetMeanQDA'], _, results.at[idx, 'wTargetCovQDA'], results.at[
        idx, 'tPropQDA'] = adaptive.OurModel(
        currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, step, 'QDA', k)
    '''

    propModelQDA, wMean, _, wCov, _, _ = adaptive.OurModelNok(
        currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, step, 'QDA', k)
    w = (wMean + wCov) / 2
    # w=np.ones(classes)
    print('weight current:', w)
    propModelLDA = propModelQDA.copy()
    # propModelLDA, _, results.at[idx, 'wTargetMeanLDA'], _, results.at[idx, 'wTargetCovLDA'], results.at[
    #     idx, 'tPropLDA'] = adaptive.OurModelFewClass(currentValues, preTrainedDataMatrix, classes, allFeatures,
    #                                                  trainFeatures, trainLabels, step, 'LDA', k)
    # propModelQDA, _, results.at[idx, 'wTargetMeanQDA'], _, results.at[idx, 'wTargetCovQDA'], results.at[
    #     idx, 'tPropQDA'] = adaptive.OurModelFewClass(currentValues, preTrainedDataMatrix, classes, allFeatures,
    #                                                  trainFeatures, trainLabels, step, 'QDA', k)

    # propModelLDA, _, _, _, _, results.at[idx, 'tPropLDA'] = adaptive.OurModelModified(
    #     currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, 'LDA', 30)
    # propModelQDA, _, _, _, _, results.at[idx, 'tPropQDA'] = adaptive.OurModelModified(
    #     currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, 'QDA', 30)

    # modelPKLDA = adaptive.weightsMCC(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures,
    #                                  trainLabels, 'LDA', 30)
    # modelPKQDA = adaptive.weightsMCC(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures,
    #                                  trainLabels, 'QDA', 30)
    #
    # propModelLDA = adaptive.joinModels(currentValues, classes, trainFeatures, trainLabels, 'LDA', modelPKLDA)
    # propModelQDA = adaptive.joinModels(currentValues, classes, trainFeatures, trainLabels, 'QDA', modelPKQDA)

    # GeneticModelLDA, results.at[idx, 'timeTrainingPropLDA'], wL1, wL2 = adaptive.GeneticModel(
    #     currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, 'LDA', propModelLDA)
    # GeneticModelQDA, results.at[idx, 'timeTrainingPropQDA'], wQ1, wQ2 = adaptive.GeneticModel(
    #     currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, 'QDA', propModelQDA)

    results.at[idx, 'person'] = person
    results.at[idx, 'subset'] = subset
    results.at[idx, '# shots'] = np.size(subset)
    results.at[idx, 'Feature Set'] = featureSet
    # LDA results
    print('indLDA')
    results.at[idx, 'AccLDAInd'], _, results.at[idx, 'tIndL'], _ = DA_Classifiers.accuracyModelLDAconfusionMatrix(
        testFeatures, testLabels, currentValues, classes)

    print('propLDA')

    results.at[idx, 'AccLDAProp'], _, results.at[idx, 'tPropL'], _ = DA_Classifiers.accuracyModelLDAMix(
        testFeatures, testLabels, currentValues, propModelLDA, classes, w, 1 - w)
    # results.at[idx, 'AccLDAProp'], _, results.at[idx, 'tPropL'], _ = DA_Classifiers.accuracyModelLDAconfusionMatrix(
    #     testFeatures, testLabels, propModelLDA, classes)

    print('indQDA')
    results.at[idx, 'AccQDAInd'], _, results.at[idx, 'tIndQ'], _ = DA_Classifiers.accuracyModelQDAconfusionMatrix(
        testFeatures, testLabels, currentValues, classes)

    print('propQDA')
    results.at[idx, 'AccQDAProp'], _, results.at[idx, 'tPropQ'], _ = DA_Classifiers.accuracyModelQDAMix(
        testFeatures, testLabels, currentValues, propModelQDA, classes, w, 1 - w)

    # results.at[idx, 'AccQDAProp'], _, results.at[idx, 'tPropQ'], _ = DA_Classifiers.accuracyModelQDAconfusionMatrix(
    #     testFeatures, testLabels, propModelQDA, classes)

    # print('modelLDA')
    # results.at[idx, 'AccLDAGenetic'], _, _, _ = DA_Classifiers.accuracyModelLDAconfusionMatrix(
    #     testFeatures, testLabels, GeneticModelLDA, classes)
    #
    # results.at[idx, 'AccLDAMulti'], results.at[idx, 'tGenL'] = DA_Classifiers.accuracyModelLDA(testFeatures, testLabels,
    #                                                                                            pkValues,
    #                                                                                            classes)
    # print('mixLDA')
    # results.at[idx, 'AccLDAMix'], results.at[idx, 'tMixL'] = DA_Classifiers.accuracyModelLDAMix(
    #     testFeatures, testLabels, currentValues, GeneticModelLDA, classes, wL1, wL2)
    #
    # print('mixQDA')
    # results.at[idx, 'AccQDAMix'], results.at[idx, 'tMixQ'] = DA_Classifiers.accuracyModelQDAMix(
    #     testFeatures, testLabels, currentValues, GeneticModelQDA, classes, wQ1, wQ2)
    '''
    # results.at[idx, 'AccLDAProp'], results.at[idx, 'tCLPropL'] = DA_Classifiers.accuracyModelLDA(testFeatures,
    #                                                                                              testLabels, propModel,
    #                                                                                              classes)
    # results.at[idx, 'AccLDALiu'], _ = DA_Classifiers.accuracyModelLDA(testFeatures, testLabels, liuModel, classes)
    # results.at[idx, 'AccLDAVidovic'], _ = DA_Classifiers.accuracyModelLDA(testFeatures, testLabels, vidovicModelL,
    #                                                                       classes)

    ## QDA results
   

    results.at[idx, 'AccQDAInd'], results.at[idx, 'tIndQ'] = DA_Classifiers.accuracyModelQDA(testFeatures, testLabels,
                                                                                             currentValues,
                                                                                             classes)


    results.at[idx, 'AccQDAMulti'], results.at[idx, 'tGenQ'] = DA_Classifiers.accuracyModelQDA(testFeatures, testLabels,
                                                                                               pkValues,
                                                                                               classes)
    
    results.at[idx, 'AccQDAProp'], results.at[idx, 'tCLPropQ'] = DA_Classifiers.accuracyModelQDA(testFeatures,
                                                                                                 testLabels, propModel,
                                                                                                 classes)
    results.at[idx, 'AccQDALiu'], _ = DA_Classifiers.accuracyModelQDA(testFeatures, testLabels, liuModel, classes)
    results.at[idx, 'AccQDAVidovic'], _ = DA_Classifiers.accuracyModelQDA(testFeatures, testLabels, vidovicModelQ,
                                                                          classes)
                                                                          
    '''

    if nameFile is not None:
        results.to_csv(nameFile)
    if printR:
        print(featureSet)
        print('Results: person= ', person, ' shot set= ', subset)
        print(results.loc[idx])

    idx += 1

    return results, idx


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
    dataMatrixFeatures = scaler.transform(dataMatrix[:, 0:allFeatures])
    return np.hstack((dataMatrixFeatures, dataMatrix[:, allFeatures:])), np.size(dataMatrixFeatures, axis=1)


### EPN
def evaluationEPN(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson, endPerson,
                  allFeatures, printR, scaler):
    results = pd.DataFrame(
        columns=['person', 'subset', '# shots', 'Feature Set', 'wTargetMeanQDA', 'wTargetCovQDA'])

    trainFeaturesGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures + 1] <= peoplePriorK)), 0:allFeatures][0]
    trainLabelsGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures + 1] <= peoplePriorK)), allFeatures + 2][0]

    idx = 0
    for person in range(peoplePriorK + startPerson, peoplePriorK + endPerson + 1):

        typeData = 1
        testFeatures = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                dataMatrix[:, allFeatures + 1] == person)), 0:allFeatures][0]
        testLabels = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                dataMatrix[:, allFeatures + 1] == person)), allFeatures + 2][0]

        typeData = 0

        shot = 1
        oneShotFeatures = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                dataMatrix[:, allFeatures + 1] == person) * (dataMatrix[:, allFeatures + 3] == shot)), 0:allFeatures][0]
        oneShotLabels = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                dataMatrix[:, allFeatures + 1] == person) * (dataMatrix[:, allFeatures + 3] == shot)), allFeatures + 2][
            0].T

        oneShotFeatures = scaler.fit_transform(oneShotFeatures)
        oneShotModel = currentDistributionValues(oneShotFeatures, oneShotLabels, classes, allFeatures)
        oneShotModelLDA = oneShotModel.copy()
        oneShotModelQDA = oneShotModel.copy()

        dataPK, allFeaturesPK = preprocessingPK(dataMatrix, allFeatures, scaler)

        preTrainedDataMatrix = PKModels(dataPK, classes, peoplePriorK, None, allFeatures, 'EPN')

        propOneShotModelModel, _, _, _, _, _ = adaptive.OurModel(
            oneShotModel, preTrainedDataMatrix, classes, allFeatures, oneShotFeatures, oneShotLabels, step=1,
            typeModel='QDA', k=1)

        shotStart = 2
        for shot in range(shotStart, numberShots + 1):

            subset = tuple(range(shotStart, shot + 1))

            trainFeatures = np.empty((0, allFeatures))
            trainLabels = []
            trainRep = []
            for auxIndex in range(np.size(subset)):
                trainFeatures = np.vstack(
                    (trainFeatures, dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                            dataMatrix[:, allFeatures + 1] == person) * (dataMatrix[:, allFeatures + 3] == subset[
                        auxIndex])), 0:allFeatures][0]))
                trainLabels = np.hstack(
                    (trainLabels, dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                            dataMatrix[:, allFeatures + 1] == person) * (dataMatrix[:, allFeatures + 3] == subset[
                        auxIndex])), allFeatures + 2][0].T))

                trainRep = np.hstack(
                    (trainRep, dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                            dataMatrix[:, allFeatures + 1] == person) * (dataMatrix[:, allFeatures + 3] == subset[
                        auxIndex])), allFeatures + 3][0].T))

            trainFeaturesGen = np.vstack((trainFeaturesGenPre, trainFeatures))
            trainLabelsGen = np.hstack((trainLabelsGenPre, trainLabels))

            k = 1 - (np.log(shot) / np.log(numberShots + 1))

            trainFeatures = scaler.transform(trainFeatures)
            trainFeaturesGen = scaler.transform(trainFeaturesGen)
            testFeaturesTransform = scaler.transform(testFeatures)

            currentValues = currentDistributionValues(trainFeatures, trainLabels, classes, allFeaturesPK)
            pkValues = currentDistributionValues(trainFeaturesGen, trainLabelsGen, classes, allFeaturesPK)
            # results, idx = resultsDataframe(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels,
            #                                 classes, allFeaturesPK, results, testFeaturesTransform, testLabels, idx,
            #                                 person, subset, featureSet, nameFile, printR, k, pkValues)

            results, idx, oneShotModelLDA, oneShotModelQDA = resultsDataframeUnsupervised(
                currentValues, preTrainedDataMatrix, trainFeatures, trainLabels, classes, allFeaturesPK, results,
                testFeaturesTransform, testLabels, idx, person, subset, featureSet, nameFile, printR, k, pkValues,
                oneShotFeatures, oneShotLabels, trainRep, oneShotModelLDA, oneShotModelQDA, oneShotModel)

    return results


### Cote-Allard

def evaluationCote(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson, endPerson,
                   allFeatures, printR, scaler):
    # Creating Variables
    results = pd.DataFrame(
        columns=['person', 'subset', '# shots', 'Feature Set', 'wTargetMeanQDA', 'wTargetCovQDA'])

    idx = 0

    typeData = 0
    trainFeaturesGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData)), 0:allFeatures][0]
    trainLabelsGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData)), allFeatures + 3][0]

    typeData = 1
    for person in range(peoplePriorK + startPerson, peoplePriorK + endPerson + 1):

        carpet = 2
        testFeatures = \
            dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person)
                                * (dataMatrix[:, allFeatures + 2] == carpet)), 0:allFeatures][0]
        testLabels = \
            dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person)
                                * (dataMatrix[:, allFeatures + 2] == carpet)), allFeatures + 3][0]

        carpet = 1
        # 4 cycles - cross_validation for 4 cycles or shots
        for shot in range(1, numberShots + 1):

            subset = tuple(range(1, shot + 1))

            trainFeatures = np.empty((0, allFeatures))
            trainLabels = []

            for auxIndex in range(np.size(subset)):
                trainFeatures = np.vstack((trainFeatures, dataMatrix[
                                                          np.where((dataMatrix[:, allFeatures] == typeData)
                                                                   * (dataMatrix[:, allFeatures + 1] == person)
                                                                   * (dataMatrix[:, allFeatures + 2] == carpet)
                                                                   * (dataMatrix[:, allFeatures + 4] == subset[
                                                              auxIndex]))
                , 0:allFeatures][0]))
                trainLabels = np.hstack((trainLabels, dataMatrix[
                    np.where((dataMatrix[:, allFeatures] == typeData)
                             * (dataMatrix[:, allFeatures + 1] == person)
                             * (dataMatrix[:, allFeatures + 2] == carpet)
                             * (dataMatrix[:, allFeatures + 4] == subset[auxIndex]))
                    , allFeatures + 3][0].T))
            trainFeaturesGen = np.vstack((trainFeaturesGenPre, trainFeatures))
            trainLabelsGen = np.hstack((trainLabelsGenPre, trainLabels))

            k = 1 - (np.log(shot) / np.log(numberShots + 1))

            scaler.fit(trainFeatures)

            trainFeatures = scaler.transform(trainFeatures)

            trainFeaturesGen = scaler.transform(trainFeaturesGen)
            testFeaturesTransform = scaler.transform(testFeatures)

            dataPK, allFeaturesPK = preprocessingPK(dataMatrix, allFeatures, scaler)

            preTrainedDataMatrix = preTrainedDataCote(dataPK, classes, peoplePriorK, allFeaturesPK)
            currentValues = currentDistributionValues(trainFeatures, trainLabels, classes, allFeaturesPK)
            pkValues = currentDistributionValues(trainFeaturesGen, trainLabelsGen, classes, allFeaturesPK)
            results, idx = resultsDataframe(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels,
                                            classes, allFeaturesPK, results, testFeaturesTransform, testLabels, idx,
                                            person, subset, featureSet, nameFile, printR, k, pkValues)

            # results, idx = adaptive.DAugResultsDataframe(
            #     currentValues, preTrainedDataMatrix, trainFeatures, trainLabels, classes, allFeaturesPK, results,
            #     testFeaturesTransform, testLabels, idx, person, subset, featureSet, nameFile, printR, trainFeaturesGen,
            #     trainLabelsGen)

    return results


def preTrainedDataEPN(dataMatrix, classes, peoplePriorK, allFeatures):
    preTrainedDataMatrix = pd.DataFrame(columns=['mean', 'cov', 'class', 'person'])

    indx = 0
    for cl in range(1, classes + 1):

        for person in range(1, peoplePriorK + 1):
            preTrainedDataMatrix.at[indx, 'cov'] = np.cov(
                dataMatrix[np.where((dataMatrix[:, allFeatures + 1] == person) * (
                        dataMatrix[:, allFeatures + 2] == cl)), 0:allFeatures][0], rowvar=False)
            preTrainedDataMatrix.at[indx, 'mean'] = np.mean(
                dataMatrix[np.where((dataMatrix[:, allFeatures + 1] == person) * (
                        dataMatrix[:, allFeatures + 2] == cl)), 0:allFeatures][0], axis=0)
            preTrainedDataMatrix.at[indx, 'class'] = cl
            preTrainedDataMatrix.at[indx, 'person'] = person
            indx += 1

    return preTrainedDataMatrix


def preTrainedDataCote(dataMatrix, classes, peoplePriorK, allFeatures):
    preTrainedDataMatrix = pd.DataFrame(columns=['mean', 'cov', 'class', 'person'])
    typeData = 0
    indx = 0
    for cl in range(1, classes + 1):

        for person in range(1, peoplePriorK + 1):
            preTrainedDataMatrix.at[indx, 'cov'] = np.cov(dataMatrix[np.where(
                (dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person) * (
                        dataMatrix[:, allFeatures + 3] == cl)), 0:allFeatures][0], rowvar=False)
            preTrainedDataMatrix.at[indx, 'mean'] = np.mean(dataMatrix[np.where(
                (dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person) * (
                        dataMatrix[:, allFeatures + 3] == cl)), 0:allFeatures][0], axis=0)
            preTrainedDataMatrix.at[indx, 'class'] = cl
            preTrainedDataMatrix.at[indx, 'person'] = person
            indx += 1

    return preTrainedDataMatrix


def preTrainedDataNina5(dataMatrix, classes, peoplePriorK, evaluatedPerson, allFeatures):
    preTrainedDataMatrix = pd.DataFrame(columns=['mean', 'cov', 'class', 'person'])

    indx = 0
    for cl in range(1, classes + 1):

        for person in range(1, peoplePriorK + 1):
            if evaluatedPerson != person:
                preTrainedDataMatrix.at[indx, 'cov'] = np.cov(dataMatrix[np.where(
                    (dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 1] == cl)), 0:allFeatures][0],
                                                              rowvar=False)
                preTrainedDataMatrix.at[indx, 'mean'] = np.mean(dataMatrix[np.where(
                    (dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 1] == cl)), 0:allFeatures][0],
                                                                axis=0)
                preTrainedDataMatrix.at[indx, 'class'] = cl
                preTrainedDataMatrix.at[indx, 'person'] = person
                indx += 1

    return preTrainedDataMatrix


### Nina Pro 5

def evaluationNina5(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile,
                    startPerson, endPerson, allFeatures, printR, scaler):
    # Creating Variables

    results = pd.DataFrame(
        columns=['person', 'subset', '# shots', 'Feature Set', 'wTargetMeanQDA', 'wTargetCovQDA'])

    idx = 0

    for person in range(startPerson, endPerson + 1):

        trainFeaturesGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] != person)), 0:allFeatures][0]
        trainLabelsGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] != person)), allFeatures + 1][0]

        testFeatures = \
            dataMatrix[np.where((dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 2] >= 5)),
            0:allFeatures][0]
        testLabels = dataMatrix[
            np.where((dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 2] >= 5)), allFeatures + 1][
            0].T

        # 4 cycles - cross_validation for 4 cycles
        for shot in range(1, numberShots + 1):
            # shot = 4
            subset = tuple(range(1, shot + 1))
            # for subset in itertools.combinations(combinationSet, shot):

            trainFeatures = np.empty((0, allFeatures))
            trainLabels = []
            for auxIndex in range(np.size(subset)):
                trainFeatures = np.vstack((trainFeatures, dataMatrix[np.where(
                    (dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 2] == subset[auxIndex])),
                                                          0:allFeatures][0]))
                trainLabels = np.hstack((trainLabels, dataMatrix[np.where(
                    (dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 2] == subset[auxIndex]))
                , allFeatures + 1][0].T))

            trainFeaturesGen = np.vstack((trainFeaturesGenPre, trainFeatures))
            trainLabelsGen = np.hstack((trainLabelsGenPre, trainLabels))

            k = 1 - (np.log(shot) / np.log(numberShots + 1))

            scaler.fit(trainFeatures)

            trainFeatures = scaler.transform(trainFeatures)

            trainFeaturesGen = scaler.transform(trainFeaturesGen)
            testFeaturesTransform = scaler.transform(testFeatures)

            dataPK, allFeaturesPK = preprocessingPK(dataMatrix, allFeatures, scaler)

            preTrainedDataMatrix = preTrainedDataNina5(dataPK, classes, peoplePriorK, person, allFeaturesPK)
            currentValues = currentDistributionValues(trainFeatures, trainLabels, classes, allFeaturesPK)
            pkValues = currentDistributionValues(trainFeaturesGen, trainLabelsGen, classes, allFeaturesPK)

            results, idx = resultsDataframe(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels,
                                            classes, allFeaturesPK, results, testFeaturesTransform, testLabels, idx,
                                            person, subset, featureSet, nameFile, printR, k, pkValues)

    return results
