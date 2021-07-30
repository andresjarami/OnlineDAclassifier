import csv
import time

import numpy as np
import scipy.io
import scipy.linalg

import Features as Features


def saveFile(logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix,
             wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3, windowFile, database):
    timesFeatures = np.vstack((timeFeatureSet1, timeFeatureSet2, timeFeatureSet3))
    auxName = 'timesFeatures' + windowFile
    myFile = database + '/' + auxName + '.npy'
    np.save(myFile,timesFeatures)

    auxName = 'mav' + windowFile
    myFile = database + '/' + auxName + '.npy'
    np.save(myFile, mavMatrix)

    auxName = 'wl' + windowFile
    myFile = database + '/' + auxName + '.npy'
    np.save(myFile, wlMatrix)

    auxName = 'zc' + windowFile
    myFile = database + '/' + auxName + '.npy'
    np.save(myFile, zcMatrix)

    auxName = 'ssc' + windowFile
    myFile = database + '/' + auxName + '.npy'
    np.save(myFile, sscMatrix)

    auxName = 'lscale' + windowFile
    myFile = database + '/' + auxName + '.npy'
    np.save(myFile, lscaleMatrix)

    auxName = 'mfl' + windowFile
    myFile = database + '/' + auxName + '.npy'
    np.save(myFile, mflMatrix)

    auxName = 'msr' + windowFile
    myFile = database + '/' + auxName + '.npy'
    np.save(myFile, msrMatrix)

    auxName = 'wamp' + windowFile
    myFile = database + '/' + auxName + '.npy'
    np.save(myFile, wampMatrix)

    auxName = 'logvar' + windowFile
    myFile = database + '/' + auxName + '.npy'
    np.save(myFile, logvarMatrix)


def appendFeatureMatrix(logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix,
                        wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3, wi, wf, ch, train, person, label,
                        repetition):
    t = time.time()
    logvarMatrix.append(
        np.hstack((Features.logVAR(auxEMG[wi:wf], ch), np.array([train, person, label, repetition]))))
    timeFeatureSet1.append(time.time() - t)
    t = time.time()
    mavMatrix.append(np.hstack((Features.MAV(auxEMG[wi:wf], ch), np.array([train, person, label, repetition]))))
    wlMatrix.append(
        np.hstack((Features.WL(auxEMG[wi:wf], ch), np.array([train, person, label, repetition]))))
    zcMatrix.append(np.hstack((Features.ZC(auxEMG[wi:wf], ch), np.array([train, person, label, repetition]))))
    sscMatrix.append(np.hstack((Features.SSC(auxEMG[wi:wf], ch), np.array([train, person, label, repetition]))))
    timeFeatureSet2.append(time.time() - t)
    t = time.time()
    lscaleMatrix.append(
        np.hstack((Features.Lscale(auxEMG[wi:wf], ch), np.array([train, person, label, repetition]))))
    mflMatrix.append(np.hstack((Features.MFL(auxEMG[wi:wf], ch), np.array([train, person, label, repetition]))))
    msrMatrix.append(np.hstack((Features.MSR(auxEMG[wi:wf], ch), np.array([train, person, label, repetition]))))
    wampMatrix.append(np.hstack((Features.WAMP(auxEMG[wi:wf], ch), np.array([train, person, label, repetition]))))
    timeFeatureSet3.append(time.time() - t)

    return logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix, wampMatrix, \
           timeFeatureSet1, timeFeatureSet2, timeFeatureSet3


def emgCoteAllard(group, gender, person, carpet, gesture):
    myarray = np.fromfile(
        place + '/MyoArmbandDataset-master/' + group + '/' + gender + str(person) + '/' + carpet + '/classe_' + str(
            gesture) + '.dat', dtype=np.int16)
    myarray = np.array(myarray, dtype=np.float32)
    emg = np.reshape(myarray, (int(len(myarray) / 8), 8))
    return emg


for database in ['EPN']:
    sampleRate = 200
    window = 295
    overlap = 290
    windowFile = '_295ms'
    windowSamples = int(window * sampleRate / 1000)
    incrmentSamples = windowSamples - int(overlap * sampleRate / 1000)
    timeFeatureSet1 = []
    timeFeatureSet2 = []
    timeFeatureSet3 = []
    logvarMatrix = []
    mavMatrix = []
    wlMatrix = []
    zcMatrix = []
    sscMatrix = []
    lscaleMatrix = []
    mflMatrix = []
    msrMatrix = []
    wampMatrix = []
    rmsMatrix = []
    # place = '../../../DA-basedAdaptationTechnique/FewShotLearningEMG/data'
    place = '..'

    if database == 'Nina5':

        ## NINA PRO 5 DATABASE
        rpt = 6
        ch = 16
        classes = 18
        people = 10

        for person in range(1, people + 1):

            aux = scipy.io.loadmat(place + '/ninaDB5/s' + str(person) + '/S' + str(person) + '_E2_A1.mat')
            auxEMG = aux['emg']
            auxRestimulus = aux['restimulus']

            stack = 0
            rp = 1
            stackR = 0
            rpR = 0
            auxIdx = 0
            auxNoGes = 0
            for i in range(np.size(auxRestimulus)):

                if rp <= 4:
                    ty = 0  # For training(0) or Testing(1)
                else:
                    ty = 1  # For training(0) or Testing(1)
                if auxRestimulus[i] != 0 and stack == 0:
                    aux1 = i
                    stack = 1
                    cl = int(auxRestimulus[i])

                elif auxRestimulus[i] == 0 and stack == 1:
                    aux2 = i
                    stack = 0
                    wi = aux1
                    segments = int((aux2 - aux1 - windowSamples) / incrmentSamples + 1)
                    for w in range(segments):
                        wf = wi + windowSamples

                        logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix, \
                        wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3 = appendFeatureMatrix(
                            logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix,
                            msrMatrix, wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3, wi, wf, ch, ty,
                            person, cl, rp)

                        wi += incrmentSamples

                    rp = rp + 1
                    if rp == 7:
                        rp = 1

                if rpR <= rpt:
                    if rpR <= 4:
                        ty = 0  # For training(0) or Testing(1)
                    else:
                        ty = 1  # For training(0) or Testing(1)
                    if auxRestimulus[i] == 0 and stackR == 0:
                        aux1R = i
                        stackR = 1
                        clR = 18

                    elif auxRestimulus[i] != 0 and stackR == 1:
                        aux2R = i
                        stackR = 0
                        wiR = aux1R
                        if rpR != 0:
                            segments = int(((aux2R - aux1R) - windowSamples) / (incrmentSamples) + 1)
                            for w in range(segments):
                                wfR = wiR + windowSamples

                                logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix, \
                                wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3 = appendFeatureMatrix(
                                    logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix,
                                    msrMatrix, wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3, wiR, wfR,
                                    ch, ty, person, clR, rpR)

                                wiR += incrmentSamples
                        rpR += 1

        saveFile(logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix,
                 wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3, windowFile, database)

    elif database == 'Cote':
        # COTE ALLARD DATABASE

        # Parameters
        ch = 8
        classes = 7
        peopleGroups = 2
        genders = 2  # Female and Male
        gesturesPerFolder = 28  # Number of Getures per folder

        person = 1
        for i in range(peopleGroups):

            for j in range(genders):
                if i == 0 and j == 0:
                    group = 'PreTrainingDataset'
                    gender = 'Female'
                    carpets = np.array(['training0'])
                    people = 7
                elif i == 0 and j == 1:
                    group = 'PreTrainingDataset'
                    gender = 'Male'
                    carpets = np.array(['training0'])
                    people = 12
                elif i == 1 and j == 0:
                    group = 'EvaluationDataset'
                    gender = 'Female'
                    carpets = np.array(['training0', 'Test0', 'Test1'])
                    people = 2
                elif i == 1 and j == 1:
                    group = 'EvaluationDataset'
                    gender = 'Male'
                    carpets = np.array(['training0', 'Test0', 'Test1'])
                    people = 15

                for person_i in range(people):

                    rp = 1
                    for carpet in carpets:
                        if carpet == 'training0':
                            ty = 0  # For training(0) or Testing(1)
                        else:
                            ty = 1  # For training(0) or Testing(1)

                        for gesture in range(gesturesPerFolder):
                            cl = gesture
                            while cl > 6:
                                cl = cl - 7

                            auxEMG = emgCoteAllard(group, gender, person_i, carpet, gesture)

                            wi = 0

                            segments = int((len(auxEMG) - windowSamples) / incrmentSamples + 1)
                            for w in range(segments):
                                wf = wi + windowSamples

                                logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix, \
                                wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3 = appendFeatureMatrix(
                                    logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix,
                                    msrMatrix, wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3, wi, wf,
                                    ch, ty, person, cl + 1, rp)

                                wi += incrmentSamples

                            if cl == 6:
                                rp = rp + 1
                    person += 1

        saveFile(logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix,
                 wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3, windowFile, database)

    elif database == 'EPN':

        # EPN DATABASE

        repetitions = 25
        ch = 8
        classes = 5
        people = 60
        types = 2  # For training(0) or Testing(1)

        for person in range(1, people + 1):
            print(person)
            for cl in range(1, classes + 1):
                rp = 0
                for ty in range(types):
                    for repetition in range(1, repetitions + 1):
                        rp += 1
                        aux = scipy.io.loadmat(
                            place + '/CollectedData/allUsers_data/detectedData/emg_person' + str(person) + '_class' + str(
                                cl) + '_rpt' + str(repetition) + '_type' + str(ty) + '.mat')
                        auxEMG = aux['emg']

                        wi = 0
                        segments = int((len(auxEMG) - windowSamples) / incrmentSamples + 1)
                        for w in range(segments):
                            wf = wi + windowSamples

                            logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix, \
                            wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3 = appendFeatureMatrix(
                                logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix,
                                msrMatrix, wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3, wi, wf, ch,
                                ty, person, cl, rp)

                            wi += incrmentSamples

        saveFile(logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix,
                 wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3, windowFile, database)
