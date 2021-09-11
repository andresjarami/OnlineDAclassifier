import time
import numpy as np
import scipy.io
import scipy.linalg

import Features as Features


def saveFile(logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix,
             wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3, windowFile, database, person):
    personFile = '_' + str(person)

    auxName = 'mav' + windowFile + personFile
    myFile = database + '/' + auxName + '.npy'
    np.save(myFile, mavMatrix)

    auxName = 'wl' + windowFile + personFile
    myFile = database + '/' + auxName + '.npy'
    np.save(myFile, wlMatrix)

    auxName = 'zc' + windowFile + personFile
    myFile = database + '/' + auxName + '.npy'
    np.save(myFile, zcMatrix)

    auxName = 'ssc' + windowFile + personFile
    myFile = database + '/' + auxName + '.npy'
    np.save(myFile, sscMatrix)

    auxName = 'lscale' + windowFile + personFile
    myFile = database + '/' + auxName + '.npy'
    np.save(myFile, lscaleMatrix)

    auxName = 'mfl' + windowFile + personFile
    myFile = database + '/' + auxName + '.npy'
    np.save(myFile, mflMatrix)

    auxName = 'msr' + windowFile + personFile
    myFile = database + '/' + auxName + '.npy'
    np.save(myFile, msrMatrix)

    auxName = 'wamp' + windowFile + personFile
    myFile = database + '/' + auxName + '.npy'
    np.save(myFile, wampMatrix)

    auxName = 'logvar' + windowFile + personFile
    myFile = database + '/' + auxName + '.npy'
    np.save(myFile, logvarMatrix)

    timesFeatures = np.vstack((timeFeatureSet1, timeFeatureSet2, timeFeatureSet3))
    auxName = 'timesFeatures' + windowFile + personFile
    myFile = database + '/' + auxName + '.npy'
    np.save(myFile, timesFeatures)


def appendFeatureMatrix(emg, logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix,
                        wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3, wi, wf, ch, day, person, label,
                        repetition):
    t = time.time()
    logvarMatrix.append(
        np.hstack((Features.logVAR(emg[wi:wf], ch), np.array([day, person, label, repetition]))))
    timeFeatureSet1.append(time.time() - t)
    t = time.time()
    mavMatrix.append(np.hstack((Features.MAV(emg[wi:wf], ch), np.array([day, person, label, repetition]))))
    wlMatrix.append(
        np.hstack((Features.WL(emg[wi:wf], ch), np.array([day, person, label, repetition]))))
    zcMatrix.append(np.hstack((Features.ZC(emg[wi:wf], ch), np.array([day, person, label, repetition]))))
    sscMatrix.append(np.hstack((Features.SSC(emg[wi:wf], ch), np.array([day, person, label, repetition]))))
    timeFeatureSet2.append(time.time() - t)
    t = time.time()
    lscaleMatrix.append(
        np.hstack((Features.Lscale(emg[wi:wf], ch), np.array([day, person, label, repetition]))))
    mflMatrix.append(np.hstack((Features.MFL(emg[wi:wf], ch), np.array([day, person, label, repetition]))))
    msrMatrix.append(np.hstack((Features.MSR(emg[wi:wf], ch), np.array([day, person, label, repetition]))))
    wampMatrix.append(np.hstack((Features.WAMP(emg[wi:wf], ch), np.array([day, person, label, repetition]))))
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


def min_max_normalization(emg, bit_resolution):
    a = -1
    b = 1
    min = np.floor(-(2 ** bit_resolution - 1) / 2)
    max = np.floor((2 ** bit_resolution - 1) / 2)
    return a + (b - a) * (emg - min) / (max - min)
    # return emg


for database in ['Nina5', 'Cote', 'Capgmyo_dbb', 'LongTerm3DC', 'EPN_612']:

    window = 290
    overlap = 280
    windowFile = '_290ms'
    place = '../Datasets'
    print(database)

    if database == 'Nina5':
        ## NINA PRO 5 DATABASE
        sampleRate = 200
        windowSamples = int(window * sampleRate / 1000)
        incrmentSamples = windowSamples - int(overlap * sampleRate / 1000)

        rpt = 6
        ch = 16
        classes = 18
        people = 10
        day = 1
        bit_resolution = 8  # myoarmband

        for person in range(1, people + 1):

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

            aux = scipy.io.loadmat(place + '/ninaDB5/s' + str(person) + '/S' + str(person) + '_E2_A1.mat')
            emg = aux['emg']
            emg = min_max_normalization(emg, bit_resolution)

            auxRestimulus = aux['restimulus']

            stack = 0
            rp = 1
            stackR = 0
            rpR = 0
            auxIdx = 0
            auxNoGes = 0
            for i in range(np.size(auxRestimulus)):

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
                            emg, logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix,
                            msrMatrix, wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3, wi, wf, ch, day,
                            person, cl, rp)

                        wi += incrmentSamples
                    print(person, day, cl, rp)

                    rp = rp + 1
                    if rp == 7:
                        rp = 1

                if rpR <= rpt:
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
                                    emg, logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix,
                                    mflMatrix, msrMatrix, wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3,
                                    wiR, wfR, ch, day, person, clR, rpR)

                                wiR += incrmentSamples
                            print(person, day, clR, rpR)
                        rpR += 1
            print('################PERSON' + str(person) + '#########')
            saveFile(logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix,
                     wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3, windowFile, database, person)

    elif database == 'Cote':
        # COTE ALLARD DATABASE
        sampleRate = 200
        windowSamples = int(window * sampleRate / 1000)
        incrmentSamples = windowSamples - int(overlap * sampleRate / 1000)

        ch = 8
        classes = 7
        peopleGroups = 2
        genders = 2  # Female and Male
        gesturesPerFolder = 28  # Number of Getures per folder
        day = 1
        bit_resolution = 8  # myoarmband

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

                    rp = 1
                    for carpet in carpets:

                        for gesture in range(gesturesPerFolder):
                            cl = gesture
                            while cl > 6:
                                cl = cl - 7

                            emg = emgCoteAllard(group, gender, person_i, carpet, gesture)
                            emg = min_max_normalization(emg, bit_resolution)

                            wi = 0

                            segments = int((len(emg) - windowSamples) / incrmentSamples + 1)
                            for w in range(segments):
                                wf = wi + windowSamples

                                logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix, \
                                wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3 = appendFeatureMatrix(
                                    emg, logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix,
                                    mflMatrix, msrMatrix, wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3,
                                    wi, wf, ch, day, person, cl + 1, rp)

                                wi += incrmentSamples
                            print(person, day, cl + 1, rp)
                            if cl == 6:
                                rp = rp + 1
                    print('################PERSON' + str(person) + '#########')
                    saveFile(logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix,
                             wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3, windowFile, database,
                             person)
                    person += 1

    elif database == 'EPN_120':
        # EPN_612 DATABASE
        sampleRate = 200
        windowSamples = int(window * sampleRate / 1000)
        incrmentSamples = windowSamples - int(overlap * sampleRate / 1000)

        repetitions = 50
        ch = 8
        classes = 5
        people = 120
        day = 1
        bit_resolution = 8  # myoarmband

        for person in range(1, people + 1):
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

            for cl in range(1, classes + 1):
                for rp in range(1, repetitions + 1):
                    aux = scipy.io.loadmat(
                        place + '/EMG_EPN120_Dataset/segmented_data/emg_person' + str(person) + '_class' + str(
                            cl) + '_rpt' + str(rp) + '.mat')
                    emg = aux['emg'] # Data is already normalized (8 bit-resolution)

                    wi = 0
                    segments = int((len(emg) - windowSamples) / incrmentSamples + 1)
                    for w in range(segments):
                        wf = wi + windowSamples
                        logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix, \
                        wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3 = appendFeatureMatrix(
                            emg, logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix,
                            msrMatrix, wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3, wi, wf, ch, day,
                            person, cl, rp)
                        wi += incrmentSamples
                    print(person, day, cl, rp)
            print('################PERSON' + str(person) + '#########')
            saveFile(logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix,
                     wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3, windowFile, database, person)

    elif database == 'LongTerm3DC':
        # LongTerm3DC DATABASE
        sampleRate = 1000
        windowSamples = int(window * sampleRate / 1000)
        incrmentSamples = windowSamples - int(overlap * sampleRate / 1000)

        repetitions = [0, 2, 3]
        ch = 10
        classes = 11
        people = 20
        days = 3
        bit_resolution = 10  # myoarmband

        person = 1
        for person_idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
            # Note that originally, 22 persons took part in this study. However, two of them (both male) had to drop out,
            # due to external circumstances. Consequently, these individuals are not included in the results and
            # analysis of this work.
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
            if person_idx == 0 or person_idx == 2:  # the dataset is incomplete
                set_days = [0, 2, 3]
            else:
                set_days = range(days)
            day = 1
            for day_idx in set_days:
                for cl in range(classes):
                    rp = 1
                    for rp_idx in repetitions:
                        # Note that in the original dataset, four cycles are recorded for each participant,
                        # with the second one recording the participant performing each gesture with maximal intensity.
                        # This second cycle was removed for this work to reduce confounding factors. In other words,
                        # cycle two and three in this work correspond to cycle three and four in the original dataset.

                        emg = []
                        with open(place + '/longterm_dataset_3DC/Participant' + str(person_idx) + '/Training' + str(
                                day_idx) + '/EMG/3dc_EMG_gesture_' + str(rp_idx) + '_' + str(cl) + '.txt') as emgFile:
                            for line in emgFile:
                                #  strip() remove the "\n" character, split separate the data in a list. np.float
                                #  transform each element of the list from a str to a float
                                emg_signal = np.float32(line.strip().split(","))
                                emg.append(emg_signal)
                        emg = min_max_normalization(np.array(emg), bit_resolution)

                        wi = 0
                        segments = int((len(emg) - windowSamples) / incrmentSamples + 1)
                        for w in range(segments):
                            wf = wi + windowSamples

                            logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix, \
                            wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3 = appendFeatureMatrix(
                                emg, logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix,
                                msrMatrix, wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3, wi, wf, ch,
                                day, person, cl + 1, rp)
                            wi += incrmentSamples
                        print(person, day, cl + 1, rp)
                        rp += 1
                day += 1

            print('################PERSON' + str(person) + '#########')
            saveFile(logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix,
                     wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3, windowFile, database, person)
            person += 1

    elif database == 'Capgmyo_dbb':
        # Capgmyo DBb DATABASE
        sampleRate = 1000
        windowSamples = int(window * sampleRate / 1000)
        incrmentSamples = windowSamples - int(overlap * sampleRate / 1000)

        repetitions = 10
        ch = 128
        classes = 8
        people = 10

        person = 0
        for person_idx in ["%.2d" % i for i in range(1, people * 2 + 1)]:
            if (int(person_idx) % 2) != 0:
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
                day = 1
                person += 1

            else:
                day = 2

            for cl in ["%.2d" % i for i in range(1, classes + 1)]:
                rp = 1
                for rp_idx in ["%.2d" % i for i in range(1, repetitions + 1)]:
                    aux = scipy.io.loadmat(
                        place + '/capgmyo_dbb/dbb-preprocessed-0' + person_idx + '/0' + person_idx + '-0' + cl +
                        '-0' + rp_idx + '.mat')
                    emg = aux['data']  # Data is already normalized (16 bit-resolution)

                    wi = 0
                    segments = int((len(emg) - windowSamples) / incrmentSamples + 1)
                    for w in range(segments):
                        wf = wi + windowSamples
                        logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix, \
                        wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3 = appendFeatureMatrix(
                            emg, logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix,
                            msrMatrix, wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3, wi, wf, ch,
                            day, person, int(cl), rp)
                        wi += incrmentSamples
                    print(person, day, int(cl), rp)
                    rp += 1

            print('################PERSON' + str(person) + '#########')
            saveFile(logvarMatrix, mavMatrix, wlMatrix, zcMatrix, sscMatrix, lscaleMatrix, mflMatrix, msrMatrix,
                     wampMatrix, timeFeatureSet1, timeFeatureSet2, timeFeatureSet3, windowFile, database, person)
