import scipy.io
import numpy as np
import scipy.linalg
import csv
import time
from scipy import signal
from math import pi


def MAVch(EMG, ch):
    if len(EMG) == 1:
        mavVector = np.zeros((1, ch))[0]
    else:
        EMG = abs(EMG)
        mavVector = EMG.mean(axis=0)
    return mavVector


def WLch(EMG, ch):
    if len(EMG) == 1:
        wlVector = np.zeros((1, ch))
    else:
        wlVector = np.zeros([1, ch])
        for i in range(0, ch):
            x_f = EMG[1:np.size(EMG, 0), i]
            x = EMG[0:np.size(EMG, 0) - 1, i]
            wlVector[0, i] = np.sum(abs(x_f - x))
    return wlVector


def ZCch(EMG, ch):
    if len(EMG) == 1:
        zcVector = np.zeros((1, ch))
    else:
        zcVector = np.zeros([1, ch])
        for i in range(0, ch):
            zcVector[0, i] = np.size(np.where(np.diff(np.sign(EMG[:, i]))), 1)
    return zcVector


def SSCch(EMG, ch):
    if len(EMG) == 1:
        sscVector = np.zeros((1, ch))
    else:
        sscVector = np.zeros([1, ch])
        for i in range(0, ch):
            x = EMG[1:np.size(EMG, 0) - 1, i]
            x_b = EMG[0:np.size(EMG, 0) - 2, i]
            x_f = EMG[2:np.size(EMG, 0), i]
            sscVector[0, i] = np.sum(abs((x - x_b) * (x - x_f)))
    return sscVector


def Lscalech(EMG, ch):
    if len(EMG) == 1:
        LscaleVector = np.zeros((1, ch))
    else:
        LscaleVector = np.zeros([1, ch])
        for i in range(0, ch):
            lengthAux = np.size(EMG[:, i], 0)
            Matrix = np.sort(np.transpose(EMG[:, i]))
            aux = (1 / lengthAux) * np.sum((np.arange(1, lengthAux, 1) / (lengthAux - 1)) * Matrix[1:lengthAux + 1])
            aux3 = np.array([[aux], [np.mean(EMG[:, i])]])
            aux5 = np.array([[2], [-1]])
            LscaleVector[0, i] = np.sum(aux5 * aux3)
    return LscaleVector


def MFLch(EMG, ch):
    if len(EMG) == 1:
        mflVector = np.zeros((1, ch))
    else:
        mflVector = np.zeros([1, ch])
        for i in range(0, ch):
            x_f = EMG[1:np.size(EMG, 0), i]
            x = EMG[0:np.size(EMG, 0) - 1, i]
            mflVector[0, i] = np.log10(np.sqrt(np.sum((x_f - x) ** 2)))
    return mflVector


def MSRch(EMG, ch):
    if len(EMG) == 1:
        msrVector = np.zeros((1, ch))
    else:
        msrVector = np.zeros([1, ch])
        for i in range(0, ch):
            msrVector[0, i] = np.sum(np.sqrt(abs(EMG[:, i]))) / np.size(EMG[:, i], 0)
    return msrVector


def WAMPch(EMG, ch):
    if len(EMG) == 1:
        wampVector = np.zeros((1, ch))
    else:
        wampVector = np.zeros([1, ch])
        for i in range(0, ch):
            x_f = EMG[1:np.size(EMG, 0), i]
            x = EMG[0:np.size(EMG, 0) - 1, i]
            wampVector[0, i] = np.sum((np.sign(x - x_f) + 1) / 2)
    return wampVector


def RMSch(EMG, ch):
    if len(EMG) == 1:
        rmsVector = np.zeros((1, ch))
    else:
        rmsVector = np.zeros([1, ch])
        for i in range(0, ch):
            rmsVector[0, i] = np.sqrt(np.sum((EMG[:, i]) ** 2) / np.size(EMG[:, i], 0))
    return rmsVector


def IAVch(EMG, ch):
    if len(EMG) == 1:
        iavVector = np.zeros((1, ch))
    else:
        iavVector = np.zeros([1, ch])
        for i in range(0, ch):
            iavVector[0, i] = np.sum(abs(EMG[:, i]) / np.size(EMG[:, i], 0))
    return iavVector


def DASDVch(EMG, ch):
    if len(EMG) == 1:
        dasdvVector = np.zeros((1, ch))
    else:
        dasdvVector = np.zeros([1, ch])
        for i in range(0, ch):
            x_f = EMG[1:np.size(EMG, 0), i]
            x = EMG[0:np.size(EMG, 0) - 1, i]
            dasdvVector[0, i] = np.sqrt(np.sum((x_f - x) ** 2) / np.size(x, 0))
    return dasdvVector


def VARch(EMG, ch):
    if len(EMG) == 1:
        varVector = np.zeros((1, ch))
    else:
        varVector = np.zeros([1, ch])
        for i in range(ch):
            varVector[0, i] = np.sum((EMG[:, i]-np.mean(EMG[:, i])) ** 2) / (np.size(EMG[:, i], 0) - 1)
    return varVector

def logVARch(EMG, ch):
    if len(EMG) == 1:
        logVarVector = np.zeros((1, ch))
    else:
        logVarVector = np.zeros([1, ch])
        for i in range(ch):
            logVarVector[0, i] = np.log(np.sum((EMG[:, i]-np.mean(EMG[:, i])) ** 2) / (np.size(EMG[:, i], 0) - 1))
    return logVarVector

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


for database in ['Nina5','Cote','EPN']:
    sampleRate = 200
    window = 295
    overlap = 290
    windowFile='295'

    if database == 'Nina5':

        ## NINA PRO 5 DATABASE
        rpt = 6
        ch = 16
        classes = 18
        people = 10

        windowSamples = int(window * sampleRate / 1000)
        incrmentSamples = windowSamples - int(overlap * sampleRate / 1000)

        t1 = []
        t2 = []
        t3 = []
        for person in range(1, people + 1):
            aux = scipy.io.loadmat('../data/ninaDB5/s' + str(person) + '/S' + str(person) + '_E2_A1.mat')
            auxEMG = aux['emg']
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

                        t = time.time()
                        logvarMatrix.append(np.hstack((logVARch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                        t1.append(time.time() - t)
                        t = time.time()
                        mavMatrix.append(np.hstack((MAVch(auxEMG[wi:wf], ch), np.array([person, cl, rp]))))
                        wlMatrix.append(np.hstack((WLch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                        zcMatrix.append(np.hstack((ZCch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                        sscMatrix.append(np.hstack((SSCch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                        t2.append(time.time() - t )
                        t = time.time()
                        lscaleMatrix.append(np.hstack((Lscalech(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                        mflMatrix.append(np.hstack((MFLch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                        msrMatrix.append(np.hstack((MSRch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                        wampMatrix.append(np.hstack((WAMPch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                        t3.append(time.time() - t)

                        wi += incrmentSamples

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

                                t = time.time()
                                logvarMatrix.append(np.hstack((logVARch(auxEMG[wiR:wfR], ch)[0], np.array([person, clR, rpR]))))
                                t1.append(time.time() - t)
                                t = time.time()
                                mavMatrix.append(np.hstack((MAVch(auxEMG[wiR:wfR], ch), np.array([person, clR, rpR]))))
                                wlMatrix.append(np.hstack((WLch(auxEMG[wiR:wfR], ch)[0], np.array([person, clR, rpR]))))
                                zcMatrix.append(np.hstack((ZCch(auxEMG[wiR:wfR], ch)[0], np.array([person, clR, rpR]))))
                                sscMatrix.append(np.hstack((SSCch(auxEMG[wiR:wfR], ch)[0], np.array([person, clR, rpR]))))
                                t2.append(time.time() - t)
                                t = time.time()
                                lscaleMatrix.append(
                                    np.hstack((Lscalech(auxEMG[wiR:wfR], ch)[0], np.array([person, clR, rpR]))))
                                mflMatrix.append(np.hstack((MFLch(auxEMG[wiR:wfR], ch)[0], np.array([person, clR, rpR]))))
                                msrMatrix.append(np.hstack((MSRch(auxEMG[wiR:wfR], ch)[0], np.array([person, clR, rpR]))))
                                wampMatrix.append(np.hstack((WAMPch(auxEMG[wiR:wfR], ch)[0], np.array([person, clR, rpR]))))
                                t3.append(time.time() - t)

                                wiR += incrmentSamples
                        rpR += 1

        timesFeatures = np.vstack((t1, t2, t3))
        auxName = 'timesFeatures'+windowFile
        myFile = 'ExtractedDataNinaDB5/' + auxName + '.csv'
        np.savetxt(myFile, timesFeatures, delimiter=',')

        auxName = 'mavMatrix'+windowFile
        myFile = open('ExtractedDataNinaDB5/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(mavMatrix)
        auxName = 'wlMatrix'+windowFile
        myFile = open('ExtractedDataNinaDB5/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(wlMatrix)
        auxName = 'zcMatrix'+windowFile
        myFile = open('ExtractedDataNinaDB5/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(zcMatrix)
        auxName = 'sscMatrix'+windowFile
        myFile = open('ExtractedDataNinaDB5/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(sscMatrix)
        auxName = 'lscaleMatrix'+windowFile
        myFile = open('ExtractedDataNinaDB5/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(lscaleMatrix)
        auxName = 'mflMatrix'+windowFile
        myFile = open('ExtractedDataNinaDB5/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(mflMatrix)
        auxName = 'msrMatrix'+windowFile
        myFile = open('ExtractedDataNinaDB5/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(msrMatrix)
        auxName = 'wampMatrix'+windowFile
        myFile = open('ExtractedDataNinaDB5/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(wampMatrix)
        auxName = 'logvarMatrix' + windowFile
        myFile = open('ExtractedDataNinaDB5/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(logvarMatrix)

    elif database == 'Cote':
        # COTE ALLARD DATABASE

        ch = 8
        classes = 7
        peopleFemalePT = 7
        peopleMalePT = 12
        peopleFemaleE = 2
        peopleMaleE = 15
        types = 2
        genders = 2
        filesPerFolder = 28

        windowSamples = int(window * sampleRate / 1000)
        incrmentSamples = windowSamples - int(overlap * sampleRate / 1000)
        t1 = []
        t2 = []
        t3 = []


        def emgMatrix(ty, gender, person, carpet, number):
            myarray = np.fromfile(
                '../data/MyoArmbandDataset-master/' + ty + '/' + gender + str(person) + '/' + carpet + '/classe_' + str(
                    number) + '.dat', dtype=np.int16)
            myarray = np.array(myarray, dtype=np.float32)
            emg = np.reshape(myarray, (int(len(myarray) / 8), 8))
            return emg


        per = 0
        for tyi in range(0, types):

            for genderi in range(0, genders):
                if tyi == 0 and genderi == 0:
                    ty = 'PreTrainingDataset'
                    gender = 'Female'
                    carpets = np.array(['training0'])
                    people = peopleFemalePT
                elif tyi == 0 and genderi == 1:
                    ty = 'PreTrainingDataset'
                    gender = 'Male'
                    carpets = np.array(['training0'])
                    people = peopleMalePT
                elif tyi == 1 and genderi == 0:
                    ty = 'EvaluationDataset'
                    gender = 'Female'
                    carpets = np.array(['training0', 'Test0', 'Test1'])
                    people = peopleFemaleE
                elif tyi == 1 and genderi == 1:
                    ty = 'EvaluationDataset'
                    gender = 'Male'
                    carpets = np.array(['training0', 'Test0', 'Test1'])
                    people = peopleMaleE

                for person in range(people):

                    rp = 1
                    for carpet in carpets:
                        if carpet == 'training0':
                            carp = 1
                        else:
                            carp = 2

                        for number in range(filesPerFolder):
                            cl = number
                            while cl > 6:
                                cl = cl - 7

                            auxEMG = emgMatrix(ty, gender, person, carpet, number)

                            wi = 0

                            segments = int((len(auxEMG) - windowSamples) / incrmentSamples + 1)
                            for w in range(segments):
                                wf = wi + windowSamples

                                t = time.time()
                                logvarMatrix.append(
                                    np.hstack((logVARch(auxEMG[wi:wf], ch)[0], np.array([tyi, per, carp, cl, rp]))))
                                t1.append(time.time() - t)
                                t = time.time()
                                mavMatrix.append(
                                    np.hstack((MAVch(auxEMG[wi:wf], ch), np.array([tyi, per, carp, cl, rp]))))
                                wlMatrix.append(
                                    np.hstack((WLch(auxEMG[wi:wf], ch)[0], np.array([tyi, per, carp, cl, rp]))))
                                zcMatrix.append(
                                    np.hstack((ZCch(auxEMG[wi:wf], ch)[0], np.array([tyi, per, carp, cl, rp]))))
                                sscMatrix.append(
                                    np.hstack((SSCch(auxEMG[wi:wf], ch)[0], np.array([tyi, per, carp, cl, rp]))))
                                t2.append(time.time() - t)
                                t = time.time()
                                lscaleMatrix.append(
                                    np.hstack((Lscalech(auxEMG[wi:wf], ch)[0], np.array([tyi, per, carp, cl, rp]))))
                                mflMatrix.append(
                                    np.hstack((MFLch(auxEMG[wi:wf], ch)[0], np.array([tyi, per, carp, cl, rp]))))
                                msrMatrix.append(
                                    np.hstack((MSRch(auxEMG[wi:wf], ch)[0], np.array([tyi, per, carp, cl, rp]))))
                                wampMatrix.append(
                                    np.hstack((WAMPch(auxEMG[wi:wf], ch)[0], np.array([tyi, per, carp, cl, rp]))))
                                t3.append(time.time() - t)

                                wi += incrmentSamples

                            if cl == 6:
                                rp = rp + 1
                    per += 1

        timesFeatures = np.vstack((t1, t2, t3))
        auxName = 'timesFeatures'+windowFile
        myFile = 'ExtractedDataCoteAllard/' + auxName + '.csv'
        np.savetxt(myFile, timesFeatures, delimiter=',')

        auxName = 'mavMatrix'+windowFile
        myFile = open('ExtractedDataCoteAllard/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(mavMatrix)
        auxName = 'wlMatrix'+windowFile
        myFile = open('ExtractedDataCoteAllard/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(wlMatrix)
        auxName = 'zcMatrix'+windowFile
        myFile = open('ExtractedDataCoteAllard/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(zcMatrix)
        auxName = 'sscMatrix'+windowFile
        myFile = open('ExtractedDataCoteAllard/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(sscMatrix)
        auxName = 'lscaleMatrix'+windowFile
        myFile = open('ExtractedDataCoteAllard/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(lscaleMatrix)
        auxName = 'mflMatrix'+windowFile
        myFile = open('ExtractedDataCoteAllard/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(mflMatrix)
        auxName = 'msrMatrix'+windowFile
        myFile = open('ExtractedDataCoteAllard/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(msrMatrix)
        auxName = 'wampMatrix'+windowFile
        myFile = open('ExtractedDataCoteAllard/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(wampMatrix)
        auxName = 'logvarMatrix' + windowFile
        myFile = open('ExtractedDataCoteAllard/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(logvarMatrix)


    elif database == 'EPN':

        # EPN DATABASE

        rpt = 25
        ch = 8
        classes = 5
        people = 60
        types = 2
        windowSamples = int(window * sampleRate / 1000)
        incrmentSamples = windowSamples - int(overlap * sampleRate / 1000)
        t1 = []
        t2 = []
        t3 = []

        for ty in range(0, types):
            for person in range(1, people + 1):
                for cl in range(1, classes + 1):
                    for rp in range(1, rpt + 1):
                        aux = scipy.io.loadmat(
                            '../data/CollectedData/detectedData/emg_person' + str(person) + '_class' + str(
                                cl) + '_rpt' + str(
                                rp) + '_type' + str(ty) + '.mat')
                        auxEMG = aux['emg']

                        wi = 0
                        segments = int((len(auxEMG) - windowSamples) / incrmentSamples + 1)
                        for w in range(segments):
                            wf = wi + windowSamples

                            t = time.time()
                            logvarMatrix.append(np.hstack((logVARch(auxEMG[wi:wf], ch)[0], np.array([ty, person, cl, rp]))))
                            t1.append(time.time() - t)
                            t = time.time()
                            mavMatrix.append(np.hstack((MAVch(auxEMG[wi:wf], ch), np.array([ty, person, cl, rp]))))
                            wlMatrix.append(np.hstack((WLch(auxEMG[wi:wf], ch)[0], np.array([ty, person, cl, rp]))))
                            zcMatrix.append(np.hstack((ZCch(auxEMG[wi:wf], ch)[0], np.array([ty, person, cl, rp]))))
                            sscMatrix.append(np.hstack((SSCch(auxEMG[wi:wf], ch)[0], np.array([ty, person, cl, rp]))))
                            t2.append(time.time() - t)
                            t = time.time()
                            lscaleMatrix.append(np.hstack((Lscalech(auxEMG[wi:wf], ch)[0], np.array([ty, person, cl, rp]))))
                            mflMatrix.append(np.hstack((MFLch(auxEMG[wi:wf], ch)[0], np.array([ty, person, cl, rp]))))
                            msrMatrix.append(np.hstack((MSRch(auxEMG[wi:wf], ch)[0], np.array([ty, person, cl, rp]))))
                            wampMatrix.append(np.hstack((WAMPch(auxEMG[wi:wf], ch)[0], np.array([ty, person, cl, rp]))))
                            t3.append((time.time() - t))

                            wi += incrmentSamples

        timesFeatures = np.vstack((t1, t2, t3))
        auxName = 'timesFeatures'+windowFile
        myFile = 'ExtractedDataCollectedData/' + auxName + '.csv'
        np.savetxt(myFile, timesFeatures, delimiter=',')

        auxName = 'mavMatrix'+windowFile
        myFile = open('ExtractedDataCollectedData/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(mavMatrix)
        auxName = 'wlMatrix'+windowFile
        myFile = open('ExtractedDataCollectedData/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(wlMatrix)
        auxName = 'zcMatrix'+windowFile
        myFile = open('ExtractedDataCollectedData/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(zcMatrix)
        auxName = 'sscMatrix'+windowFile
        myFile = open('ExtractedDataCollectedData/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(sscMatrix)
        auxName = 'lscaleMatrix'+windowFile
        myFile = open('ExtractedDataCollectedData/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(lscaleMatrix)
        auxName = 'mflMatrix'+windowFile
        myFile = open('ExtractedDataCollectedData/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(mflMatrix)
        auxName = 'msrMatrix'+windowFile
        myFile = open('ExtractedDataCollectedData/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(msrMatrix)
        auxName = 'wampMatrix'+windowFile
        myFile = open('ExtractedDataCollectedData/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(wampMatrix)
        auxName = 'logvarMatrix' + windowFile
        myFile = open('ExtractedDataCollectedData/' + auxName + '.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(logvarMatrix)
