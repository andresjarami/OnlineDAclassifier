import numpy as np
import warnings

warnings.filterwarnings("error")


def logVAR(EMG, ch):
    if len(EMG) == 1:
        logVarVector = np.zeros(ch)
    else:
        logVarVector = np.zeros(ch)
        for i in range(ch):
            logVarVector[i] = np.log(np.sum((EMG[:, i] - np.mean(EMG[:, i])) ** 2) / (np.size(EMG[:, i], 0) - 1))
    return logVarVector


def MAV(EMG, ch):
    if len(EMG) == 1:
        mavVector = np.zeros(ch)
    else:
        EMG = abs(EMG)
        mavVector = EMG.mean(axis=0)
    return mavVector


def WL(EMG, ch):
    if len(EMG) == 1:
        wlVector = np.zeros(ch)
    else:
        wlVector = np.zeros(ch)
        for i in range(ch):
            x_f = EMG[1:np.size(EMG, 0), i]
            x = EMG[0:np.size(EMG, 0) - 1, i]
            wlVector[i] = np.sum(abs(x_f - x))
    return wlVector


def ZC(EMG, ch):
    if len(EMG) == 1:
        zcVector = np.zeros(ch)
    else:
        zcVector = np.zeros(ch)
        for i in range(ch):
            zcVector[i] = np.size(np.where(np.diff(np.sign(EMG[:, i]))), 1)
    return zcVector


def SSC(EMG, ch):
    if len(EMG) == 1:
        sscVector = np.zeros(ch)
    else:
        sscVector = np.zeros(ch)
        for i in range(ch):
            x = EMG[1:np.size(EMG, 0) - 1, i]
            x_b = EMG[0:np.size(EMG, 0) - 2, i]
            x_f = EMG[2:np.size(EMG, 0), i]
            sscVector[i] = np.sum(abs((x - x_b) * (x - x_f)))
    return sscVector


def Lscale(EMG, ch):
    if len(EMG) == 1:
        LscaleVector = np.zeros(ch)
    else:
        LscaleVector = np.zeros(ch)
        for i in range(ch):
            lengthAux = np.size(EMG[:, i], 0)
            Matrix = np.sort(np.transpose(EMG[:, i]))
            aux = (1 / lengthAux) * np.sum((np.arange(1, lengthAux, 1) / (lengthAux - 1)) * Matrix[1:lengthAux + 1])
            aux3 = np.array([[aux], [np.mean(EMG[:, i])]])
            aux5 = np.array([[2], [-1]])
            LscaleVector[i] = np.sum(aux5 * aux3)
    return LscaleVector


def MFL(EMG, ch):
    if len(EMG) == 1:
        mflVector = np.zeros(ch)
    else:
        mflVector = np.zeros(ch)
        for i in range(ch):
            x_f = EMG[1:np.size(EMG, 0), i]
            x = EMG[0:np.size(EMG, 0) - 1, i]
            try:
                mflVector[i] = np.log10(np.sqrt(np.sum((x_f - x) ** 2)))
            except RuntimeWarning:
                print('error negative value', x_f, x, np.sum((x_f - x) ** 2))
                mflVector[i] = 0

    return mflVector


def MSR(EMG, ch):
    if len(EMG) == 1:
        msrVector = np.zeros(ch)
    else:
        msrVector = np.zeros(ch)
        for i in range(0, ch):
            msrVector[i] = np.sum(np.sqrt(abs(EMG[:, i]))) / np.size(EMG[:, i], 0)
    return msrVector


def WAMP(EMG, ch):
    if len(EMG) == 1:
        wampVector = np.zeros(ch)
    else:
        wampVector = np.zeros(ch)
        for i in range(ch):
            x_f = EMG[1:np.size(EMG, 0), i]
            x = EMG[0:np.size(EMG, 0) - 1, i]
            wampVector[i] = np.sum((np.sign(x - x_f) + 1) / 2)
    return wampVector


def RMS(EMG, ch):
    if len(EMG) == 1:
        rmsVector = np.zeros(ch)
    else:
        rmsVector = np.zeros(ch)
        for i in range(ch):
            rmsVector[i] = np.sqrt(np.sum((EMG[:, i]) ** 2) / np.size(EMG[:, i], 0))
    return rmsVector


def IAV(EMG, ch):
    if len(EMG) == 1:
        iavVector = np.zeros(ch)
    else:
        iavVector = np.zeros(ch)
        for i in range(ch):
            iavVector[i] = np.sum(abs(EMG[:, i]) / np.size(EMG[:, i], 0))
    return iavVector


def DASDV(EMG, ch):
    if len(EMG) == 1:
        dasdvVector = np.zeros(ch)
    else:
        dasdvVector = np.zeros(ch)
        for i in range(ch):
            x_f = EMG[1:np.size(EMG, 0), i]
            x = EMG[0:np.size(EMG, 0) - 1, i]
            dasdvVector[i] = np.sqrt(np.sum((x_f - x) ** 2) / np.size(x, 0))
    return dasdvVector


def VAR(EMG, ch):
    if len(EMG) == 1:
        varVector = np.zeros(ch)
    else:
        varVector = np.zeros(ch)
        for i in range(ch):
            varVector[i] = np.sum((EMG[:, i] - np.mean(EMG[:, i])) ** 2) / (np.size(EMG[:, i], 0) - 1)
    return varVector
