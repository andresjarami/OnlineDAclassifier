# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %% Functions
def uploadResultsDatabase(folder, database):
    if database == 'Nina5':
        samples = 3
        peoplei_i = 1
        peoplei_f = 10
        rows = 18 * 3
        times = 8
    elif database == 'Cote':
        samples = 3
        peoplei_i = 20
        peoplei_f = 36
        rows = 7 * 3
        times = 18
    elif database == 'EPN':
        samples = 24
        peoplei_i = 31
        peoplei_f = 60
        rows = 5 * 24
        times = 4
    place = folder + database
    shotStart = 1
    k = 1
    try:
        auxFrame = pd.read_csv(place + "_FeatureSet_1_startPerson_" + str(peoplei_i) + "_endPerson_" + str(peoplei_i)
                               + 'shotStart' + str(shotStart) + 'memmory' + str(k) + ".csv")
        resultsTest = auxFrame
    except:
        print('file not found')
        print(place + "_FeatureSet_1_startPerson_" + str(peoplei_i) + "_endPerson_" + str(peoplei_i)
              + 'shotStart' + str(shotStart) + 'memmory' + str(k) + ".csv")

        resultsTest = pd.DataFrame()
    # if len(resultsTest) != samples:
    #     print('error' + ' 1' + ' 1')
    #     print(len(resultsTest))

    for i in range(peoplei_i + 1, peoplei_f + 1):
        try:
            auxFrame = pd.read_csv(place + "_FeatureSet_1_startPerson_" + str(i) + "_endPerson_" + str(i)
                                   + 'shotStart' + str(shotStart) + 'memmory' + str(k) + ".csv")
            resultsTest = pd.concat([resultsTest, auxFrame], ignore_index=True)

        except:
            print('file not found')
            print(place + "_FeatureSet_1_startPerson_" + str(i) + "_endPerson_" + str(i)
                  + 'shotStart' + str(shotStart) + 'memmory' + str(k) + ".csv")
        # if len(auxFrame) != samples:
        #     print('error' + ' 1 ' + str(i))
        #     print(len(auxFrame))
    # for j in range(2, 4):
    #     for i in range(peoplei_i, peoplei_f + 1):
    #         auxFrame = pd.read_csv(
    #             place + "_FeatureSet_" + str(j) + "_startPerson_" + str(i) + "_endPerson_" + str(i) + ".csv")
    #         resultsTest = pd.concat([resultsTest, auxFrame], ignore_index=True)
    #
    #         if len(auxFrame) != samples:
    #             print('error' + ' ' + str(j) + ' ' + str(i))
    #             print(len(auxFrame))

    return resultsTest.drop(columns='Unnamed: 0'), rows


def vectors_calculation(results, rows):
    LDA_NoAdaptive = []
    LDA_Baseline = []
    LDA_PostProb_MSDA = []
    LDA_PostProb = []
    LDA_MSDA = []
    LDA_PostProb_MSDA_KL = []
    LDA_MSDA_KL = []

    QDA_NoAdaptive = []
    QDA_Baseline = []
    QDA_PostProb_MSDA = []
    QDA_PostProb = []
    QDA_MSDA = []
    QDA_PostProb_MSDA_KL = []
    QDA_MSDA_KL = []

    for i in range(1, rows):
        LDA_NoAdaptive.append(results['AccLDAfew'].loc[results['# shots'] == i].mean())
        LDA_Baseline.append(results['LDA_ACC_Baseline'].loc[results['# shots'] == i].mean())
        # LDA_PostProb_MSDA.append(results['LDA_ACC_PostProb_MSDA'].loc[results['# shots'] == i].mean())
        LDA_PostProb.append(results['LDA_ACC_PostProb'].loc[results['# shots'] == i].mean())
        # LDA_MSDA.append(results['LDA_ACC_MSDA'].loc[results['# shots'] == i].mean())
        LDA_PostProb_MSDA_KL.append(results['LDA_ACC_PostProb_MSDA_KL'].loc[results['# shots'] == i].mean())
        # LDA_MSDA_KL.append(results['LDA_ACC_MSDA_KL'].loc[results['# shots'] == i].mean())

        QDA_NoAdaptive.append(results['AccQDAfew'].loc[results['# shots'] == i].mean())
        QDA_Baseline.append(results['QDA_ACC_Baseline'].loc[results['# shots'] == i].mean())
        # QDA_PostProb_MSDA.append(results['QDA_ACC_PostProb_MSDA'].loc[results['# shots'] == i].mean())
        QDA_PostProb.append(results['QDA_ACC_PostProb'].loc[results['# shots'] == i].mean())
        # QDA_MSDA.append(results['QDA_ACC_MSDA'].loc[results['# shots'] == i].mean())
        QDA_PostProb_MSDA_KL.append(results['QDA_ACC_PostProb_MSDA_KL'].loc[results['# shots'] == i].mean())
        # QDA_MSDA_KL.append(results['QDA_ACC_MSDA_KL'].loc[results['# shots'] == i].mean())
    return LDA_NoAdaptive, LDA_Baseline, LDA_PostProb_MSDA, LDA_PostProb, LDA_MSDA, LDA_PostProb_MSDA_KL, \
           LDA_MSDA_KL, QDA_NoAdaptive, QDA_Baseline, QDA_PostProb_MSDA, QDA_PostProb, QDA_MSDA, QDA_PostProb_MSDA_KL, \
           QDA_MSDA_KL


def graphs(rows, title, LDA_NoAdaptive, LDA_Baseline, LDA_PostProb_MSDA, LDA_PostProb, LDA_MSDA, LDA_PostProb_MSDA_KL,
           LDA_MSDA_KL, QDA_NoAdaptive, QDA_Baseline, QDA_PostProb_MSDA, QDA_PostProb, QDA_MSDA, QDA_PostProb_MSDA_KL,
           QDA_MSDA_KL):
    x = [*range(1, rows)]
    plt.plot(x, LDA_NoAdaptive, label='LDA_NoAdaptive')
    plt.plot(x, LDA_Baseline, label='LDA_Optimal')
    # plt.plot(x, LDA_PostProb_MSDA, label='LDA_PostProb_MSDA')
    plt.plot(x, LDA_PostProb, label='LDA_PostProb')
    # plt.plot(x, LDA_MSDA, label='LDA_MSDA')
    plt.plot(x, LDA_PostProb_MSDA_KL, label='LDA_PostProb_MSDA_KL')
    # plt.plot(x, LDA_MSDA_KL, label='LDA_MSDA_KL')
    plt.xlabel('shots')
    plt.ylabel('Acc')
    plt.title(title)
    plt.legend()
    # plt.ylim(0.5, 1)
    plt.show()

    x = [*range(1, rows)]
    plt.plot(x, QDA_NoAdaptive, label='QDA_NoAdaptive')
    plt.plot(x, QDA_Baseline, label='QDA_Optimal')
    # plt.plot(x, QDA_PostProb_MSDA, label='QDA_PostProb_MSDA')
    plt.plot(x, QDA_PostProb, label='QDA_PostProb')
    # plt.plot(x, QDA_MSDA, label='QDA_MSDA')
    plt.plot(x, QDA_PostProb_MSDA_KL, label='QDA_PostProb_MSDA_KL')
    # plt.plot(x, QDA_MSDA_KL, label='QDA_MSDA_KL')
    plt.xlabel('shots')
    plt.ylabel('Acc')
    plt.title(title)
    plt.legend()
    # plt.ylim(0.5, 1)
    plt.show()


def analysis(folder, database):
    results, rows = uploadResultsDatabase(folder, database)
    LDA_NoAdaptive, LDA_Baseline, LDA_PostProb_MSDA, LDA_PostProb, LDA_MSDA, LDA_PostProb_MSDA_KL, \
    LDA_MSDA_KL, QDA_NoAdaptive, QDA_Baseline, QDA_PostProb_MSDA, QDA_PostProb, QDA_MSDA, QDA_PostProb_MSDA_KL, \
    QDA_MSDA_KL = vectors_calculation(results, rows)
    graphs(rows, database, LDA_NoAdaptive, LDA_Baseline, LDA_PostProb_MSDA, LDA_PostProb, LDA_MSDA,
           LDA_PostProb_MSDA_KL,
           LDA_MSDA_KL, QDA_NoAdaptive, QDA_Baseline, QDA_PostProb_MSDA, QDA_PostProb, QDA_MSDA, QDA_PostProb_MSDA_KL,
           QDA_MSDA_KL)


# %% Analysis
# place = 'resultsPBS/'

place = 'resultsPBS_2/Cote_'
database = 'Cote'
analysis(place, database)

place = 'resultsPBS_2/EPN_'
database = 'EPN'
analysis(place, database)

place = 'resultsPBS_2/Nina5_'
database = 'Nina5'
analysis(place, database)

# # Nina Pro 5 database
# database='NinaPro5'
# resultsNina5=uploadResultsDatabase(place, database)
#
# # EPN database
# database='EPN'
# resultsEPN=uploadResultsDatabase(place, database)


# %%
