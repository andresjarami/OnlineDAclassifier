# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# %% Upload results of the three databases
def uploadResults(place, samples, people, windowSize):
    resultsTest = pd.read_csv(place + "_FeatureSet_1_startPerson_" + str(1) + "_endPerson_" + str(
        people) + '_windowSize_' + windowSize + ".csv")
    if len(resultsTest) != samples * people:
        print('error' + ' 1')
        print(len(resultsTest))
    for j in range(2, 4):
        auxFrame = pd.read_csv(
            place + "_FeatureSet_" + str(j) + "_startPerson_" + str(1) + "_endPerson_" + str(
                people) + '_windowSize_' + windowSize + ".csv")
        resultsTest = pd.concat([resultsTest, auxFrame], ignore_index=True)
        if len(auxFrame) != samples * people:
            print('error' + ' ' + str(j))
            print(len(auxFrame))
    return resultsTest.drop(columns='Unnamed: 0')


def uploadResultsDatabasesVF1(folder, database, windowSize):
    if database == 'Nina5':
        samples = 4
        people = 10
        shots = 4
        x = [a for a in range(0, 18 * samples - 1, 18)]
    elif database == 'Cote':
        samples = 4
        people = 17
        shots = 4
        x = [a for a in range(0, 7 * samples - 1, 7)]
    elif database == 'EPN':
        samples = 25
        people = 30
        shots = 25
        x = [a for a in range(0, 5 * samples - 1, 5)]

    place = folder + database

    return analysisResults(uploadResults(place, samples, people, windowSize), shots), x


def analysisResults(resultDatabase, shots):
    results = pd.DataFrame(columns=['Feature Set', '# shots'])
    timeOurTechnique = pd.DataFrame(columns=[])

    idx = 0
    for j in range(1, 4):
        for i in range(1, shots + 1):
            subset = str(tuple(range(1, i + 1)))
            results.at[idx, 'Feature Set'] = j
            results.at[idx, '# shots'] = i

            # the accuracy for all LDA and QDA approaches
            IndLDA = resultDatabase['AccLDAInd'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            IndQDA = resultDatabase['AccQDAInd'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            MultiLDA = resultDatabase['AccLDAMulti'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            MultiQDA = resultDatabase['AccQDAMulti'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            LiuLDA = resultDatabase['AccLDALiu'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            LiuQDA = resultDatabase['AccQDALiu'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            VidLDA = resultDatabase['AccLDAVidovic'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            VidQDA = resultDatabase['AccQDAVidovic'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            OurLDA = resultDatabase['AccLDAProp'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            OurQDA = resultDatabase['AccQDAProp'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            results.at[idx, 'IndLDA'] = IndLDA.mean(axis=0)
            results.at[idx, 'IndLDAstd'] = IndLDA.std(axis=0)
            results.at[idx, 'IndQDA'] = IndQDA.mean(axis=0)
            results.at[idx, 'IndQDAstd'] = IndQDA.std(axis=0)
            results.at[idx, 'MultiLDA'] = MultiLDA.mean(axis=0)
            results.at[idx, 'MultiLDAstd'] = MultiLDA.std(axis=0)
            results.at[idx, 'MultiQDA'] = MultiQDA.mean(axis=0)
            results.at[idx, 'MultiQDAstd'] = MultiQDA.std(axis=0)
            results.at[idx, 'LiuLDA'] = LiuLDA.mean(axis=0)
            results.at[idx, 'LiuLDAstd'] = LiuLDA.std(axis=0)
            results.at[idx, 'LiuQDA'] = LiuQDA.mean(axis=0)
            results.at[idx, 'LiuQDAstd'] = LiuQDA.std(axis=0)
            results.at[idx, 'VidLDA'] = VidLDA.mean(axis=0)
            results.at[idx, 'VidLDAstd'] = VidLDA.std(axis=0)
            results.at[idx, 'VidQDA'] = VidQDA.mean(axis=0)
            results.at[idx, 'VidQDAstd'] = VidQDA.std(axis=0)
            results.at[idx, 'OurLDA'] = OurLDA.mean(axis=0)
            results.at[idx, 'OurLDAstd'] = OurLDA.std(axis=0)
            results.at[idx, 'OurQDA'] = OurQDA.mean(axis=0)
            results.at[idx, 'OurQDAstd'] = OurQDA.std(axis=0)

            # the weights λ and w for our LDA and QDA adaptive classifiers
            wLDA = resultDatabase['wTargetMeanLDA'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            lLDA = resultDatabase['wTargetCovLDA'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            wQDA = resultDatabase['wTargetMeanQDA'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            lQDA = resultDatabase['wTargetCovQDA'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            results.at[idx, 'wLDA'] = wLDA.mean(axis=0)
            results.at[idx, 'lLDA'] = lLDA.mean(axis=0)
            results.at[idx, 'wQDA'] = wQDA.mean(axis=0)
            results.at[idx, 'lQDA'] = lQDA.mean(axis=0)

            idx += 1

        ### Times of the adaptive classifier
        # the time of the adaptation plus the training of our adaptive classifier both LDA and QDA
        ourLDAtrainTime = resultDatabase['tPropLDA'].loc[(resultDatabase['Feature Set'] == j)]
        ourQDAtrainTime = resultDatabase['tPropQDA'].loc[(resultDatabase['Feature Set'] == j)]
        timeOurTechnique.at[j, 'meanTrainLDA'] = round(ourLDAtrainTime.mean(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'stdTrainLDA'] = round(ourLDAtrainTime.std(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'varTrainLDA'] = round(ourLDAtrainTime.var(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'meanTrainQDA'] = round(ourQDAtrainTime.mean(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'stdTrainQDA'] = round(ourQDAtrainTime.std(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'varTrainQDA'] = round(ourQDAtrainTime.var(axis=0) * 1000, 2)

        # the time of the classification and the preprocessing (min-maxd normalization) of our adaptive classifier both LDA and QDA
        ourLDAclassifyTime = resultDatabase['tCLPropL'].loc[(resultDatabase['Feature Set'] == j)]
        ourQDAclassifyTime = resultDatabase['tCLPropQ'].loc[(resultDatabase['Feature Set'] == j)]
        ourNormTime = resultDatabase['tPre'].loc[(resultDatabase['Feature Set'] == j)]
        # means, standar deviation, and variance
        timeOurTechnique.at[j, 'meanClLDA'] = round(ourLDAclassifyTime.mean(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'stdClLDA'] = round(ourLDAclassifyTime.std(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'varClLDA'] = round(ourLDAclassifyTime.var(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'meanClQDA'] = round(ourQDAclassifyTime.mean(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'stdClQDA'] = round(ourQDAclassifyTime.std(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'varClQDA'] = round(ourQDAclassifyTime.var(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'meanNorm'] = round(ourNormTime.mean(axis=0) * 1000000, 2)
        timeOurTechnique.at[j, 'stdNorm'] = round(ourNormTime.std(axis=0) * 1000000, 2)
        timeOurTechnique.at[j, 'varNorm'] = round(ourNormTime.var(axis=0) * 1000000, 2)

    return results


# %% ******************************************************************************************************

# %% Functions
def uploadResultsDatabase(folder, database):
    if database == 'Nina5':
        samples = 3
        peoplei_i = 1
        peoplei_f = 10
        rows = 18 * samples
        times = 8
    elif database == 'Cote':
        samples = 3
        peoplei_i = 20
        peoplei_f = 36
        rows = 7 * samples
        times = 18
    elif database == 'EPN':
        samples = 24
        peoplei_i = 31
        peoplei_f = 60
        rows = 5 * samples
        times = 4
    place = folder + database
    shotStart = 1
    k = 1
    try:
        auxFrame = pd.read_csv(place + "_FeatureSet_1_startPerson_" + str(peoplei_i) + "_endPerson_" + str(peoplei_i)
                               + 'shotStart' + str(shotStart) + 'memmory' + str(k) + ".csv")
        resultsTest = auxFrame[:times * rows]
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
            resultsTest = pd.concat([resultsTest, auxFrame[:times * rows]], ignore_index=True)

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

    for i in range(rows + 1):
        if i == 0:
            LDA_NoAdaptive.append(results['AccLDAfew'].loc[results['# shots'] == 1].mean() * 100)
            LDA_Baseline.append(results['AccLDAfew'].loc[results['# shots'] == 1].mean() * 100)
            # LDA_PostProb_MSDA.append(results['AccLDAfew'].loc[results['# shots'] == 1].mean() * 100)
            LDA_PostProb.append(results['AccLDAfew'].loc[results['# shots'] == 1].mean() * 100)
            # LDA_MSDA.append(results['AccLDAfew'].loc[results['# shots'] == 1].mean() * 100)
            LDA_PostProb_MSDA_KL.append(results['AccLDAfew'].loc[results['# shots'] == 1].mean() * 100)
            # LDA_MSDA_KL.append(results['AccLDAfew'].loc[results['# shots'] == 1].mean() * 100)

            QDA_NoAdaptive.append(results['AccQDAfew'].loc[results['# shots'] == 1].mean() * 100)
            QDA_Baseline.append(results['AccQDAfew'].loc[results['# shots'] == 1].mean() * 100)
            # QDA_PostProb_MSDA.append(results['AccQDAfew'].loc[results['# shots'] == 1].mean() * 100)
            QDA_PostProb.append(results['AccQDAfew'].loc[results['# shots'] == 1].mean() * 100)
            # QDA_MSDA.append(results['AccQDAfew'].loc[results['# shots'] == 1].mean() * 100)
            QDA_PostProb_MSDA_KL.append(results['AccQDAfew'].loc[results['# shots'] == 1].mean() * 100)
            # QDA_MSDA_KL.append(results['AccQDAfew'].loc[results['# shots'] == 1].mean() * 100)
        else:
            LDA_NoAdaptive.append(results['AccLDAfew'].loc[results['# shots'] == i].mean() * 100)
            LDA_Baseline.append(results['LDA_ACC_Baseline'].loc[results['# shots'] == i].mean() * 100)
            # LDA_PostProb_MSDA.append(results['LDA_ACC_PostProb_MSDA'].loc[results['# shots'] == i].mean()*100)
            LDA_PostProb.append(results['LDA_ACC_PostProb'].loc[results['# shots'] == i].mean() * 100)
            # LDA_MSDA.append(results['LDA_ACC_MSDA'].loc[results['# shots'] == i].mean()*100)
            LDA_PostProb_MSDA_KL.append(results['LDA_ACC_PostProb_MSDA_KL'].loc[results['# shots'] == i].mean() * 100)
            # LDA_MSDA_KL.append(results['LDA_ACC_MSDA_KL'].loc[results['# shots'] == i].mean()*100)

            QDA_NoAdaptive.append(results['AccQDAfew'].loc[results['# shots'] == i].mean() * 100)
            QDA_Baseline.append(results['QDA_ACC_Baseline'].loc[results['# shots'] == i].mean() * 100)
            # QDA_PostProb_MSDA.append(results['QDA_ACC_PostProb_MSDA'].loc[results['# shots'] == i].mean()*100)
            QDA_PostProb.append(results['QDA_ACC_PostProb'].loc[results['# shots'] == i].mean() * 100)
            # QDA_MSDA.append(results['QDA_ACC_MSDA'].loc[results['# shots'] == i].mean()*100)
            QDA_PostProb_MSDA_KL.append(results['QDA_ACC_PostProb_MSDA_KL'].loc[results['# shots'] == i].mean() * 100)
            # QDA_MSDA_KL.append(results['QDA_ACC_MSDA_KL'].loc[results['# shots'] == i].mean()*100)
    return LDA_NoAdaptive, LDA_Baseline, LDA_PostProb_MSDA, LDA_PostProb, LDA_MSDA, LDA_PostProb_MSDA_KL, \
           LDA_MSDA_KL, QDA_NoAdaptive, QDA_Baseline, QDA_PostProb_MSDA, QDA_PostProb, QDA_MSDA, QDA_PostProb_MSDA_KL, \
           QDA_MSDA_KL, results['time_LDA_ACC_PostProb_MSDA_KL'].mean(), results[
               'time_QDA_ACC_PostProb_MSDA_KL'].mean(), \
           results['time_LDA_ACC_PostProb_MSDA_KL'].std(), results['time_QDA_ACC_PostProb_MSDA_KL'].std()


def graphs(rows, title, LDA_NoAdaptive, LDA_Baseline, LDA_PostProb_MSDA, LDA_PostProb, LDA_MSDA, LDA_PostProb_MSDA_KL,
           LDA_MSDA_KL, QDA_NoAdaptive, QDA_Baseline, QDA_PostProb_MSDA, QDA_PostProb, QDA_MSDA, QDA_PostProb_MSDA_KL,
           QDA_MSDA_KL, x_Old, yLDA, yQDA, yLDA_L, yQDA_L, yLDA_V, yQDA_V, yLDA_O, yQDA_O):
    x = [*range(rows + 1)]
    plt.plot(x, LDA_NoAdaptive, label='Baseline', color='tab:orange')
    plt.plot(x_Old, yLDA_L, label='Liu', color='tab:green')
    plt.plot(x_Old, yLDA_V, label='Vidovic', color='tab:red')
    plt.plot(x_Old, yLDA_O, label='Our_RQ1', color='tab:blue')

    # plt.plot(x, LDA_Baseline, label='LDA_Optimal')
    # plt.plot(x, LDA_PostProb_MSDA, label='LDA_PostProb_MSDA')
    # plt.plot(x, LDA_PostProb, label='LDA_PostProb')
    # plt.plot(x, LDA_MSDA, label='LDA_MSDA')
    plt.plot(x, LDA_PostProb_MSDA_KL, label='Our_RQ2', color='tab:purple')
    # plt.plot(x, LDA_MSDA_KL, label='LDA_MSDA_KL')

    # plt.plot(x, LDA_PostProb_MSDA_KL, label='LDA_Propose')

    plt.plot(x_Old, yLDA, label='Ideal', color='black', linestyle='--')

    plt.xlabel('number of unlabeled gestures\n (over time)')
    plt.ylabel('accuracy [%]')
    plt.title(title + ' (LDA)')
    # plt.legend(loc='lower center', bbox_to_anchor=(2, -0.7), ncol=1)
    # plt.ylim(0.5, 1)
    plt.grid(color='gainsboro', linewidth=1)
    plt.show()

    x = [*range(rows + 1)]
    plt.plot(x, QDA_NoAdaptive, label='Baseline', color='tab:orange')
    plt.plot(x_Old, yQDA_L, label='Liu', color='tab:green')
    plt.plot(x_Old, yQDA_V, label='Vidovic', color='tab:red')
    plt.plot(x_Old, yQDA_O, label='Our_RQ1', color='tab:blue')
    # plt.plot(x, QDA_Baseline, label='QDA_Optimal')
    # plt.plot(x, QDA_PostProb_MSDA, label='QDA_PostProb_MSDA')
    # plt.plot(x, QDA_PostProb, label='QDA_PostProb')
    # plt.plot(x, QDA_MSDA, label='QDA_MSDA')
    if len(x) > 100:
        x = [0] + [*range(10, rows + 1)]
        z = np.array(QDA_PostProb_MSDA_KL)
        plt.plot(x, z[x], label='Our_RQ2', color='tab:purple')
    else:
        plt.plot(x, QDA_PostProb_MSDA_KL, label='Our_RQ2', color='tab:purple')
    # plt.plot(x, QDA_MSDA_KL, label='QDA_MSDA_KL')

    # plt.plot(x, QDA_PostProb_MSDA_KL, label='QDA_Propose')

    plt.plot(x_Old, yQDA, label='Ideal', color='black', linestyle='--')

    plt.xlabel('number of unlabeled gestures\n (over time)')
    plt.ylabel('accuracy [%]')
    plt.title(title + ' (QDA)')
    # plt.legend(loc='lower center', bbox_to_anchor=(1.5, -1), ncol=1)
    # plt.ylim(0.5, 1)
    plt.grid(color='gainsboro', linewidth=1)
    plt.show()


def analysis(folder, database):
    results, rows = uploadResultsDatabase(folder, database)
    results_Old, x_Old = uploadResultsDatabasesVF1('../ResultsExp1_RQ1/', database, windowSize='295')
    LDA_NoAdaptive, LDA_Baseline, LDA_PostProb_MSDA, LDA_PostProb, LDA_MSDA, LDA_PostProb_MSDA_KL, \
    LDA_MSDA_KL, QDA_NoAdaptive, QDA_Baseline, QDA_PostProb_MSDA, QDA_PostProb, QDA_MSDA, QDA_PostProb_MSDA_KL, \
    QDA_MSDA_KL, time_LDA, time_QDA, time_LDA_std, time_QDA_std = vectors_calculation(results, rows)
    print(database)
    print('timeLDA', time_LDA)
    print('std', time_LDA_std)
    print('timeQDA', time_QDA)
    print('std', time_QDA_std)
    yLDA, yQDA, yLDA_L, yQDA_L, yLDA_V, yQDA_V, yLDA_O, yQDA_O = vectors_calculation_Old(results_Old)
    graphs(rows, database, LDA_NoAdaptive, LDA_Baseline, LDA_PostProb_MSDA, LDA_PostProb, LDA_MSDA,
           LDA_PostProb_MSDA_KL, LDA_MSDA_KL, QDA_NoAdaptive, QDA_Baseline, QDA_PostProb_MSDA, QDA_PostProb, QDA_MSDA,
           QDA_PostProb_MSDA_KL, QDA_MSDA_KL, x_Old, yLDA, yQDA, yLDA_L, yQDA_L, yLDA_V, yQDA_V, yLDA_O, yQDA_O)


# shots
def vectors_calculation_Old(results_Old):
    yLDA = np.array(results_Old['IndLDA'].loc[results_Old['Feature Set'] == 1]) * 100
    yQDA = np.array(results_Old['IndQDA'].loc[results_Old['Feature Set'] == 1]) * 100
    yLDA_L = []
    yQDA_L = []
    yLDA_V = []
    yQDA_V = []
    yLDA_O = []
    yQDA_O = []
    for i in range(len(yLDA)):
        yLDA_L.append(np.array(
            results_Old['LiuLDA'].loc[(results_Old['Feature Set'] == 1) & (results_Old['# shots'] == 1)]) * 100)
        yQDA_L.append(np.array(
            results_Old['LiuQDA'].loc[(results_Old['Feature Set'] == 1) & (results_Old['# shots'] == 1)]) * 100)
        yLDA_V.append(np.array(
            results_Old['VidLDA'].loc[(results_Old['Feature Set'] == 1) & (results_Old['# shots'] == 1)]) * 100)
        yQDA_V.append(np.array(
            results_Old['VidQDA'].loc[(results_Old['Feature Set'] == 1) & (results_Old['# shots'] == 1)]) * 100)
        yLDA_O.append(np.array(
            results_Old['OurLDA'].loc[(results_Old['Feature Set'] == 1) & (results_Old['# shots'] == 1)]) * 100)
        yQDA_O.append(np.array(
            results_Old['OurQDA'].loc[(results_Old['Feature Set'] == 1) & (results_Old['# shots'] == 1)]) * 100)
    return yLDA, yQDA, yLDA_L, yQDA_L, yLDA_V, yQDA_V, yLDA_O, yQDA_O


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
