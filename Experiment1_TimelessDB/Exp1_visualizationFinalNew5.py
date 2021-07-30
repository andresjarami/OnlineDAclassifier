# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FormatStrFormatter
from scipy import stats


# %% Upload results of the three databases
def uploadResults(folder, samples, people, windowSize):
    resultsTest = pd.read_csv(folder + "_FeatureSet_1_startPerson_" + str(1) + "_endPerson_" + str(
        people) + '_windowSize_' + windowSize + ".csv")
    if len(resultsTest) != samples * people:
        print('error' + ' 1')
        print(len(resultsTest))
    for j in range(2, 4):
        auxFrame = pd.read_csv(
            folder + "_FeatureSet_" + str(j) + "_startPerson_" + str(1) + "_endPerson_" + str(
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

    return analysisResults(uploadResults(folder + database, samples, people, windowSize), shots), x


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
# def AUXuploadResultsDatabase(folder, database, samplesInMemory, featureSet, shotStart):
#     times = 1
#     if database == 'Nina5':
#         repetitions = 4
#         peoplei_i = 1
#         peoplei_f = 10
#         classes = 18
#         rows = classes * (repetitions - 2)
#     elif database == 'Cote':
#         repetitions = 4
#         peoplei_i = 20
#         peoplei_f = 36
#         classes = 7
#         rows = classes * (repetitions - 2)
#     elif database == 'EPN':
#         repetitions = 25
#         peoplei_i = 31
#         peoplei_f = 60
#         classes = 5
#         rows = classes * (repetitions - 2)
#
#     resultsTest = pd.DataFrame()
#
#     for expTime in range(1, times + 1):
#         for i in range(peoplei_i, peoplei_f + 1):
#             try:
#                 auxFrame = pd.read_csv(folder + database + '_FeatureSet_' + str(featureSet) + '_startPerson_' + str(
#                     i) + '_endPerson_' + str(i) + '_shotStart_' + str(shotStart) + '_memmory_' + str(
#                     samplesInMemory) + '_initialExpTime_' + str(expTime) + '_finalExpTime_' + str(expTime) + '.csv')
#
#                 resultsTest = pd.concat([resultsTest, auxFrame[:rows]], ignore_index=True)
#                 if len(auxFrame[:rows]) != rows:
#                     print('error_person' + ' ' + str(i) + ' correct: ' + str(rows))
#                     print('current error: ' + len(auxFrame))
#
#             except:
#                 print('file not found')
#                 print(folder + database + '_FeatureSet_' + str(featureSet) + '_startPerson_' + str(
#                     i) + '_endPerson_' + str(i) + '_shotStart_' + str(shotStart) + '_memmory_' + str(
#                     samplesInMemory) + '_initialExpTime_' + str(expTime) + '_finalExpTime_' + str(expTime) + '.csv')
#
#     return resultsTest.drop(columns='Unnamed: 0'), rows
#
#
# def uploadResultsDatabase(folder, database, samplesInMemory, featureSet, times, shotStart):
#     if database == 'Nina5':
#         repetitions = 4
#         peoplei_i = 1
#         peoplei_f = 10
#         classes = 18
#         rows = classes * (repetitions - shotStart)
#     elif database == 'Cote':
#         repetitions = 4
#         peoplei_i = 20
#         peoplei_f = 36
#         classes = 7
#         rows = classes * (repetitions - shotStart)
#     elif database == 'EPN':
#         repetitions = 25
#         peoplei_i = 31
#         peoplei_f = 60
#         classes = 5
#         rows = classes * (repetitions - shotStart)
#
#     resultsTest = pd.DataFrame()
#
#     for expTime in range(1, times + 1):
#         for i in range(peoplei_i, peoplei_f + 1):
#             try:
#                 auxFrame = pd.read_csv(folder + database + '_FeatureSet_' + str(featureSet) + '_startPerson_' + str(
#                     i) + '_endPerson_' + str(i) + '_shotStart_' + str(shotStart) + '_memmory_' + str(
#                     samplesInMemory) + '_initialExpTime_' + str(expTime) + '_finalExpTime_' + str(expTime) + '.csv')
#
#                 resultsTest = pd.concat([resultsTest, auxFrame], ignore_index=True)
#                 if len(auxFrame) != rows:
#                     print('error_person' + ' ' + str(i) + ' correct: ' + str(rows))
#                     print('current error: ' + len(auxFrame))
#
#             except:
#                 print('file not found')
#                 print(folder + database + '_FeatureSet_' + str(featureSet) + '_startPerson_' + str(
#                     i) + '_endPerson_' + str(i) + '_shotStart_' + str(shotStart) + '_memmory_' + str(
#                     samplesInMemory) + '_initialExpTime_' + str(expTime) + '_finalExpTime_' + str(expTime) + '.csv')
#
#     return resultsTest.drop(columns='Unnamed: 0'), rows
#

def uploadResultsDatabase(folder, database, featureSet, times, shotStart):
    if database == 'Nina5':
        repetitions = 4
        peoplei_i = 1
        peoplei_f = 10
        classes = 18
        rows = classes * (repetitions - shotStart)
    elif database == 'Cote':
        repetitions = 4
        peoplei_i = 20
        peoplei_f = 36
        classes = 7
        rows = classes * (repetitions - shotStart)
    elif database == 'EPN':
        repetitions = 25
        peoplei_i = 31
        peoplei_f = 60
        classes = 5
        rows = classes * (repetitions - shotStart)

    resultsTest = pd.DataFrame()

    for expTime in range(1, times + 1):
        for i in range(peoplei_i, peoplei_f + 1):
            try:
                auxFrame = pd.read_csv(folder + '_' + database + '_FS_' + str(featureSet) + '_sP_' + str(
                    i) + '_eP_' + str(i) + '_sStart_' + str(shotStart) + '_inTime_' + str(
                    expTime) + '_fiTime_' + str(expTime) + '.csv')

                resultsTest = pd.concat([resultsTest, auxFrame], ignore_index=True)
                if len(auxFrame) != rows:
                    print('exp. Time' + ' ' + str(expTime) + 'error_person' + ' ' + str(i) + ' correct: ' + str(rows))
                    print('current error: ' + str(len(auxFrame)))

            except:
                print('file not found')
                print(folder + '_' + database + '_FS_' + str(featureSet) + '_sP_' + str(
                    i) + '_eP_' + str(i) + '_sStart_' + str(shotStart) + '_inTime_' + str(
                    expTime) + '_fiTime_' + str(expTime) + '.csv')

    return resultsTest.drop(columns='Unnamed: 0')


def vectors_calculation(results, rows):
    LDA_Ideal = []
    LDA_NoAdapted = []
    LDA_PostProb = []
    LDA_Labels = []
    LDA_PostProb_MSDA = []
    LDA_Adapted = []
    LDA_PostProb_Adapted = []
    LDA_Labels_Adapted = []
    LDA_PostProb_MSDA_Adapted = []

    QDA_Ideal = []
    QDA_NoAdapted = []
    QDA_PostProb = []
    QDA_Labels = []
    QDA_PostProb_MSDA = []
    QDA_Adapted = []
    QDA_PostProb_Adapted = []
    QDA_Labels_Adapted = []
    QDA_PostProb_MSDA_Adapted = []

    for i in range(rows + 1):
        LDA_NoAdapted.append(results['AccLDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
        LDA_Adapted.append(results['AccLDA_Adapted'].loc[results['# shots'] == 1].mean() * 100)
        QDA_NoAdapted.append(results['AccQDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
        QDA_Adapted.append(results['AccQDA_Adapted'].loc[results['# shots'] == 1].mean() * 100)

        if i == 0:

            LDA_Ideal.append(results['AccLDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            LDA_PostProb.append(results['AccLDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            # LDA_Labels.append(results['AccLDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            LDA_PostProb_MSDA.append(results['AccLDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            LDA_PostProb_Adapted.append(results['AccLDA_Adapted'].loc[results['# shots'] == 1].mean() * 100)
            # LDA_Labels_Adapted.append(results['AccLDA_Adapted'].loc[results['# shots'] == 1].mean() * 100)
            LDA_PostProb_MSDA_Adapted.append(results['AccLDA_Adapted'].loc[results['# shots'] == 1].mean() * 100)

            QDA_Ideal.append(results['AccQDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            QDA_PostProb.append(results['AccQDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            # QDA_Labels.append(results['AccQDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            QDA_PostProb_MSDA.append(results['AccQDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            QDA_PostProb_Adapted.append(results['AccQDA_Adapted'].loc[results['# shots'] == 1].mean() * 100)
            # QDA_Labels_Adapted.append(results['AccQDA_Adapted'].loc[results['# shots'] == 1].mean() * 100)
            QDA_PostProb_MSDA_Adapted.append(results['AccQDA_Adapted'].loc[results['# shots'] == 1].mean() * 100)

        else:
            LDA_Ideal.append(results['AccLDA_Ideal'].loc[results['# shots'] == i].mean() * 100)
            LDA_PostProb.append(results['AccLDA_PostProb'].loc[results['# shots'] == i].mean() * 100)
            # LDA_Labels.append(results['AccLDA_Labels'].loc[results['# shots'] == i].mean() * 100)
            LDA_PostProb_MSDA.append(results['AccLDA_PostProb_MSDA'].loc[results['# shots'] == i].mean() * 100)
            LDA_PostProb_Adapted.append(results['AccLDA_PostProb_Adapted'].loc[results['# shots'] == i].mean() * 100)
            # LDA_Labels_Adapted.append(results['AccLDA_Labels_Adapted'].loc[results['# shots'] == i].mean() * 100)
            LDA_PostProb_MSDA_Adapted.append(
                results['AccLDA_PostProb_MSDA_Adapted'].loc[results['# shots'] == i].mean() * 100)

            QDA_Ideal.append(results['AccQDA_Ideal'].loc[results['# shots'] == i].mean() * 100)
            QDA_PostProb.append(results['AccQDA_PostProb'].loc[results['# shots'] == i].mean() * 100)
            # QDA_Labels.append(results['AccQDA_Labels'].loc[results['# shots'] == i].mean() * 100)
            QDA_PostProb_MSDA.append(results['AccQDA_PostProb_MSDA'].loc[results['# shots'] == i].mean() * 100)
            QDA_PostProb_Adapted.append(results['AccQDA_PostProb_Adapted'].loc[results['# shots'] == i].mean() * 100)
            # QDA_Labels_Adapted.append(results['AccLDA_Labels_Adapted'].loc[results['# shots'] == i].mean() * 100)
            QDA_PostProb_MSDA_Adapted.append(
                results['AccQDA_PostProb_MSDA_Adapted'].loc[results['# shots'] == i].mean() * 100)

    return LDA_Ideal, LDA_NoAdapted, LDA_PostProb, LDA_PostProb_MSDA, LDA_Adapted, LDA_PostProb_Adapted, \
           LDA_PostProb_MSDA_Adapted, QDA_Ideal, QDA_NoAdapted, QDA_PostProb, \
           QDA_PostProb_MSDA, QDA_Adapted, QDA_PostProb_Adapted, QDA_PostProb_MSDA_Adapted

    # results['time_LDA_ACC_PostProb_MSDA_JS'].mean(), results['time_QDA_ACC_PostProb_MSDA_JS'].mean(), \
    # results['time_LDA_ACC_PostProb_MSDA_JS'].std(), results['time_QDA_ACC_PostProb_MSDA_JS'].std(), \
    # results['time_LDA_ACC_PostProb_MSDA_JS_adapt'].mean(), results['time_QDA_ACC_PostProb_MSDA_JS_adapt'].mean(), \
    # results['time_LDA_ACC_PostProb_MSDA_JS_adapt'].std(), results['time_QDA_ACC_PostProb_MSDA_JS_adapt'].std()


def vectors_calculation2(results, rows):
    LDA_Ideal = []
    LDA_NoAdapted = []
    LDA_incre_gestures_labels = []
    LDA_incre_gestures_weight = []
    LDA_incre_gestures_weight_MSDA = []
    LDA_incre_samples_labels = []
    LDA_incre_samples_prob = []
    LDA_semi_gestures_labels = []
    LDA_semi_gestures_weight = []
    LDA_semi_gestures_weight_MSDA = []
    LDA_semi_samples_labels = []
    LDA_semi_samples_prob = []

    LDA_incre_gestures_weight_MSDA_Adapted = []

    QDA_Ideal = []
    QDA_NoAdapted = []
    QDA_incre_gestures_labels = []
    QDA_incre_gestures_weight = []
    QDA_incre_gestures_weight_MSDA = []
    QDA_incre_samples_labels = []
    QDA_incre_samples_prob = []
    QDA_semi_gestures_labels = []
    QDA_semi_gestures_weight = []
    QDA_semi_gestures_weight_MSDA = []
    QDA_semi_samples_labels = []
    QDA_semi_samples_prob = []

    QDA_incre_gestures_weight_MSDA_Adapted = []

    E1_LDA_incre_gestures_labels = []
    E1_LDA_incre_gestures_weight = []
    E1_LDA_incre_gestures_weight_MSDA = []
    E1_QDA_incre_gestures_labels = []
    E1_QDA_incre_gestures_weight = []
    E1_QDA_incre_gestures_weight_MSDA = []

    E2_LDA_incre_gestures_labels = []
    E2_LDA_incre_gestures_weight = []
    E2_LDA_incre_gestures_weight_MSDA = []
    E2_QDA_incre_gestures_labels = []
    E2_QDA_incre_gestures_weight = []
    E2_QDA_incre_gestures_weight_MSDA = []

    for i in range(rows + 1):
        LDA_NoAdapted.append(results['AccLDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
        QDA_NoAdapted.append(results['AccQDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)

        if i == 0:

            LDA_Ideal.append(results['AccLDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            LDA_incre_gestures_labels.append(results['AccLDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            LDA_incre_gestures_weight.append(results['AccLDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            LDA_incre_gestures_weight_MSDA.append(
                results['AccLDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            LDA_incre_samples_labels.append(results['AccLDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            LDA_incre_samples_prob.append(results['AccLDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            LDA_semi_gestures_labels.append(results['AccLDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            LDA_semi_gestures_weight.append(results['AccLDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            LDA_semi_gestures_weight_MSDA.append(results['AccLDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            LDA_semi_samples_labels.append(results['AccLDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            LDA_semi_samples_prob.append(results['AccLDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)

            LDA_incre_gestures_weight_MSDA_Adapted.append(
                results['AccLDA_Adapted'].loc[results['# shots'] == 1].mean() * 100)

            QDA_Ideal.append(results['AccQDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            QDA_incre_gestures_labels.append(results['AccQDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            QDA_incre_gestures_weight.append(results['AccQDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            QDA_incre_gestures_weight_MSDA.append(
                results['AccQDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            QDA_incre_samples_labels.append(results['AccQDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            QDA_incre_samples_prob.append(results['AccQDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            QDA_semi_gestures_labels.append(results['AccQDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            QDA_semi_gestures_weight.append(results['AccQDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            QDA_semi_gestures_weight_MSDA.append(results['AccQDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            QDA_semi_samples_labels.append(results['AccQDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)
            QDA_semi_samples_prob.append(results['AccQDA_NoAdapted'].loc[results['# shots'] == 1].mean() * 100)

            QDA_incre_gestures_weight_MSDA_Adapted.append(
                results['AccQDA_Adapted'].loc[results['# shots'] == 1].mean() * 100)

        else:
            LDA_Ideal.append(results['AccLDA_Ideal'].loc[results['# shots'] == i].mean() * 100)
            LDA_incre_gestures_labels.append(
                results['AccLDA_incre_gestures_labels'].loc[results['# shots'] == i].mean() * 100)
            LDA_incre_gestures_weight.append(
                results['AccLDA_incre_gestures_weight'].loc[results['# shots'] == i].mean() * 100)
            LDA_incre_gestures_weight_MSDA.append(
                results['AccLDA_incre_gestures_weight_MSDA'].loc[results['# shots'] == i].mean() * 100)
            LDA_incre_samples_labels.append(
                results['AccLDA_incre_samples_labels'].loc[results['# shots'] == i].mean() * 100)
            LDA_incre_samples_prob.append(
                results['AccLDA_incre_samples_prob'].loc[results['# shots'] == i].mean() * 100)
            LDA_semi_gestures_labels.append(
                results['AccLDA_semi_gestures_labels'].loc[results['# shots'] == i].mean() * 100)
            LDA_semi_gestures_weight.append(
                results['AccLDA_semi_gestures_weight'].loc[results['# shots'] == i].mean() * 100)
            LDA_semi_gestures_weight_MSDA.append(
                results['AccLDA_semi_gestures_weight_MSDA'].loc[results['# shots'] == i].mean() * 100)
            LDA_semi_samples_labels.append(
                results['AccLDA_semi_samples_labels'].loc[results['# shots'] == i].mean() * 100)
            LDA_semi_samples_prob.append(results['AccLDA_semi_samples_prob'].loc[results['# shots'] == i].mean() * 100)

            LDA_incre_gestures_weight_MSDA_Adapted.append(
                results['AccLDA_incre_gestures_weight_MSDA_Adapted'].loc[results['# shots'] == i].mean() * 100)

            QDA_Ideal.append(results['AccQDA_Ideal'].loc[results['# shots'] == i].mean() * 100)
            QDA_incre_gestures_labels.append(
                results['AccQDA_incre_gestures_labels'].loc[results['# shots'] == i].mean() * 100)
            QDA_incre_gestures_weight.append(
                results['AccQDA_incre_gestures_weight'].loc[results['# shots'] == i].mean() * 100)
            QDA_incre_gestures_weight_MSDA.append(
                results['AccQDA_incre_gestures_weight_MSDA'].loc[results['# shots'] == i].mean() * 100)
            QDA_incre_samples_labels.append(
                results['AccQDA_incre_samples_labels'].loc[results['# shots'] == i].mean() * 100)
            QDA_incre_samples_prob.append(
                results['AccQDA_incre_samples_prob'].loc[results['# shots'] == i].mean() * 100)
            QDA_semi_gestures_labels.append(
                results['AccQDA_semi_gestures_labels'].loc[results['# shots'] == i].mean() * 100)
            QDA_semi_gestures_weight.append(
                results['AccQDA_semi_gestures_weight'].loc[results['# shots'] == i].mean() * 100)
            QDA_semi_gestures_weight_MSDA.append(
                results['AccQDA_semi_gestures_weight_MSDA'].loc[results['# shots'] == i].mean() * 100)
            QDA_semi_samples_labels.append(
                results['AccQDA_semi_samples_labels'].loc[results['# shots'] == i].mean() * 100)
            QDA_semi_samples_prob.append(results['AccQDA_semi_samples_prob'].loc[results['# shots'] == i].mean() * 100)

            QDA_incre_gestures_weight_MSDA_Adapted.append(
                results['AccQDA_incre_gestures_weight_MSDA_Adapted'].loc[results['# shots'] == i].mean() * 100)

            ### error
            E1_LDA_incre_gestures_weight.append(
                results['1_ErrorLDA_incre_gestures_weight'].loc[results['# shots'] == i].mean())
            E1_LDA_incre_gestures_labels.append(
                results['1_ErrorLDA_incre_gestures_labels'].loc[results['# shots'] == i].mean())
            E1_LDA_incre_gestures_weight_MSDA.append(
                results['1_ErrorLDA_incre_gestures_weight_MSDA'].loc[results['# shots'] == i].mean())

            E1_QDA_incre_gestures_weight.append(
                results['1_ErrorQDA_incre_gestures_weight'].loc[results['# shots'] == i].mean())
            E1_QDA_incre_gestures_labels.append(
                results['1_ErrorQDA_incre_gestures_labels'].loc[results['# shots'] == i].mean())
            E1_QDA_incre_gestures_weight_MSDA.append(
                results['1_ErrorQDA_incre_gestures_weight_MSDA'].loc[results['# shots'] == i].mean())

            E2_LDA_incre_gestures_weight.append(
                results['1_ErrorLDA_incre_gestures_weight'].loc[results['# shots'] == i].mean())
            E2_LDA_incre_gestures_labels.append(
                results['1_ErrorLDA_incre_gestures_labels'].loc[results['# shots'] == i].mean())
            E2_LDA_incre_gestures_weight_MSDA.append(
                results['1_ErrorLDA_incre_gestures_weight_MSDA'].loc[results['# shots'] == i].mean())

            E2_QDA_incre_gestures_weight.append(
                results['1_ErrorQDA_incre_gestures_weight'].loc[results['# shots'] == i].mean())
            E2_QDA_incre_gestures_labels.append(
                results['1_ErrorQDA_incre_gestures_labels'].loc[results['# shots'] == i].mean())
            E2_QDA_incre_gestures_weight_MSDA.append(
                results['1_ErrorQDA_incre_gestures_weight_MSDA'].loc[results['# shots'] == i].mean())

    return LDA_Ideal, LDA_NoAdapted, LDA_incre_gestures_labels, LDA_incre_gestures_weight, \
           LDA_incre_gestures_weight_MSDA, LDA_incre_samples_labels, LDA_incre_samples_prob, LDA_semi_gestures_labels, \
           LDA_semi_gestures_weight, LDA_semi_gestures_weight_MSDA, LDA_semi_samples_labels, LDA_semi_samples_prob, \
           LDA_incre_gestures_weight_MSDA_Adapted, QDA_Ideal, QDA_NoAdapted, QDA_incre_gestures_labels, \
           QDA_incre_gestures_weight, QDA_incre_gestures_weight_MSDA, QDA_incre_samples_labels, \
           QDA_incre_samples_prob, QDA_semi_gestures_labels, QDA_semi_gestures_weight, QDA_semi_gestures_weight_MSDA, \
           QDA_semi_samples_labels, QDA_semi_samples_prob, QDA_incre_gestures_weight_MSDA_Adapted, \
           E1_LDA_incre_gestures_labels, E1_LDA_incre_gestures_weight, E1_LDA_incre_gestures_weight_MSDA, \
           E1_QDA_incre_gestures_labels, E1_QDA_incre_gestures_weight, E1_QDA_incre_gestures_weight_MSDA, \
           E2_LDA_incre_gestures_labels, E2_LDA_incre_gestures_weight, E2_LDA_incre_gestures_weight_MSDA, \
           E2_QDA_incre_gestures_labels, E2_QDA_incre_gestures_weight, E2_QDA_incre_gestures_weight_MSDA


# results['time_LDA_ACC_PostProb_MSDA_JS'].mean(), results['time_QDA_ACC_PostProb_MSDA_JS'].mean(), \
# results['time_LDA_ACC_PostProb_MSDA_JS'].std(), results['time_QDA_ACC_PostProb_MSDA_JS'].std(), \
# results['time_LDA_ACC_PostProb_MSDA_JS_adapt'].mean(), results['time_QDA_ACC_PostProb_MSDA_JS_adapt'].mean(), \
# results['time_LDA_ACC_PostProb_MSDA_JS_adapt'].std(), results['time_QDA_ACC_PostProb_MSDA_JS_adapt'].std()


def graphs(rows1, rows2, database, LDA_Ideal_2, LDA_NoAdapted_2, LDA_PostProb_2, LDA_PostProb_MSDA_2, LDA_Adapted_2,
           LDA_PostProb_Adapted_2, LDA_PostProb_MSDA_Adapted_2, QDA_Ideal_2, QDA_NoAdapted_2, QDA_PostProb_2,
           QDA_PostProb_MSDA_2, QDA_Adapted_2, QDA_PostProb_Adapted_2, QDA_PostProb_MSDA_Adapted_2, x_Old, yLDA, yQDA,
           yLDA_L, yQDA_L, yLDA_V, yQDA_V, yLDA_O, yQDA_O, samplesInMemory, featureSet, LDA_Ideal_1, LDA_NoAdapted_1,
           LDA_incre_gestures_labels_1, LDA_incre_gestures_weight_1, LDA_incre_gestures_weight_MSDA_1,
           LDA_incre_samples_labels_1, LDA_incre_samples_prob_1, LDA_semi_gestures_labels_1,
           LDA_semi_gestures_weight_1, LDA_semi_gestures_weight_MSDA_1, LDA_semi_samples_labels_1,
           LDA_semi_samples_prob_1, LDA_incre_gestures_weight_MSDA_Adapted_1, QDA_Ideal_1, QDA_NoAdapted_1,
           QDA_incre_gestures_labels_1, QDA_incre_gestures_weight_1, QDA_incre_gestures_weight_MSDA_1,
           QDA_incre_samples_labels_1, QDA_incre_samples_prob_1, QDA_semi_gestures_labels_1,
           QDA_semi_gestures_weight_1, QDA_semi_gestures_weight_MSDA_1, QDA_semi_samples_labels_1,
           QDA_semi_samples_prob_1, QDA_incre_gestures_weight_MSDA_Adapted_1, E1_LDA_incre_gestures_labels_1,
           E1_LDA_incre_gestures_weight_1, E1_LDA_incre_gestures_weight_MSDA_1, E1_QDA_incre_gestures_labels_1,
           E1_QDA_incre_gestures_weight_1, E1_QDA_incre_gestures_weight_MSDA_1, E2_LDA_incre_gestures_labels_1,
           E2_LDA_incre_gestures_weight_1, E2_LDA_incre_gestures_weight_MSDA_1, E2_QDA_incre_gestures_labels_1,
           E2_QDA_incre_gestures_weight_1, E2_QDA_incre_gestures_weight_MSDA_1):
    x1 = [*range(rows1 + 1)]
    x2 = [*range(rows2 + 1)]

    fig, ax = plt.subplots(nrows=2, ncols=2, sharey='row', sharex='col')
    # fig.suptitle('Accuracy vs. Unlabeled Gestures (over time)')

    ax[0, 0].plot(x1, LDA_Ideal_1, label='All labeled')
    ax[0, 0].plot(x1, LDA_NoAdapted_1, label='Baseline')
    ax[0, 0].plot(x1, LDA_incre_gestures_weight_1, label='incre_gestures_weights')
    # ax[0, 0].plot(x1, LDA_incre_gestures_labels_1, label='incre_gestures_labels')
    ax[0, 0].plot(x1, LDA_incre_gestures_weight_MSDA_1, label='incre_gestures_weights_MSDA')
    # ax[0, 0].plot(x1, LDA_incre_samples_labels_1, label='incre_samples_labels')
    # ax[0, 0].plot(x1, LDA_incre_samples_prob_1, label='incre_gestures_weights_MSDA')

    ax[0, 0].plot(x1, LDA_semi_gestures_weight_1, label='semi_gestures_weights')
    # ax[0, 0].plot(x1, LDA_semi_gestures_labels_1, label='semi_gestures_labels')
    ax[0, 0].plot(x1, LDA_semi_gestures_weight_MSDA_1, label='semi_gestures_weights_MSDA')
    # ax[0, 0].plot(x1, LDA_semi_samples_labels_1, label='semi_samples_labels')
    # ax[0, 0].plot(x1, LDA_semi_samples_prob_1, label='semi_gestures_weights_MSDA')

    # ax[0, 0].plot(x1, LDA_incre_gestures_weight_MSDA_Adapted_1, label='incre_gestures_weights_MSDA_Adapted')

    # ax[0, 0].plot(x_Old, yLDA, label='Ideal', linestyle='--')

    # ax[0, 0].set_xlabel('number of unlabeled gestures\n (over time)')
    ax[0, 0].set_ylabel('accuracy [%]')
    ax[0, 0].set_title(database + ' (LDA)' + ' labeled gestures' + str(1))
    ax[0, 0].grid(color='gainsboro', linewidth=1)

    ax[0, 1].plot(x2, LDA_Ideal_2, label='LDA Ideal (using labeles)')
    ax[0, 1].plot(x2, LDA_NoAdapted_2, label='LDA Baseline')
    ax[0, 1].plot(x2, LDA_PostProb_2, label='LDA using PostProb')
    ax[0, 1].plot(x2, LDA_PostProb_MSDA_2, label='LDA Proposed')
    # ax[0, 1].plot(x2, LDA_Adapted_2, label='LDA Baseline (MSDA)')
    # ax[0, 1].plot(x2, LDA_PostProb_Adapted_2, label='LDA using PostProb (MSDA)')
    ax[0, 1].plot(x2, LDA_PostProb_MSDA_Adapted_2, label='LDA Proposed (MSDA)')

    # ax[0, 1].set_xlabel('number of unlabeled gestures\n (over time)')
    # ax[0, 1].set_ylabel('accuracy [%]')
    ax[0, 1].set_title(database + ' (LDA)' + ' labeled gestures' + str(2))
    ax[0, 1].grid(color='gainsboro', linewidth=1)

    ax[1, 0].plot(x1, QDA_Ideal_1, label='All labeled')
    ax[1, 0].plot(x1, QDA_NoAdapted_1, label='Baseline')
    ax[1, 0].plot(x1, QDA_incre_gestures_weight_1, label='incre_gestures_weights')
    # ax[1, 0].plot(x1, QDA_incre_gestures_labels_1, label='incre_gestures_labels')
    ax[1, 0].plot(x1, QDA_incre_gestures_weight_MSDA_1, label='incre_gestures_weights_MSDA')
    # ax[1, 0].plot(x1, QDA_incre_samples_labels_1, label='incre_samples_labels')
    # ax[1, 0].plot(x1, QDA_incre_samples_prob_1, label='incre_samples_prob')

    ax[1, 0].plot(x1, QDA_semi_gestures_weight_1, label='semi_gestures_weights')
    # ax[1, 0].plot(x1, QDA_semi_gestures_labels_1, label='semi_gestures_labels')
    ax[1, 0].plot(x1, QDA_semi_gestures_weight_MSDA_1, label='semi_gestures_weights_MSDA')
    # ax[1, 0].plot(x1, QDA_semi_samples_labels_1, label='semi_samples_labels')
    # ax[1, 0].plot(x1, QDA_semi_samples_prob_1, label='semi_samples_prob')

    # ax[1, 0].plot(x1, QDA_incre_gestures_weight_MSDA_Adapted_1, label='incre_gestures_weights_MSDA_Adapted')

    ax[1, 0].legend()

    ax[1, 0].set_xlabel('number of unlabeled gestures\n (over time)')
    ax[1, 0].set_ylabel('accuracy [%]')
    ax[1, 0].set_title(database + ' (QDA)' + ' labeled gestures' + str(1))
    ax[1, 0].grid(color='gainsboro', linewidth=1)

    ax[1, 1].plot(x2, QDA_Ideal_2, label='QDA Ideal (using labeles)')
    ax[1, 1].plot(x2, QDA_NoAdapted_2, label='QDA Baseline')
    ax[1, 1].plot(x2, QDA_PostProb_2, label='QDA using PostProb')
    ax[1, 1].plot(x2, QDA_PostProb_MSDA_2, label='QDA Proposed')
    # ax[1, 1].plot(x2, QDA_Adapted_2, label='QDA Baseline (MSDA)')
    # ax[1, 1].plot(x2, QDA_PostProb_Adapted_2, label='QDA using PostProb (MSDA)')
    ax[1, 1].plot(x2, QDA_PostProb_MSDA_Adapted_2, label='QDA Proposed (MSDA)')

    ax[1, 1].set_xlabel('number of unlabeled gestures\n (over time)')
    # ax[1, 1].set_ylabel('accuracy [%]')
    ax[1, 1].set_title(database + ' (QDA)' + ' labeled gestures' + str(2))
    ax[1, 1].grid(color='gainsboro', linewidth=1)

    ax[1, 1].legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(2, -1.7), ncol=1)
    # plt.ylim(0.5, 1)

    plt.show()
    ##############################################################ERRORS
    x1 = [*range(rows1)]

    fig1, ax = plt.subplots(nrows=2, ncols=1, sharey='row', sharex='col')
    # fig.suptitle('Accuracy vs. Unlabeled Gestures (over time)')

    ax[0].plot(x1, E1_LDA_incre_gestures_weight_1, label='LDA_PostProb_e1')
    ax[0].plot(x1, E1_LDA_incre_gestures_weight_MSDA_1, label='LDA_PostProb_MSDA_e1')
    ax[0].plot(x1, E1_LDA_incre_gestures_labels_1, label='LDA_Labels_e1')

    ax[0].plot(x1, E2_LDA_incre_gestures_weight_1, label='LDA_PostProb_e2')
    ax[0].plot(x1, E2_LDA_incre_gestures_weight_MSDA_1, label='LDA_PostProb_MSDA_e2')
    ax[0].plot(x1, E2_LDA_incre_gestures_labels_1, label='LDA_Labels_e2')

    ax[0].set_ylabel('error')
    ax[0].set_title(database + ' (LDA)')
    ax[0].grid(color='gainsboro', linewidth=1)

    ax[1].plot(x1, E1_QDA_incre_gestures_weight_1, label='QDA_PostProb_e1')
    ax[1].plot(x1, E1_QDA_incre_gestures_weight_MSDA_1, label='QDA_PostProb_MSDA_e1')
    ax[1].plot(x1, E1_QDA_incre_gestures_labels_1, label='QDA_Labels_e1')

    ax[1].plot(x1, E2_QDA_incre_gestures_weight_1, label='QDA_PostProb_e2')
    ax[1].plot(x1, E2_QDA_incre_gestures_weight_MSDA_1, label='QDA_PostProb_MSDA_e2')
    ax[1].plot(x1, E2_QDA_incre_gestures_labels_1, label='QDA_Labels_e2')

    ax[1].set_ylabel('error')
    ax[1].set_title(database + ' (QDA)')
    ax[1].grid(color='gainsboro', linewidth=1)

    ax[1].legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(2, -1.7), ncol=1)
    # plt.ylim(0.5, 1)

    plt.show()


def vectorNigam(database, results, ax, typeDA='LDA', shotStart=1):
    if database == 'EPN':
        classes = 5
        rep = 25
        AllGestures = [0, 1, 5, 10, 20, 40, 60, 80, 100, 120, rep * classes - shotStart * classes]
        evaluatedGestures = [a for a in AllGestures if a <= rep * classes - shotStart * classes]
    elif database == 'Nina5':
        classes = 18
        rep = 4
        AllGestures = [0, 1, 5, 10, 20, 30, 40, 50, 54, rep * classes - shotStart * classes]
        evaluatedGestures = [a for a in AllGestures if a <= rep * classes - shotStart * classes]
    elif database == 'Cote':
        classes = 7
        rep = 4
        AllGestures = [0, 1, 6, 11, 16, 21, rep * classes - shotStart * classes]
        evaluatedGestures = [a for a in AllGestures if a <= rep * classes - shotStart * classes]

    nigam01 = []
    nigam02 = []
    nigam04 = []
    nigam06 = []
    nigam08 = []
    nigam10 = []
    weak = []
    for i in evaluatedGestures:
        if i != 0:
            nigam01.append(results[typeDA + '_incre_Nigam_0.1'].loc[results['unlabeled Gesture'] == i].mean())
            nigam02.append(results[typeDA + '_incre_Nigam_0.2'].loc[results['unlabeled Gesture'] == i].mean())
            nigam04.append(results[typeDA + '_incre_Nigam_0.4'].loc[results['unlabeled Gesture'] == i].mean())
            nigam06.append(results[typeDA + '_incre_Nigam_0.6'].loc[results['unlabeled Gesture'] == i].mean())
            nigam08.append(results[typeDA + '_incre_Nigam_0.8'].loc[results['unlabeled Gesture'] == i].mean())
            nigam10.append(results[typeDA + '_incre_Nigam_1.0'].loc[results['unlabeled Gesture'] == i].mean())
            weak.append(results[typeDA + '_weak'].loc[results['unlabeled Gesture'] == 1].mean())
        else:
            nigam01.append(results[typeDA + '_weak'].loc[results['unlabeled Gesture'] == 1].mean())
            nigam02.append(results[typeDA + '_weak'].loc[results['unlabeled Gesture'] == 1].mean())
            nigam04.append(results[typeDA + '_weak'].loc[results['unlabeled Gesture'] == 1].mean())
            nigam06.append(results[typeDA + '_weak'].loc[results['unlabeled Gesture'] == 1].mean())
            nigam08.append(results[typeDA + '_weak'].loc[results['unlabeled Gesture'] == 1].mean())
            nigam10.append(results[typeDA + '_weak'].loc[results['unlabeled Gesture'] == 1].mean())
            weak.append(results[typeDA + '_weak'].loc[results['unlabeled Gesture'] == 1].mean())

    ax.plot(evaluatedGestures, nigam01, label='nigam01')
    ax.plot(evaluatedGestures, nigam02, label='nigam02')
    ax.plot(evaluatedGestures, nigam04, label='nigam04')
    ax.plot(evaluatedGestures, nigam06, label='nigam06')
    ax.plot(evaluatedGestures, nigam08, label='nigam08')
    ax.plot(evaluatedGestures, nigam10, label='nigam10')
    ax.plot(evaluatedGestures, weak, label='weak')
    ax.grid(color='gainsboro', linewidth=1)
    return ax


def graphNigam(database, results1, results2):
    fig, ax = plt.subplots(nrows=2, ncols=2, sharey='row', sharex='col', dpi=300)
    ax[0, 0] = vectorNigam(database, results1, ax[0, 0], typeDA='LDA', shotStart=1)
    ax[0, 1] = vectorNigam(database, results2, ax[0, 1], typeDA='LDA', shotStart=2)
    ax[1, 0] = vectorNigam(database, results1, ax[1, 0], typeDA='QDA', shotStart=1)
    ax[1, 1] = vectorNigam(database, results2, ax[1, 1], typeDA='QDA', shotStart=2)
    ax[1, 1].legend(prop={'size': 6})
    ax[0, 0].set_title('1 labeled gesture')
    ax[0, 1].set_title('2 labeled gesture')
    ax[0, 0].set_ylabel('Acc LDA' + database)
    ax[1, 0].set_ylabel('Acc QDA' + database)
    ax[1, 0].set_xlabel('Number Unlabeled Gestures')
    ax[1, 1].set_xlabel('Number Unlabeled Gestures')
    plt.show()


def vectorAll(database, results, ax, typeDA='LDA', shotStart=1, max=1, min=1):
    if database == 'EPN':
        classes = 5
        rep = 25
        AllGestures = [0, 1, 5, 10, 20, 40, 60, 80, 100, 120, rep * classes - shotStart * classes]
        evaluatedGestures = [a for a in AllGestures if a <= rep * classes - shotStart * classes]
        if typeDA == 'LDA':
            nigamBest = 0.4
        else:
            nigamBest = 1.0
    elif database == 'Nina5':
        classes = 18
        rep = 4
        AllGestures = [0, 1, 5, 10, 20, 30, 40, 50, 54, rep * classes - shotStart * classes]
        evaluatedGestures = [a for a in AllGestures if a <= rep * classes - shotStart * classes]
        if typeDA == 'LDA':
            nigamBest = 0.2
        else:
            nigamBest = 1.0
    elif database == 'Cote':
        classes = 7
        rep = 4
        AllGestures = [0, 1, 6, 11, 16, 21, rep * classes - shotStart * classes]
        evaluatedGestures = [a for a in AllGestures if a <= rep * classes - shotStart * classes]
        if typeDA == 'LDA':
            nigamBest = 1.0
        else:
            nigamBest = 1.0

    Mcc = []
    label = []
    sequential = []
    supervised = []
    nigam = []
    weak = []
    wilcoxonValue = 0
    for i in evaluatedGestures:
        if i != 0:
            Mcc.append(results[typeDA + '_incre_proposedMcc'].loc[results['unlabeled Gesture'] == i].mean())
            label.append(results[typeDA + '_incre_proposedLabel'].loc[results['unlabeled Gesture'] == i].mean())
            sequential.append(results[typeDA + '_incre_sequential'].loc[results['unlabeled Gesture'] == i].mean())
            supervised.append(results[typeDA + '_incre_supervised'].loc[results['unlabeled Gesture'] == i].mean())
            nigam.append(
                results[typeDA + '_incre_Nigam_' + str(nigamBest)].loc[results['unlabeled Gesture'] == i].mean())
            weak.append(results[typeDA + '_weak'].loc[results['unlabeled Gesture'] == 1].mean())

            ######Wilcoxon
            modelw = results[typeDA + '_weak'].loc[results['unlabeled Gesture'] == 1].values
            model1 = results[typeDA + '_incre_proposedMcc'].loc[results['unlabeled Gesture'] == i].values
            model2 = results[typeDA + '_incre_proposedLabel'].loc[results['unlabeled Gesture'] == i].values
            model3 = results[typeDA + '_incre_Nigam_' + str(nigamBest)].loc[results['unlabeled Gesture'] == i].values

            pValue = 0.01
            if stats.wilcoxon(model1, modelw, zero_method='zsplit')[
                1] < pValue and wilcoxonValue == 0:
                wilcoxonValue = np.mean(model1)
            # if stats.wilcoxon(model1, model2, alternative='greater', zero_method='zsplit')[1] < pValue and \
            #         stats.wilcoxon(model1, model3, alternative='greater', zero_method='zsplit')[
            #             1] < pValue and wilcoxonValue == 0:
            #     wilcoxonValue = np.mean(model1)
            # if stats.wilcoxon(model2, model1, alternative='greater', zero_method='zsplit')[1] < pValue and \
            #         stats.wilcoxon(model2, model3, alternative='greater', zero_method='zsplit')[
            #             1] < pValue and wilcoxonValue == 0:
            #     wilcoxonValue = np.mean(model2)
            # if stats.wilcoxon(model3, model1, alternative='greater', zero_method='zsplit')[1] < pValue and \
            #         stats.wilcoxon(model3, model2, alternative='greater', zero_method='zsplit')[
            #             1] < pValue and wilcoxonValue == 0:
            #     wilcoxonValue = np.mean(model3)


        else:
            Mcc.append(results[typeDA + '_weak'].loc[results['unlabeled Gesture'] == 1].mean())
            label.append(results[typeDA + '_weak'].loc[results['unlabeled Gesture'] == 1].mean())
            sequential.append(results[typeDA + '_weak'].loc[results['unlabeled Gesture'] == 1].mean())
            supervised.append(results[typeDA + '_weak'].loc[results['unlabeled Gesture'] == 1].mean())
            nigam.append(results[typeDA + '_weak'].loc[results['unlabeled Gesture'] == 1].mean())
            weak.append(results[typeDA + '_weak'].loc[results['unlabeled Gesture'] == 1].mean())

    maxRef = np.floor(np.max(Mcc) * 100) / 100
    if maxRef > max or max == 0:
        max = maxRef
    minRef = np.floor(np.min(sequential) * 100) / 100
    if minRef < min or min == 0:
        min = minRef
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_ticks(np.arange(0, np.max(evaluatedGestures) + 1, np.floor(np.max(evaluatedGestures) / 5)))
    ax.plot(evaluatedGestures, Mcc, label='proposed', zorder=8)
    ax.plot(evaluatedGestures, label, label='label', zorder=5)
    ax.plot(evaluatedGestures, nigam, label='nigam(' + str(nigamBest) + ')', zorder=6)
    ax.plot(evaluatedGestures, sequential, label='sequential', zorder=4)
    ax.plot(evaluatedGestures, supervised, label='supervised incremental', zorder=7)
    ax.plot(evaluatedGestures, weak, label='weak', color='black', zorder=2)
    if wilcoxonValue != 0:
        wilcoxon = np.ones(len(evaluatedGestures)) * wilcoxonValue
        ax.plot(evaluatedGestures, wilcoxon, label='wilcoxon(95%)', color='black', zorder=3, linestyle='--')

    ax.grid(color='gainsboro', linewidth=1, zorder=1)
    return ax, max, min


def graphAll(database, results1, results2, results3):
    fig, ax = plt.subplots(nrows=2, ncols=3, sharey='row', sharex='col', dpi=300)
    fig.subplots_adjust(wspace=0, hspace=0)
    # fig.tight_layout()
    max = 0
    min = 0
    ax[0, 0], max, min = vectorAll(database, results1, ax[0, 0], typeDA='LDA', shotStart=1, max=max, min=min)
    ax[0, 1], max, min = vectorAll(database, results2, ax[0, 1], typeDA='LDA', shotStart=2, max=max, min=min)
    ax[0, 2], max, min = vectorAll(database, results3, ax[0, 2], typeDA='LDA', shotStart=3, max=max, min=min)
    # ax[0, 0].yaxis.set_ticks(np.arange(min, max, np.floor(max * 100 / 5) / 100))
    max = 0
    min = 0
    ax[1, 0], max, min = vectorAll(database, results1, ax[1, 0], typeDA='QDA', shotStart=1, max=max, min=min)
    ax[1, 1], max, min = vectorAll(database, results2, ax[1, 1], typeDA='QDA', shotStart=2, max=max, min=min)
    ax[1, 2], max, min = vectorAll(database, results3, ax[1, 2], typeDA='QDA', shotStart=3, max=max, min=min)
    # ax[1, 0].yaxis.set_ticks(np.arange(min, max, np.floor((max-min) * 100 / 5) / 100))

    # ax[1, 2].legend(loc='upper center', bbox_to_anchor=(-0.5, -0.05),
    #       fancybox=True, shadow=True,prop={'size': 8},ncol=4)
    ax[0, 0].set_title('1 LG per class')
    ax[0, 1].set_title('2 LG per class')
    ax[0, 2].set_title('3 LG per class')
    ax[0, 0].set_ylabel('Acc (LDA ' + database + ')')
    ax[1, 0].set_ylabel('Acc (QDA ' + database + ')')
    ax[1, 0].set_xlabel('Number UG')
    ax[1, 1].set_xlabel('Number UG')
    ax[1, 2].set_xlabel('Number UG')

    plt.show()


def scatterAnalysis(results1, results2, results3, database, shotStart, marker, typeDA, nigam):
    if database == 'EPN':
        classes = 5
        rep = 25
        AllGestures = [1, 5, 40, 80, 120, rep * classes - shotStart * classes]
        evaluatedGestures = [a for a in AllGestures if a <= rep * classes - shotStart * classes]
    elif database == 'Nina5':
        classes = 18
        rep = 4
        AllGestures = [1, 5, 20, 40, 54, rep * classes - shotStart * classes]
        evaluatedGestures = [a for a in AllGestures if a <= rep * classes - shotStart * classes]
    elif database == 'Cote':
        classes = 7
        rep = 4
        AllGestures = [1, 6, 11, 16, 21, rep * classes - shotStart * classes]
        evaluatedGestures = [a for a in AllGestures if a <= rep * classes - shotStart * classes]

    colorsList = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    plt.grid(color='gainsboro', linewidth=1, zorder=1)
    idxGlobal = plot_plt(0, results1, typeDA, nigam, evaluatedGestures, marker, colorsList)
    idxGlobal = plot_plt(idxGlobal, results2, typeDA, nigam, evaluatedGestures, marker, colorsList)
    _ = plot_plt(idxGlobal, results3, typeDA, nigam, evaluatedGestures, marker, colorsList)

    # plt.xticks(np.arange(1, 13),
    #            ['Sequential', 'Label', 'Best Nigam', 'Proposed', 'Sequential', 'Label', 'Best Nigam', 'Proposed',
    #             'Sequential', 'Label', 'Best Nigam', 'Proposed'], rotation=45)

    plt.show()


def plot_plt(idxGlobal, results, typeDA, nigam, evaluatedGestures, marker, colorsList):
    weak = results[typeDA + '_weak'].loc[results[typeDA + '_weak'].notna()].mean()

    xAxisList = np.arange(len(evaluatedGestures)) + idxGlobal
    methodNameList = ['_incre_sequential', '_incre_proposedLabel', '_incre_Nigam_' + str(nigam),
                      '_incre_proposedMcc']

    idx = 0
    for methodName in methodNameList:
        method = []
        for unlabeled in evaluatedGestures:
            method.append(results[typeDA + methodName].loc[
                              (results[typeDA + methodName].notna()) & (
                                      results['unlabeled Gesture'] == unlabeled)].mean())

        plt.plot(xAxisList, method, c=colorsList[idx], alpha=0.2, zorder=3)
        plt.scatter(xAxisList, method, marker=marker, c=colorsList[idx], label=methodName, zorder=4)
        idx += 1
    plt.plot([idxGlobal, idxGlobal + len(evaluatedGestures)], [weak, weak], color='black', zorder=2)
    return idxGlobal + len(evaluatedGestures)


def graphError(database, results1):
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex='col', dpi=300)
    fig.subplots_adjust(wspace=0, hspace=0)
    # fig.tight_layout()
    if database == 'EPN':
        evaluatedGestures = list(np.arange(1, 121))
    elif database == 'Nina5':
        evaluatedGestures = list(np.arange(1, 55))
    elif database == 'Cote':
        evaluatedGestures = list(np.arange(1, 22))
    idx = 0
    for typeDA in ['LDA', 'QDA']:
        errorP1 = []
        errorL1 = []
        errorN1 = []
        errorP2 = []
        errorL2 = []
        errorN2 = []
        nigamBest = 1.0
        for unlabeled in evaluatedGestures:
            true = results1['w_predicted_' + typeDA + '_incre_supervised'].loc[(
                    results1['unlabeled Gesture'] == unlabeled)].values
            proposed = results1['w_predicted_' + typeDA + '_incre_proposedMcc'].loc[(
                    results1['unlabeled Gesture'] == unlabeled)].values
            labels = results1['w_predicted_' + typeDA + '_incre_proposedLabel'].loc[(
                    results1['unlabeled Gesture'] == unlabeled)].values
            nigam = results1['w_predicted_' + typeDA + '_incre_Nigam_' + str(nigamBest)].loc[(
                    results1['unlabeled Gesture'] == unlabeled)].values
            true = serie2array(true)
            proposed = serie2array(proposed)
            labels = serie2array(labels)
            nigam = serie2array(nigam)
            errorP1, errorP2 = errorValues(true, proposed, errorP1, errorP2)
            errorL1, errorL2 = errorValues(true, labels, errorL1, errorL2)
            errorN1, errorN2 = errorValues(true, nigam, errorN1, errorN2)

        ax[idx, 0].plot(evaluatedGestures, errorP1, label='proposed')
        ax[idx, 0].plot(evaluatedGestures, errorL1, label='label')
        ax[idx, 0].plot(evaluatedGestures, errorN1, label='nigam')
        ax[idx, 0].grid(color='gainsboro', linewidth=1, zorder=1)
        ax[idx, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ax[idx, 1].plot(evaluatedGestures, errorP2, label='proposed')
        ax[idx, 1].plot(evaluatedGestures, errorL2, label='label')
        ax[idx, 1].plot(evaluatedGestures, errorN2, label='nigam')
        ax[idx, 1].grid(color='gainsboro', linewidth=1, zorder=1)
        ax[idx, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        idx += 1

    # ax[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #       fancybox=True, shadow=True,prop={'size': 8},ncol=3)
    ax[0, 0].set_title('Type I')
    ax[0, 1].set_title('Type II')
    ax[0, 0].set_ylabel('MSE (LDA ' + database + ')')
    ax[1, 0].set_ylabel('MSE (QDA ' + database + ')')
    ax[1, 0].set_xlabel('Number UG')
    ax[1, 1].set_xlabel('Number UG')

    plt.show()


def analysis1(folder, database, featureSet, times):
    results1 = uploadResultsDatabase(folder, database, featureSet, times, shotStart=1)
    results2 = uploadResultsDatabase(folder, database, featureSet, times, shotStart=2)
    results3 = uploadResultsDatabase(folder, database, featureSet, times, shotStart=3)

    # graphNigam(database, results1, results2)
    graphAll(database, results1, results2, results3)
    graphError(database, results1)

    # typeDA = ['LDA', 'QDA']
    # for idx in range(2):
    #     scatterAnalysis(results1, results2, results3, database, shotStart=1, marker='o', typeDA=typeDA[idx],
    #                     nigam=nigam[idx])


def serie2array(vector):
    vectorC = np.array(vector[0].replace('\n', '').replace('  ', ' ').replace('  ', ' ').replace(
        '  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('[ ', '')
                       .replace(' ]', '').replace('[', '').replace(']', '').split(' ')).astype(np.float)
    vectorC = vectorC.reshape((1, len(vectorC)))
    for i in range(1, len(vector)):
        vectorC = np.vstack((vectorC, np.array(
            vector[i].replace('\n', '').replace('  ', ' ').replace('  ', ' ').replace(
                '  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('[ ', '')
                .replace(' ]', '').replace('[', '').replace(']', '').split(' ')).astype(np.float)))
    return vectorC


def errorValues(vectorTrue, vectorObs, vector1, vector2):
    error_type1 = 0
    error_type2 = 0
    x, y = np.shape(vectorTrue)

    for i in range(x):
        aux1 = 0
        aux2 = 0
        idx1 = 0
        idx2 = 0
        for j in range(y):
            if vectorTrue[i, j] == 1:
                aux1 += (vectorTrue[i, j] - vectorObs[i, j]) ** 2
                idx1 += 1
            else:
                aux2 += (vectorTrue[i, j] - vectorObs[i, j]) ** 2
                idx2 += 1
        error_type1 += aux1 / idx1
        error_type2 += aux2 / idx2
    if len(vector1) != 0:
        vector1.append((error_type1 / x) + vector1[-1])
        vector2.append((error_type2 / x) + vector2[-1])
    else:
        vector1.append((error_type1 / x))
        vector2.append((error_type2 / x))
    return vector1, vector2


# shots
def vectors_calculation_Old(results_Old, shotStart, featureSet):
    yLDA = np.array(results_Old['IndLDA'].loc[results_Old['Feature Set'] == featureSet]) * 100
    yQDA = np.array(results_Old['IndQDA'].loc[results_Old['Feature Set'] == featureSet]) * 100
    yLDA_L = []
    yQDA_L = []
    yLDA_V = []
    yQDA_V = []
    yLDA_O = []
    yQDA_O = []
    for i in range(len(yLDA)):
        yLDA_L.append(np.array(
            results_Old['LiuLDA'].loc[
                (results_Old['Feature Set'] == featureSet) & (results_Old['# shots'] == shotStart)]) * 100)
        yQDA_L.append(np.array(
            results_Old['LiuQDA'].loc[
                (results_Old['Feature Set'] == featureSet) & (results_Old['# shots'] == shotStart)]) * 100)
        yLDA_V.append(np.array(
            results_Old['VidLDA'].loc[
                (results_Old['Feature Set'] == featureSet) & (results_Old['# shots'] == shotStart)]) * 100)
        yQDA_V.append(np.array(
            results_Old['VidQDA'].loc[
                (results_Old['Feature Set'] == featureSet) & (results_Old['# shots'] == shotStart)]) * 100)
        yLDA_O.append(np.array(
            results_Old['OurLDA'].loc[
                (results_Old['Feature Set'] == featureSet) & (results_Old['# shots'] == shotStart)]) * 100)
        yQDA_O.append(np.array(
            results_Old['OurQDA'].loc[
                (results_Old['Feature Set'] == featureSet) & (results_Old['# shots'] == shotStart)]) * 100)
    return yLDA, yQDA, yLDA_L, yQDA_L, yLDA_V, yQDA_V, yLDA_O, yQDA_O


# %% Analysis
import Experimet1_Visualization as EV

# analysis1(folder='resultsNew/', database='Cote', featureSet=1, times=4)
analysis1(folder='resultsNew3/', database='Cote', featureSet=1, times=8)

# analysis1(folder='resultsNew/', database='EPN', featureSet=1, times=4)
analysis1(folder='resultsNew3/', database='EPN', featureSet=1, times=8)

# analysis1(folder='resultsNew/', database='Nina5', featureSet=1, times=4)
analysis1(folder='resultsNew3/', database='Nina5', featureSet=1, times=8)
# analysis1(folder='resultsNew/', database='Nina5', featureSet=1, times=4)
# analysis1(folder='resultsNew/', database='EPN', featureSet=1, times=4)
