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
def uploadResultsDatabase(place, database, shotStart, samplesInMemory, featureSet):
    if database == 'Nina5':
        repetitions = 4
        peoplei_i = 1
        peoplei_f = 10
        classes = 18
        rows = classes * (repetitions - shotStart)
        times = 2
    elif database == 'Cote':
        repetitions = 4
        peoplei_i = 20
        peoplei_f = 36
        classes = 7
        rows = classes * (repetitions - shotStart)
        times = 2
    elif database == 'EPN':
        repetitions = 25
        peoplei_i = 31
        peoplei_f = 60
        classes = 5
        rows = classes * (repetitions - shotStart)
        times = 2
    try:
        auxFrame = pd.read_csv(place + '_' + database + '_FeatureSet_' + str(featureSet) + '_startPerson_' + str(
            peoplei_i) + '_endPerson_' + str(peoplei_i) + '_shotStart_' + str(shotStart) + '_memmory_' + str(
            samplesInMemory) + '.csv')
        resultsTest = auxFrame[:times * rows]
    except:
        print('file not found')
        print(place + '_' + database + '_FeatureSet_' + str(featureSet) + '_startPerson_' + str(
            peoplei_i) + '_endPerson_' + str(peoplei_i) + '_shotStart_' + str(shotStart) + '_memmory_' + str(
            samplesInMemory) + '.csv')

        resultsTest = pd.DataFrame()

    for i in range(peoplei_i + 1, peoplei_f + 1):
        try:
            auxFrame = pd.read_csv(place + '_' + database + '_FeatureSet_' + str(featureSet) + '_startPerson_' + str(
                i) + '_endPerson_' + str(i) + '_shotStart_' + str(shotStart) + '_memmory_' + str(
                samplesInMemory) + '.csv')
            resultsTest = pd.concat([resultsTest, auxFrame[:times * rows]], ignore_index=True)

        except:
            print('file not found')
            print(place + '_' + database + '_FeatureSet_' + str(featureSet) + '_startPerson_' + str(
                i) + '_endPerson_' + str(i) + '_shotStart_' + str(shotStart) + '_memmory_' + str(
                samplesInMemory) + '.csv')

    return resultsTest.drop(columns='Unnamed: 0'), rows


def vectors_calculation(results, rows):
    LDA_NoAdaptive = []
    LDA_MSDA_JS = []
    LDA_NoAdaptive_SemiSuperv_Baseline = []
    LDA_NoAdaptive_SemiSuperv_PostProb = []
    LDA_NoAdaptive_SemiSuperv_PostProb_MSDA_JS = []
    LDA_MSDA_JS_SemiSuperv_Baseline = []
    LDA_MSDA_JS_SemiSuperv_PostProb = []
    LDA_MSDA_JS_SemiSuperv_PostProb_MSDA_JS = []

    QDA_NoAdaptive = []
    QDA_MSDA_JS = []
    QDA_NoAdaptive_SemiSuperv_Baseline = []
    QDA_NoAdaptive_SemiSuperv_PostProb = []
    QDA_NoAdaptive_SemiSuperv_PostProb_MSDA_JS = []
    QDA_MSDA_JS_SemiSuperv_Baseline = []
    QDA_MSDA_JS_SemiSuperv_PostProb = []
    QDA_MSDA_JS_SemiSuperv_PostProb_MSDA_JS = []

    for i in range(rows + 1):
        if i == 0:
            LDA_NoAdaptive.append(results['AccLDAfew'].loc[results['# shots'] == 1].mean() * 100)
            LDA_MSDA_JS.append(results['AccLDAProp_JS'].loc[results['# shots'] == 1].mean() * 100)
            LDA_NoAdaptive_SemiSuperv_Baseline.append(
                results['AccLDAfew'].loc[results['# shots'] == 1].mean() * 100)
            LDA_NoAdaptive_SemiSuperv_PostProb.append(
                results['AccLDAfew'].loc[results['# shots'] == 1].mean() * 100)
            LDA_NoAdaptive_SemiSuperv_PostProb_MSDA_JS.append(
                results['AccLDAfew'].loc[results['# shots'] == 1].mean() * 100)
            LDA_MSDA_JS_SemiSuperv_Baseline.append(
                results['AccLDAProp_JS'].loc[results['# shots'] == 1].mean() * 100)
            LDA_MSDA_JS_SemiSuperv_PostProb.append(
                results['AccLDAProp_JS'].loc[results['# shots'] == 1].mean() * 100)
            LDA_MSDA_JS_SemiSuperv_PostProb_MSDA_JS.append(
                results['AccLDAProp_JS'].loc[results['# shots'] == 1].mean() * 100)

            QDA_NoAdaptive.append(results['AccQDAfew'].loc[results['# shots'] == 1].mean() * 100)
            QDA_MSDA_JS.append(results['AccQDAProp_JS'].loc[results['# shots'] == 1].mean() * 100)
            QDA_NoAdaptive_SemiSuperv_Baseline.append(
                results['AccQDAfew'].loc[results['# shots'] == 1].mean() * 100)
            QDA_NoAdaptive_SemiSuperv_PostProb.append(
                results['AccQDAfew'].loc[results['# shots'] == 1].mean() * 100)
            QDA_NoAdaptive_SemiSuperv_PostProb_MSDA_JS.append(
                results['AccQDAfew'].loc[results['# shots'] == 1].mean() * 100)
            QDA_MSDA_JS_SemiSuperv_Baseline.append(
                results['AccQDAProp_JS'].loc[results['# shots'] == 1].mean() * 100)
            QDA_MSDA_JS_SemiSuperv_PostProb.append(
                results['AccQDAProp_JS'].loc[results['# shots'] == 1].mean() * 100)
            QDA_MSDA_JS_SemiSuperv_PostProb_MSDA_JS.append(
                results['AccQDAProp_JS'].loc[results['# shots'] == 1].mean() * 100)
        else:
            LDA_NoAdaptive.append(results['AccLDAfew'].loc[results['# shots'] == 1].mean() * 100)
            LDA_MSDA_JS.append(results['AccLDAProp_JS'].loc[results['# shots'] == 1].mean() * 100)
            LDA_NoAdaptive_SemiSuperv_Baseline.append(
                results['LDA_ACC_Baseline'].loc[results['# shots'] == i].mean() * 100)
            LDA_NoAdaptive_SemiSuperv_PostProb.append(
                results['LDA_ACC_PostProb'].loc[results['# shots'] == i].mean() * 100)
            LDA_NoAdaptive_SemiSuperv_PostProb_MSDA_JS.append(
                results['LDA_ACC_PostProb_MSDA_JS'].loc[results['# shots'] == i].mean() * 100)
            LDA_MSDA_JS_SemiSuperv_Baseline.append(
                results['LDA_ACC_Baseline_adapt'].loc[results['# shots'] == i].mean() * 100)
            LDA_MSDA_JS_SemiSuperv_PostProb.append(
                results['LDA_ACC_PostProb_adapt'].loc[results['# shots'] == i].mean() * 100)
            LDA_MSDA_JS_SemiSuperv_PostProb_MSDA_JS.append(
                results['LDA_ACC_PostProb_MSDA_JS_adapt'].loc[results['# shots'] == i].mean() * 100)

            QDA_NoAdaptive.append(results['AccQDAfew'].loc[results['# shots'] == 1].mean() * 100)
            QDA_MSDA_JS.append(results['AccQDAProp_JS'].loc[results['# shots'] == 1].mean() * 100)
            QDA_NoAdaptive_SemiSuperv_Baseline.append(
                results['QDA_ACC_Baseline'].loc[results['# shots'] == i].mean() * 100)
            QDA_NoAdaptive_SemiSuperv_PostProb.append(
                results['QDA_ACC_PostProb'].loc[results['# shots'] == i].mean() * 100)
            QDA_NoAdaptive_SemiSuperv_PostProb_MSDA_JS.append(
                results['QDA_ACC_PostProb_MSDA_JS'].loc[results['# shots'] == i].mean() * 100)
            QDA_MSDA_JS_SemiSuperv_Baseline.append(
                results['QDA_ACC_Baseline_adapt'].loc[results['# shots'] == i].mean() * 100)
            QDA_MSDA_JS_SemiSuperv_PostProb.append(
                results['QDA_ACC_PostProb_adapt'].loc[results['# shots'] == i].mean() * 100)
            QDA_MSDA_JS_SemiSuperv_PostProb_MSDA_JS.append(
                results['QDA_ACC_PostProb_MSDA_JS_adapt'].loc[results['# shots'] == i].mean() * 100)
    return LDA_NoAdaptive, LDA_MSDA_JS, LDA_NoAdaptive_SemiSuperv_Baseline, LDA_NoAdaptive_SemiSuperv_PostProb, \
           LDA_NoAdaptive_SemiSuperv_PostProb_MSDA_JS, LDA_MSDA_JS_SemiSuperv_Baseline, LDA_MSDA_JS_SemiSuperv_PostProb, \
           LDA_MSDA_JS_SemiSuperv_PostProb_MSDA_JS, QDA_NoAdaptive, QDA_MSDA_JS, QDA_NoAdaptive_SemiSuperv_Baseline, \
           QDA_NoAdaptive_SemiSuperv_PostProb, QDA_NoAdaptive_SemiSuperv_PostProb_MSDA_JS, \
           QDA_MSDA_JS_SemiSuperv_Baseline, QDA_MSDA_JS_SemiSuperv_PostProb, QDA_MSDA_JS_SemiSuperv_PostProb_MSDA_JS, \
           results['time_LDA_ACC_PostProb_MSDA_JS'].mean(), results['time_QDA_ACC_PostProb_MSDA_JS'].mean(), \
           results['time_LDA_ACC_PostProb_MSDA_JS'].std(), results['time_QDA_ACC_PostProb_MSDA_JS'].std(), \
           results['time_LDA_ACC_PostProb_MSDA_JS_adapt'].mean(), results['time_QDA_ACC_PostProb_MSDA_JS_adapt'].mean(), \
           results['time_LDA_ACC_PostProb_MSDA_JS_adapt'].std(), results['time_QDA_ACC_PostProb_MSDA_JS_adapt'].std()


def graphs(rows, title, LDA_NoAdaptive, LDA_MSDA_JS, LDA_NoAdaptive_SemiSuperv_Baseline,
           LDA_NoAdaptive_SemiSuperv_PostProb, LDA_NoAdaptive_SemiSuperv_PostProb_MSDA_JS,
           LDA_MSDA_JS_SemiSuperv_Baseline, LDA_MSDA_JS_SemiSuperv_PostProb, LDA_MSDA_JS_SemiSuperv_PostProb_MSDA_JS,
           QDA_NoAdaptive, QDA_MSDA_JS, QDA_NoAdaptive_SemiSuperv_Baseline, QDA_NoAdaptive_SemiSuperv_PostProb,
           QDA_NoAdaptive_SemiSuperv_PostProb_MSDA_JS, QDA_MSDA_JS_SemiSuperv_Baseline, QDA_MSDA_JS_SemiSuperv_PostProb,
           QDA_MSDA_JS_SemiSuperv_PostProb_MSDA_JS, x_Old, yLDA, yQDA, yLDA_L, yQDA_L, yLDA_V, yQDA_V, yLDA_O, yQDA_O):
    x = [*range(rows + 1)]
    plt.plot(x, LDA_NoAdaptive, label='Baseline')
    plt.plot(x, LDA_MSDA_JS, label='Baseline_adapt')
    plt.plot(x_Old, yLDA_L, label='Liu')
    plt.plot(x_Old, yLDA_V, label='Vidovic')
    plt.plot(x_Old, yLDA_O, label='Our_RQ1')

    plt.plot(x, LDA_NoAdaptive_SemiSuperv_Baseline, label='LDA_Optimal', color='black', linestyle='--')
    plt.plot(x, LDA_MSDA_JS_SemiSuperv_Baseline, label='LDA_Optimal_adapt', linestyle='--')
    plt.plot(x, LDA_NoAdaptive_SemiSuperv_PostProb, label='Our_RQ2_onlypost')
    plt.plot(x, LDA_MSDA_JS_SemiSuperv_PostProb, label='Our_RQ2_onlypost_adapt')
    plt.plot(x, LDA_NoAdaptive_SemiSuperv_PostProb_MSDA_JS, label='Our_RQ2', color='tab:purple')
    plt.plot(x, LDA_MSDA_JS_SemiSuperv_PostProb_MSDA_JS, label='Our_RQ2_adapt')

    # plt.plot(x_Old, yLDA, label='Ideal', color='black', linestyle='--')

    plt.xlabel('number of unlabeled gestures\n (over time)')
    plt.ylabel('accuracy [%]')
    plt.title(title + ' (LDA)')
    plt.legend()
    # plt.legend(loc='lower center', bbox_to_anchor=(2, -0.7), ncol=1)
    # plt.ylim(0.5, 1)
    plt.grid(color='gainsboro', linewidth=1)
    plt.show()

    x = [*range(rows + 1)]
    plt.plot(x, QDA_NoAdaptive, label='Baseline')
    plt.plot(x, QDA_MSDA_JS, label='Baseline_adapt')
    plt.plot(x_Old, yQDA_L, label='Liu')
    plt.plot(x_Old, yQDA_V, label='Vidovic')
    plt.plot(x_Old, yQDA_O, label='Our_RQ1')

    plt.plot(x, QDA_NoAdaptive_SemiSuperv_Baseline, label='QDA_Optimal', color='black', linestyle='--')
    plt.plot(x, QDA_MSDA_JS_SemiSuperv_Baseline, label='QDA_Optimal_adapt', linestyle='--')
    plt.plot(x, QDA_NoAdaptive_SemiSuperv_PostProb, label='Our_RQ2_onlypost')
    plt.plot(x, QDA_MSDA_JS_SemiSuperv_PostProb, label='Our_RQ2_onlypost_adapt')
    plt.plot(x, QDA_NoAdaptive_SemiSuperv_PostProb_MSDA_JS, label='Our_RQ2', color='tab:purple')
    plt.plot(x, QDA_MSDA_JS_SemiSuperv_PostProb_MSDA_JS, label='Our_RQ2_adapt')

    # plt.plot(x_Old, yQDA, label='Ideal', color='black', linestyle='--')

    # if len(x) > 100:
    #     x = [0] + [*range(10, rows + 1)]
    #     z = np.array(QDA_PostProb_MSDA_KL)
    #     plt.plot(x, z[x], label='Our_RQ2', color='tab:purple')
    # else:
    #     plt.plot(x, QDA_PostProb_MSDA_KL, label='Our_RQ2', color='tab:purple')

    plt.xlabel('number of unlabeled gestures\n (over time)')
    plt.ylabel('accuracy [%]')
    plt.title(title + ' (QDA)')
    plt.legend()
    # plt.legend(loc='lower center', bbox_to_anchor=(1.5, -1), ncol=1)
    # plt.ylim(0.5, 1)
    plt.grid(color='gainsboro', linewidth=1)
    plt.show()


def analysis(place, database, shotStart, samplesInMemory, featureSet):
    results, rows = uploadResultsDatabase(place, database, shotStart, samplesInMemory, featureSet)
    results_Old, x_Old = uploadResultsDatabasesVF1('../ResultsExp1_RQ1/', database, windowSize='295')
    LDA_NoAdaptive, LDA_MSDA_JS, LDA_NoAdaptive_SemiSuperv_Baseline, LDA_NoAdaptive_SemiSuperv_PostProb, \
    LDA_NoAdaptive_SemiSuperv_PostProb_MSDA_JS, LDA_MSDA_JS_SemiSuperv_Baseline, LDA_MSDA_JS_SemiSuperv_PostProb, \
    LDA_MSDA_JS_SemiSuperv_PostProb_MSDA_JS, QDA_NoAdaptive, QDA_MSDA_JS, QDA_NoAdaptive_SemiSuperv_Baseline, \
    QDA_NoAdaptive_SemiSuperv_PostProb, QDA_NoAdaptive_SemiSuperv_PostProb_MSDA_JS, \
    QDA_MSDA_JS_SemiSuperv_Baseline, QDA_MSDA_JS_SemiSuperv_PostProb, QDA_MSDA_JS_SemiSuperv_PostProb_MSDA_JS, \
    time_LDA, time_QDA, time_LDA_std, time_QDA_std, time_LDA_adapt, time_QDA_adapt, time_LDA_std_adapt, time_QDA_std_adapt \
        = vectors_calculation(results, rows)
    print(database)
    print('timeLDA', time_LDA)
    print('std', time_LDA_std)
    print('timeQDA', time_QDA)
    print('std', time_QDA_std)
    print('timeLDA_adapt', time_LDA_adapt)
    print('std_adapt', time_LDA_std_adapt)
    print('timeQDA_adapt', time_QDA_adapt)
    print('std_adapt', time_QDA_std_adapt)
    yLDA, yQDA, yLDA_L, yQDA_L, yLDA_V, yQDA_V, yLDA_O, yQDA_O = vectors_calculation_Old(results_Old, shotStart,
                                                                                         featureSet)
    graphs(rows, database, LDA_NoAdaptive, LDA_MSDA_JS, LDA_NoAdaptive_SemiSuperv_Baseline,
           LDA_NoAdaptive_SemiSuperv_PostProb, LDA_NoAdaptive_SemiSuperv_PostProb_MSDA_JS,
           LDA_MSDA_JS_SemiSuperv_Baseline, LDA_MSDA_JS_SemiSuperv_PostProb, LDA_MSDA_JS_SemiSuperv_PostProb_MSDA_JS,
           QDA_NoAdaptive, QDA_MSDA_JS, QDA_NoAdaptive_SemiSuperv_Baseline, QDA_NoAdaptive_SemiSuperv_PostProb,
           QDA_NoAdaptive_SemiSuperv_PostProb_MSDA_JS, QDA_MSDA_JS_SemiSuperv_Baseline, QDA_MSDA_JS_SemiSuperv_PostProb,
           QDA_MSDA_JS_SemiSuperv_PostProb_MSDA_JS, x_Old, yLDA, yQDA, yLDA_L, yQDA_L, yLDA_V, yQDA_V, yLDA_O, yQDA_O)


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

######### Cote
analysis(place='results_JS/', database='Cote', shotStart=1, samplesInMemory=1, featureSet=1)

analysis(place='results_JS/', database='Cote', shotStart=2, samplesInMemory=1, featureSet=1)

EV.analysis(place='resultsPBS_2/Cote_', database='Cote')

####### Nina5
# analysis(place='results_JS/', database = 'Nina5', shotStart=1, samplesInMemory=1, featureSet=1)

# analysis(place='results_JS/', database = 'Nina5', shotStart=2, samplesInMemory=1, featureSet=1)
#
# EV.analysis(place = 'resultsPBS_2/Nina5_', database = 'Nina5')


########## EPN
# analysis(place='results_JS/', database = 'EPN', shotStart=1, samplesInMemory=1, featureSet=1)

# analysis(place='results_JS/', database = 'EPN', shotStart=1, samplesInMemory=1, featureSet=1)

# EV.analysis(place = 'resultsPBS_2/EPN_', database = 'EPN')
