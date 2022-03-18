# %% Libraries

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# %% Friedman rank test for all DA approaches

def friedman_test(*args):
    """
        From: https://github.com/citiususc/stac/blob/master/stac/nonparametric_tests.py
        Performs a Friedman ranking test.
        Tests the hypothesis that in a set of k dependent samples groups (where k >= 2) at least two of the groups represent populations with different median values.

        Parameters
        ----------
        sample1, sample2, ... : array_like
            The sample measurements for each group.

        Returns
        -------
        F-value : float
            The computed F-value of the test.
        p-value : float
            The associated p-value from the F-distribution.
        rankings : array_like
            The ranking for each group.
        pivots : array_like
            The pivotal quantities for each group.

        References
        ----------
        M. Friedman, The use of ranks to avoid the assumption of normality implicit in the analysis of variance, Journal of the American Statistical Association 32 (1937) 674–701.
        D.J. Sheskin, Handbook of parametric and nonparametric statistical procedures. crc Press, 2003, Test 25: The Friedman Two-Way Analysis of Variance by Ranks
    """
    k = len(args)
    if k < 2: raise ValueError('Less than 2 levels')
    n = len(args[0])
    if len(set([len(v) for v in args])) != 1: raise ValueError('Unequal number of samples')

    rankings = []
    for i in range(n):
        row = [col[i] for col in args]
        row_sort = sorted(row, reverse=True)
        rankings.append([row_sort.index(v) + 1 + (row_sort.count(v) - 1) / 2. for v in row])

    rankings_avg = [np.mean([case[j] for case in rankings]) for j in range(k)]
    rankings_cmp = [r / np.sqrt(k * (k + 1) / (6. * n)) for r in rankings_avg]
    aux = [r ** 2 for r in rankings_avg]
    chi2 = ((12 * n) / float((k * (k + 1)))) * (
            (np.sum(aux)) - ((k * (k + 1) ** 2) / float(4)))
    iman_davenport = ((n - 1) * chi2) / float((n * (k - 1) - chi2))

    p_value = 1 - stats.f.cdf(iman_davenport, k - 1, (k - 1) * (n - 1))

    return iman_davenport, p_value, rankings_avg, rankings_cmp


def holm_test(ranks, control=None):
    """
        From: https://github.com/citiususc/stac/blob/master/stac/nonparametric_tests.py
        Performs a Holm post-hoc test using the pivot quantities obtained by a ranking test.
        Tests the hypothesis that the ranking of the control method is different to each of the other methods.

        Parameters
        ----------
        pivots : dictionary_like
            A dictionary with format 'groupname':'pivotal quantity'
        control : string optional
            The name of the control method (one vs all), default None (all vs all)

        Returns
        ----------
        Comparions : array-like
            Strings identifier of each comparison with format 'group_i vs group_j'
        Z-values : array-like
            The computed Z-value statistic for each comparison.
        p-values : array-like
            The associated p-value from the Z-distribution wich depends on the index of the comparison
        Adjusted p-values : array-like
            The associated adjusted p-values wich can be compared with a significance level

        References
        ----------
        O.J. S. Holm, A simple sequentially rejective multiple test procedure, Scandinavian Journal of Statistics 6 (1979) 65–70.
    """
    k = len(ranks)
    values = list(ranks.values())
    keys = list(ranks.keys())
    if not control:
        control_i = values.index(min(values))
    else:
        control_i = control - 1

    comparisons = [keys[control_i] + " vs " + keys[i] for i in range(k) if i != control_i]
    z_values = [abs(values[control_i] - values[i]) for i in range(k) if i != control_i]
    p_values = [2 * (1 - stats.norm.cdf(abs(z))) for z in z_values]
    # Sort values by p_value so that p_0 < p_1
    p_values, z_values, comparisons = map(list, zip(*sorted(zip(p_values, z_values, comparisons), key=lambda t: t[0])))
    adj_p_values = [min(max((k - (j + 1)) * p_values[j] for j in range(i + 1)), 1) for i in range(k - 1)]

    return comparisons, z_values, p_values, adj_p_values, keys[control_i]


def friedman_analysis(data_frame):
    data_array = np.asarray(data_frame)
    num_datasets, num_methods = data_array.shape
    # print("Number of classifiers: " + str(num_methods) + " and Number of samples: " + str(num_datasets))

    alpha = 0.05  # Set this to the desired alpha/signifance level

    stat, p = stats.friedmanchisquare(*data_array)

    reject = p <= alpha
    # print("Should we reject H0 (i.e. is there a difference in the means) at the", (1 - alpha) * 100,
    #       "% confidence level?", reject, '\n')

    if not reject:
        print(
            "Exiting early. The rankings are only relevant if there was a difference in the means i.e. if we rejected h0 above")
    else:
        statistic, p_value, ranking, rank_cmp = friedman_test(*np.transpose(data_array))
        ranks = {key: ranking[i] for i, key in enumerate(list(data_frame.columns))}
        ranksComp = {key: rank_cmp[i] for i, key in enumerate(list(data_frame.columns))}

        comparisons, z_values, p_values, adj_p_values, best = holm_test(ranksComp)

        adj_p_values = np.asarray(p_values)

        for method, rank in ranks.items():
            print(method + ":", "%.1f" % rank)
        print('\n The best classifier is: ', best)
        holm_scores = pd.DataFrame({"p": adj_p_values, "sig": adj_p_values < alpha}, index=comparisons)
        print(holm_scores)


# %% Upload results of the three databases

def uploadResultsDatasets(folder, database, featureSet, times, shotStart, analysis_time=False, graph_acc=False,
                          graph_acc_batch=False, best_probs=0, best_threshold=0):
    if database == 'Nina5':
        samplesTrain = 4
        classes = 18
        people = 10
        days = 1
    elif database == 'Cote':
        samplesTrain = 9
        classes = 7
        people = 17
        days = 1
    elif database == 'EPN_120':
        samplesTrain = 25
        classes = 5
        people = 120
        days = 1
    elif database == 'Capgmyo_dbb':
        samplesTrain = 7
        classes = 8
        people = 10
        days = 2
    elif database == 'LongTerm3DC':
        samplesTrain = 2
        classes = 11
        people = 20
        days = 3

    folder = folder + '_' + database

    all_samples = (samplesTrain - shotStart) * classes
    number_gestures = 5
    gestureSet = np.append(np.arange(np.floor(all_samples / number_gestures), all_samples,
                                     np.floor(all_samples / number_gestures)), all_samples)

    dataFrame = pd.DataFrame()
    for time in range(1, times + 1):
        if database == 'Cote':
            initialperson = 20
            finalPerson = 36 + 1
        else:
            initialperson = 1
            finalPerson = people + 1
        for person in range(initialperson, finalPerson):
            try:
                resultsTest = pd.read_csv(folder + '_FS_' + str(featureSet) + '_sP_' + str(
                    person) + '_eP_' + str(person) + '_sStart_' + str(shotStart) + '_inTime_' + str(
                    time) + '_fiTime_' + str(time) + '.csv')

                if len(resultsTest) != days:
                    print('error' + '_time' + str(time) + '_person' + str(person) + '_FS' + str(featureSet))
                    print(len(resultsTest), '/', days)

                dataFrame = dataFrame.append(resultsTest)

            except:
                print('error' + '_time' + str(time) + '_person' + str(person) + '_FS' + str(featureSet))

    data_acc_mean_all = get_data_acc_mean_all(dataFrame, gestureSet)
    if best_probs != 0 and best_threshold != 0:
        data_acc_mean_best = get_data_acc_mean_ours_all(dataFrame, gestureSet, best_probs, best_threshold,
                                                        wilcoxon=graph_acc)
        data_acc_best = get_data_acc_ours_all(dataFrame, gestureSet, best_probs, best_threshold)

    else:
        data_acc_mean_best = 0
        data_acc_best = 0
    if analysis_time:
        get_data_time(dataFrame, gestureSet)
    data_acc_mean_batch = get_data_acc_mean_batch(dataFrame, gestureSet, wilcoxon=graph_acc_batch)
    return data_acc_mean_all, data_acc_mean_best, data_acc_best, data_acc_mean_batch


def get_data_acc_mean_all(dataFrame, gestureSet):
    metric = 'acc'

    data = pd.DataFrame()
    for type_DA in [['LDA', 0], ['QDA', 1]]:
        classifierSet = [type_DA[0] + '_' + 'weak', type_DA[0] + '_' + 'batch']
        for type_model in [['ours', 0], ['state_art', 1]]:
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_soft_labels')
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_labels')
            for weight in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                classifierSet.append(
                    type_DA[0] + '_' + type_model[0] + '_probs_' + str(weight))
            for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                classifierSet.append(
                    type_DA[0] + '_' + type_model[0] + '_threshold_' + str(threshold))

        weak_vect = dataFrame[metric + '_' + type_DA[0] + '_' + 'weak'].loc[
            dataFrame['number gesture'] == gestureSet[len(gestureSet) - 1]]
        for classifier in classifierSet:
            vect = dataFrame[metric + '_' + classifier].loc[
                dataFrame['number gesture'] == gestureSet[len(gestureSet) - 1]]

            data.at[0, classifier] = vect.mean() - weak_vect.mean()

    return data


def get_data_acc_mean_ours_all(dataFrame, gestureSet, best_probs, best_threshold, wilcoxon=False):
    metric = 'acc'
    p_value_control = 0.05

    data = pd.DataFrame()
    for type_DA in [['LDA', 0], ['QDA', 1]]:
        classifierSet = [type_DA[0] + '_' + 'weak']
        for type_model in [['ours', 0]]:
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_soft_labels')
            classifierSet.append(
                type_DA[0] + '_' + type_model[0] + '_probs_' + str(best_probs[type_DA[1]][type_model[1]]))
            classifierSet.append(
                type_DA[0] + '_' + type_model[0] + '_threshold_' + str(best_threshold[type_DA[1]][type_model[1]]))
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_labels')

        weak_vect = dataFrame[metric + '_' + type_DA[0] + '_' + 'weak'].loc[
            dataFrame['number gesture'] == gestureSet[len(gestureSet) - 1]]
        ours_soft_vect = dataFrame[metric + '_' + type_DA[0] + '_' + type_model[0] + '_soft_labels'].loc[
            dataFrame['number gesture'] == gestureSet[len(gestureSet) - 1]]
        for classifier in classifierSet:
            vect = dataFrame[metric + '_' + classifier].loc[
                dataFrame['number gesture'] == gestureSet[len(gestureSet) - 1]]

            if wilcoxon:
                if classifier != type_DA[0] + '_' + 'weak':
                    _, p = stats.wilcoxon(weak_vect, vect)

                    if p > p_value_control:
                        print('p value =', p, ', ' + type_DA[0] +
                              '_initial_classifier and ' + classifier + ' come from the same distribution')

                if classifier != type_DA[0] + '_' + type_model[0] + '_soft_labels':
                    _, p = stats.wilcoxon(vect, ours_soft_vect)

                    if p > p_value_control:
                        print('p value', p, ', ' + type_DA[0] +
                              '_' + type_model[
                                  0] + 'using pseudo-labels and ' + classifier + ' and comes from the same distribution')

            data.at[0, classifier] = vect.mean()

    return data


def get_data_time(dataFrame, gestureSet):
    metric = 'time_update'
    for type_DA in [['LDA', 0], ['QDA', 1]]:
        classifierSet = [type_DA[0] + '_' + 'batch']
        for type_model in [['ours', 0]]:
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_soft_labels')
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_labels')

        for classifier in classifierSet:
            vect = dataFrame[metric + '_' + classifier].loc[
                dataFrame['number gesture'] == gestureSet[len(gestureSet) - 1]]

            print('time[s]: ' + classifier, round(vect.mean() * 1000, 2), '±', round(vect.std() * 1000, 2))


def get_data_acc_ours_all(dataFrame, gestureSet, best_probs, best_threshold):
    metric = 'acc'
    data = pd.DataFrame()

    for type_DA in [['LDA', 0], ['QDA', 1]]:
        classifierSet = [type_DA[0] + '_' + 'weak']
        classifierSet_names = [type_DA[0] + '_' + 'weak']
        for type_model in [['ours', 0], ['state_art', 1]]:
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_soft_labels')
            classifierSet_names.append(type_DA[0] + '_' + type_model[0] + '_soft_labels')

            if type_model[0] == 'ours':
                classifierSet.append(
                    type_DA[0] + '_' + type_model[0] + '_probs_' + str(best_probs[type_DA[1]][type_model[1]]))
                classifierSet_names.append(type_DA[0] + '_' + type_model[0] + '_best_probs')

                classifierSet.append(
                    type_DA[0] + '_' + type_model[0] + '_threshold_' + str(best_threshold[type_DA[1]][type_model[1]]))
                classifierSet_names.append(type_DA[0] + '_' + type_model[0] + '_best_threshold')

            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_labels')
            classifierSet_names.append(type_DA[0] + '_' + type_model[0] + '_labels')

        for i in range(len(classifierSet)):
            vect = dataFrame[metric + '_' + classifierSet[i]].loc[
                dataFrame['number gesture'] == gestureSet[len(gestureSet) - 1]]
            data[classifierSet_names[i]] = vect

    return data


def get_data_acc_mean_batch(dataFrame, gestureSet, wilcoxon=False):
    metric = 'acc'
    p_value_control = 0.05

    data = pd.DataFrame()
    for type_DA in [['LDA', 0], ['QDA', 1]]:
        classifierSet = []
        for type_model in [['ours', 0], ['state_art', 1]]:
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_soft_labels')
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_labels')

        weak_vect = dataFrame[metric + '_' + type_DA[0] + '_' + 'weak'].loc[
            dataFrame['number gesture'] == gestureSet[len(gestureSet) - 1]]
        soft_label_state_vect = dataFrame[metric + '_' + type_DA[0] + '_state_art_soft_labels'].loc[
            dataFrame['number gesture'] == gestureSet[len(gestureSet) - 1]]
        label_state_vect = dataFrame[metric + '_' + type_DA[0] + '_state_art_labels'].loc[
            dataFrame['number gesture'] == gestureSet[len(gestureSet) - 1]]
        for classifier in classifierSet:
            vect = dataFrame[metric + '_' + classifier].loc[
                dataFrame['number gesture'] == gestureSet[len(gestureSet) - 1]]

            if wilcoxon:
                if classifier == type_DA[0] + '_ours_soft_labels':
                    _, p = stats.wilcoxon(vect, soft_label_state_vect)

                    if p > p_value_control:
                        print('p value: ', p, ', ' + type_DA[0] +
                              ': weighted and traditional batch classifiers using pseudo-labels come from the same distribution')

                if classifier == type_DA[0] + '_ours_labels':
                    _, p = stats.wilcoxon(vect, label_state_vect)

                    if p > p_value_control:
                        print('p value= ', p, ', ' + type_DA[0] +
                              ': weighted and traditional batch classifiers using labels come from the same distribution')

            data.at[0, classifier] = vect.mean() - weak_vect.mean()

    return data


def vect_bar_ours_all(vect, data, best_probs, best_threshold):
    for type_DA in [['LDA', 0], ['QDA', 1]]:
        classifierSet = [type_DA[0] + '_' + 'weak']
        for type_model in [['ours', 0]]:
            classifierSet.append(
                type_DA[0] + '_' + type_model[0] + '_probs_' + str(best_probs[type_DA[1]][type_model[1]]))
            classifierSet.append(
                type_DA[0] + '_' + type_model[0] + '_threshold_' + str(best_threshold[type_DA[1]][type_model[1]]))
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_soft_labels')
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_labels')

        vect.append(list(data[classifierSet].loc[len(data) - 1].values * 100))
    return vect


def vect_bar_improve(vect_LDA, vect_QDA, data, type_label):
    for type_DA in [['LDA', 0], ['QDA', 1]]:
        classifierSet = []
        for type_model in [['state_art', 0], ['ours', 0]]:
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_' + type_label)
        if type_DA[0] == 'LDA':
            vect_LDA.append(list(data[classifierSet].loc[len(data) - 1].values * 100))
        else:
            vect_QDA.append(list(data[classifierSet].loc[len(data) - 1].values * 100))
    return vect_LDA, vect_QDA


def vect_bar_NIGAM(data_acc, best_parameters_Nigam_thresholding):
    best_parameters = []
    for type_DA in [['LDA', 0], ['QDA', 1]]:
        aux_best_parameters = []

        for type_model in [['ours', 0]]:
            classifierSet = []
            list_weight = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            for weight in list_weight:
                classifierSet.append(type_DA[0] + '_' + type_model[0] + '_probs_' + str(weight))

            values = data_acc[classifierSet].loc[len(data_acc) - 1].values
            idx = np.argmax(values)
            if best_parameters_Nigam_thresholding:
                print('Best parameter λ (for Nigam-based classifier) is ' + str(list_weight[
                                                                                    idx]) + '. Accuracy Difference (wrt initial classifier)=' + str(
                    round(values[idx] * 100, 2)))
            aux_best_parameters.append(list_weight[idx])
        best_parameters.append(aux_best_parameters)
    return best_parameters


def vect_bar_THRESHOLD(data_acc, best_parameters_Nigam_thresholding):
    best_parameters = []
    for type_DA in [['LDA', 0], ['QDA', 1]]:
        aux_best_parameters = []
        for type_model in [['ours', 0]]:
            classifierSet = []
            list_threshold = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for threshold in list_threshold:
                classifierSet.append(type_DA[0] + '_' + type_model[0] + '_threshold_' + str(threshold))

            values = data_acc[classifierSet].loc[len(data_acc) - 1].values
            idx = np.argmax(values)
            if best_parameters_Nigam_thresholding:
                print('Best parameter τ (for thresholding-based classifier) is ' + str(list_threshold[
                                                                                           idx]) + '. Accuracy Difference (wrt initial classifier)=' + str(
                    round(values[idx] * 100, 2)))
            aux_best_parameters.append(list_threshold[idx])
        best_parameters.append(aux_best_parameters)
    return best_parameters


# %% Set initial variables

times = 2
windowSize = 290
shotStart = 1
folder = 'Results/'

# %% GRAPH online classifiers

def experiment1(best_parameters_Nigam_thresholding=False, analysis_time=False, graph_acc=False, friedman=False):

    data_frame_total = pd.DataFrame()
    if graph_acc:
        fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(8, 11), sharex='col')
    idx = 0
    database_names = ['NinaPro5', 'Capgmyo_dbb', 'MyoArmband', 'Long-Term 3DC', 'EMG-EPN-120']

    for database in ['Nina5', 'Capgmyo_dbb', 'Cote', 'LongTerm3DC', 'EPN_120']:


        data_frame_database = pd.DataFrame()

        data_acc_mean_all_1, _, _, _ = uploadResultsDatasets(
            folder, database, 1, times, shotStart, best_probs=0, best_threshold=0)

        data_acc_mean_all_2, _, _, _ = uploadResultsDatasets(
            folder, database, 2, times, shotStart, best_probs=0, best_threshold=0)

        data_acc_mean_all_3, _, _, _ = uploadResultsDatasets(
            folder, database, 3, times, shotStart, best_probs=0, best_threshold=0)

        ##### graph best acc
        if best_parameters_Nigam_thresholding:
            print('\nFEATURE 1: ' + database)
        best_probs1 = vect_bar_NIGAM(data_acc_mean_all_1, best_parameters_Nigam_thresholding)
        best_threshold1 = vect_bar_THRESHOLD(data_acc_mean_all_1, best_parameters_Nigam_thresholding)
        if best_parameters_Nigam_thresholding:
            print('FEATURE 2: ' + database)
        best_probs2 = vect_bar_NIGAM(data_acc_mean_all_2, best_parameters_Nigam_thresholding)
        best_threshold2 = vect_bar_THRESHOLD(data_acc_mean_all_2, best_parameters_Nigam_thresholding)
        if best_parameters_Nigam_thresholding:
            print('FEATURE 3: ' + database)
        best_probs3 = vect_bar_NIGAM(data_acc_mean_all_3, best_parameters_Nigam_thresholding)
        best_threshold3 = vect_bar_THRESHOLD(data_acc_mean_all_3, best_parameters_Nigam_thresholding)

        if graph_acc:
            print(
                '\nANALYSIS WILCOXON (CONFIDENCE LEVEL 95%) shows the accuracy of two classifiers that come from the same distribution (' + database + '): ')
        if graph_acc or analysis_time:
            print('\nFEATURE 1: ' + database)
        _, data_acc_mean_best1, data_acc_best1, _ = uploadResultsDatasets(
            folder, database, 1, times, shotStart, analysis_time=analysis_time, graph_acc=graph_acc,
            best_probs=best_probs1, best_threshold=best_threshold1)
        data_frame_database = data_frame_database.append(data_acc_best1, ignore_index=True, sort=False)

        if graph_acc or analysis_time:
            print('\nFEATURE 2: ' + database)
        _, data_acc_mean_best2, data_acc_best2, _ = uploadResultsDatasets(
            folder, database, 2, times, shotStart, analysis_time=analysis_time, graph_acc=graph_acc,
            best_probs=best_probs2, best_threshold=best_threshold2)
        data_frame_database = data_frame_database.append(data_acc_best2, ignore_index=True, sort=False)

        if graph_acc or analysis_time:
            print('\nFEATURE 3: ' + database)
        _, data_acc_mean_best3, data_acc_best3, _ = uploadResultsDatasets(
            folder, database, 3, times, shotStart, analysis_time=analysis_time, graph_acc=graph_acc,
            best_probs=best_probs3, best_threshold=best_threshold3)
        data_frame_database = data_frame_database.append(data_acc_best3, ignore_index=True, sort=False)

        data_frame_total = data_frame_total.append(data_frame_database, ignore_index=True, sort=False)

        ##### graph best acc
        vect = []
        vect = vect_bar_ours_all(vect, data_acc_mean_best1, best_probs1, best_threshold1)
        vect = vect_bar_ours_all(vect, data_acc_mean_best2, best_probs2, best_threshold2)
        vect = vect_bar_ours_all(vect, data_acc_mean_best3, best_probs3, best_threshold3)

        classifierSet_names_all = ['Initial classifier']
        classifierSet_names_all.append('Nigam-based classifier')
        classifierSet_names_all.append('Thresholding-based classifier')
        classifierSet_names_all.append('Our online classifier using pseudo-labels')
        classifierSet_names_all.append('Our online classifier using labels')

        if graph_acc:
            vect = np.array(vect)
            ax[idx].grid(axis='y', color='gainsboro', linewidth=1, zorder=1)
            X = np.arange(2 * 3)

            slide = 0.1
            list_colors = ['tab:gray', 'tab:orange', 'tab:green', 'tab:red', 'tab:blue']
            for i in range(len(classifierSet_names_all)):
                ax[idx].bar(X + (slide + 0.009) * i, vect[:, i], width=slide, zorder=i + 2, color=list_colors[i])

                if idx == 0:
                    ax[idx].set_yticks(np.arange(30, 76, step=5))
                    ax[idx].set_ylim([30, 76])
                elif idx == 1:
                    ax[idx].set_yticks(np.arange(70, 101, step=5))
                    ax[idx].set_ylim([70, 101])
                elif idx == 2:
                    ax[idx].set_yticks(np.arange(80, 101, step=5))
                    ax[idx].set_ylim([80, 101])
                elif idx == 3:
                    ax[idx].set_yticks(np.arange(60, 91, step=5))
                    ax[idx].set_ylim([60, 91])
                elif idx == 4:
                    ax[idx].set_yticks(np.arange(55, 91, step=5))
                    ax[idx].set_ylim([55, 91])

            ax[idx].set_title(database_names[idx])
            ax[idx].set_ylabel('accuracy [%]')

        idx += 1

    if friedman:
        print('\n\nFRIEDMAN TOTAL (CONFIDENCE LEVEL 95%) soft-labelling techniques')
        frame = pd.DataFrame()
        for type_DA in ['LDA', 'QDA']:
            print('\nType DA classifier: ' + type_DA)
            for type_model in ['ours']:
                classifierSet_names = []
                for type_updating in ['soft_labels', 'best_probs', 'best_threshold']:
                    classifierSet_names.append(type_DA + '_' + type_model + '_' + type_updating)

                aux_frame = data_frame_total[classifierSet_names]
                aux_frame.columns = ['ours_soft_labelling', 'Nigam\'s technique', 'thresholding']
                frame = frame.append(aux_frame, ignore_index=True, sort=False)

            friedman_analysis(frame)

        print('\n\nFRIEDMAN TOTAL (CONFIDENCE LEVEL 95%) batch classifiers')

        for type_updating in ['labels', 'soft_labels']:
            for type_DA in ['LDA', 'QDA']:
                frame = pd.DataFrame()
                classifierSet_names = []
                print('\nType DA classifier: ' + type_DA)
                for type_model in ['ours', 'state_art']:
                    classifierSet_names.append(type_DA + '_' + type_model + '_' + type_updating)
                aux_frame = data_frame_total[classifierSet_names]
                aux_frame.columns = ['weighted classifier', 'traditional classifier']
                frame = frame.append(aux_frame, ignore_index=True, sort=False)
                print(type_updating)
                friedman_analysis(frame)

    if graph_acc:
        plt.xticks(X + (slide + 0.009) * 2,
                   ('LDA FS1', 'QDA FS1', 'LDA FS2', 'QDA FS2', 'LDA FS3', 'QDA FS3'))
        ax[idx - 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
                           fancybox=True, shadow=True, labels=classifierSet_names_all, ncol=2)
        plt.tight_layout()
        # plt.savefig('Images/acc2.png', dpi=300)
        plt.show()


# %% GRAPH Batch classifiers
def experiment2():
    vect_LDA_soft = []
    vect_QDA_soft = []
    vect_LDA_label = []
    vect_QDA_label = []
    for database in ['Nina5', 'Capgmyo_dbb', 'Cote', 'LongTerm3DC', 'EPN_120']:


        print(
            '\nANALYSIS WILCOXON (CONFIDENCE LEVEL 95%) shows the accuracy of two classifiers that come from the same distribution (' + database + '): ')

        print('\nFEATURE 1: ' + database)
        _, _, _, data_acc_improve1 = uploadResultsDatasets(
            folder, database, 1, times, shotStart, graph_acc_batch=True, best_probs=0, best_threshold=0)

        print('\nFEATURE 2: ' + database)
        _, _, _, data_acc_improve2 = uploadResultsDatasets(
            folder, database, 2, times, shotStart, graph_acc_batch=True, best_probs=0, best_threshold=0)

        print('\nFEATURE 3: ' + database)
        _, _, _, data_acc_improve3 = uploadResultsDatasets(
            folder, database, 3, times, shotStart, graph_acc_batch=True, best_probs=0, best_threshold=0)

        ##### graph best acc
        type_label = 'soft_labels'
        vect_LDA_soft, vect_QDA_soft = vect_bar_improve(vect_LDA_soft, vect_QDA_soft, data_acc_improve1, type_label)
        vect_LDA_soft, vect_QDA_soft = vect_bar_improve(vect_LDA_soft, vect_QDA_soft, data_acc_improve2, type_label)
        vect_LDA_soft, vect_QDA_soft = vect_bar_improve(vect_LDA_soft, vect_QDA_soft, data_acc_improve3, type_label)

        type_label = 'labels'
        vect_LDA_label, vect_QDA_label = vect_bar_improve(vect_LDA_label, vect_QDA_label, data_acc_improve1, type_label)
        vect_LDA_label, vect_QDA_label = vect_bar_improve(vect_LDA_label, vect_QDA_label, data_acc_improve2, type_label)
        vect_LDA_label, vect_QDA_label = vect_bar_improve(vect_LDA_label, vect_QDA_label, data_acc_improve3, type_label)

    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(9, 72 / 7), sharex=True)
    vect_list = [vect_LDA_label, vect_QDA_label]
    classifierSet_names = []
    list_colors = ['k', 'tab:blue']
    for type_model in [['Traditional classifier', 0], ['Weighted classifier', 1]]:
        classifierSet_names.append(type_model[0])
    X = np.arange(3 * 5)
    slide = 0.25
    name_DA=['LDA','QDA','LDA','QDA']
    for j in range(2):
        vect = vect_list[j]
        vect = np.array(vect)
        ax[j].grid(axis='y', color='gainsboro', linewidth=1, zorder=1)
        for i in range(len(classifierSet_names)):
            ax[j].bar(X + (slide + 0.01) * i, vect[:, i], width=slide, zorder=i + 2, color=list_colors[i])

        ax[j].set_ylabel(name_DA[j]+'\naccuracy difference[%]')

    ax[0].set_title('Labeled Gestures')

    vect_list = [vect_LDA_soft, vect_QDA_soft]
    classifierSet_names = []
    for type_model in [['Traditional classifier', 0], ['Weighted classifier', 1]]:
        classifierSet_names.append(type_model[0])
    X = np.arange(3 * 5)
    slide = 0.25
    for j in range(2, 4):
        vect = vect_list[j - 2]
        vect = np.array(vect)
        ax[j].grid(axis='y', color='gainsboro', linewidth=1, zorder=1)

        for i in range(len(classifierSet_names)):
            ax[j].bar(X + (slide + 0.01) * i, vect[:, i], width=slide, zorder=i + 2, color=list_colors[i])

        ax[j].set_ylabel(name_DA[j]+'\naccuracy difference[%]')

    ax[2].set_title('Pseudo-Labeled Gestures')
    ax[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
                 fancybox=True, shadow=True, labels=classifierSet_names, ncol=2)
    plt.xticks(X + (slide + 0.01) * 0.5,
               (
                   'FS1', 'FS2', 'FS3', 'FS1', 'FS2', 'FS3', 'FS1', 'FS2', 'FS3', 'FS1', 'FS2', 'FS3', 'FS1', 'FS2',
                   'FS3'))
    plt.tight_layout()
    plt.savefig('Images/acc_improve.png', dpi=300)
    plt.show()

# experiment2()

