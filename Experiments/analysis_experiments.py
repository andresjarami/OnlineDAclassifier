# %%

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FormatStrFormatter
from scipy import stats
import matplotlib.pyplot as plt
import models as models


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

def uploadResults(folder, database, samplesTrain, people, times, featureSet, classes, days, shotStart, best_probs,
                  best_threshold):
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
                a = 3
                # print('error' + '_time' + str(time) + '_person' + str(person) + '_FS' + str(featureSet))

    data_acc_mean_all = get_data_acc_mean_all(dataFrame, gestureSet)
    if best_probs != 0 and best_threshold != 0:
        # data_acc_best = get_data_acc_best(dataFrame, gestureSet, best_probs, best_threshold)
        data_acc_mean_best = get_data_acc_mean_ours_all(dataFrame, gestureSet, best_probs, best_threshold)
        data_acc_best = get_data_acc_ours_all(dataFrame, gestureSet, best_probs, best_threshold)

    else:
        data_acc_mean_best = 0
        data_acc_best = 0
        get_data_time(dataFrame, gestureSet)
    data_acc_mean_improve = get_data_acc_mean_improve(dataFrame, gestureSet, best_probs, best_threshold)
    # data_acc_mean_batch = get_data_acc_mean_batch(dataFrame, gestureSet)
    # data_acc_batch = get_data_acc_batch(dataFrame, gestureSet)

    # data_acc = get_data_acc(dataFrame, gestureSet)

    # data_time_update = get_data_time_update(dataFrame, gestureSet)
    # data_time_update = np.vstack((np.ones((1, np.size(data_time_update, axis=1))) * 0, data_time_update))

    # data_time_weight = get_data_time_weight(dataFrame, gestureSet)
    # data_weight = get_data_weight(dataFrame, gestureSet, classes, database)
    # data_weight = np.vstack((np.ones((1, np.size(data_weight, axis=1))) * 0, data_weight))

    return data_acc_mean_all, data_acc_mean_best, data_acc_best, data_acc_mean_improve, 0


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


def get_data_acc_mean_ours_all(dataFrame, gestureSet, best_probs, best_threshold):
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

            if classifier != type_DA[0] + '_' + 'weak':
                _, p = stats.wilcoxon(weak_vect, vect)

                if p > p_value_control:
                    print('p value', p)
                    print(
                        type_DA[
                            0] + '_initial_classifier and ' + classifier + ' come from the same distribution')

            if classifier != type_DA[0] + '_' + type_model[0] + '_soft_labels':
                _, p = stats.wilcoxon(vect, ours_soft_vect)

                if p > p_value_control:
                    print('p value', p)
                    print(
                        type_DA[0] + '_' + type_model[
                            0] + '_soft_labels and ' + classifier + ' and comes from the same distribution')

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

            print('time: '+classifier,round(vect.mean()*1000,2),round(vect.std()*1000,2))




def get_data_acc_ours_all(dataFrame, gestureSet, best_probs, best_threshold):
    metric = 'acc'
    data = pd.DataFrame()

    for type_DA in [['LDA', 0], ['QDA', 1]]:
        classifierSet = [type_DA[0] + '_' + 'weak']
        classifierSet_names = [type_DA[0] + '_' + 'weak']
        for type_model in [['ours', 0],['state_art', 1]]:
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_soft_labels')
            classifierSet_names.append(type_DA[0] + '_' + type_model[0] + '_soft_labels')

            if type_model[0] =='ours':
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

    # for type_DA in ['LDA', 'QDA']:
    #     for type_model in ['best_probs', 'best_threshold']:
    #         _, p = stats.wilcoxon(data[type_DA + '_ours_soft_labels'], data[type_DA + '_ours_' + type_model])
    #
    #         if p > p_value_control:
    #             print(
    #                 type_DA + '_ours_' + type_model + ' and ' + type_DA + '_ours_soft_labels and comes from the same distribution')

    # print('\n\nFRIEDMAN TOTAL (CONFIDENCE LEVEL 95%) soft-labelling techniques')
    # frame = pd.DataFrame()
    # for type_DA in ['LDA', 'QDA']:
    #     print('\nType DA classifier: ' + type_DA)
    #     for type_model in ['ours']:
    #         classifierSet_names = []
    #         for type_updating in ['soft_labels', 'best_probs', 'best_threshold']:
    #             classifierSet_names.append(type_DA + '_' + type_model + '_' + type_updating)
    #
    #         aux_frame = data[classifierSet_names]
    #         aux_frame.columns = ['ours_soft_labelling', 'best_probs', 'best_threshold']
    #         frame = frame.append(aux_frame, ignore_index=True, sort=False)
    #
    #     friedman_analysis(frame)

    return data


def get_data_acc_mean_improve(dataFrame, gestureSet, best_probs, best_threshold):
    metric = 'acc'
    p_value_control = 0.05

    data = pd.DataFrame()
    for type_DA in [['LDA', 0], ['QDA', 1]]:
        classifierSet = []
        for type_model in [['ours', 0], ['state_art', 1]]:
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_soft_labels')
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_labels')
            # classifierSet.append(
            #     type_DA[0] + '_' + type_model[0] + '_probs_' + str(best_probs[type_model[1]][type_DA[1]]))
            # classifierSet.append(
            #     type_DA[0] + '_' + type_model[0] + '_threshold_' + str(best_threshold[type_model[1]][type_DA[1]]))

        weak_vect = dataFrame[metric + '_' + type_DA[0] + '_' + 'weak'].loc[
            dataFrame['number gesture'] == gestureSet[len(gestureSet) - 1]]
        soft_label_state_vect = dataFrame[metric + '_' + type_DA[0] + '_state_art_soft_labels'].loc[
            dataFrame['number gesture'] == gestureSet[len(gestureSet) - 1]]
        label_state_vect = dataFrame[metric + '_' + type_DA[0] + '_state_art_labels'].loc[
            dataFrame['number gesture'] == gestureSet[len(gestureSet) - 1]]
        for classifier in classifierSet:
            vect = dataFrame[metric + '_' + classifier].loc[
                dataFrame['number gesture'] == gestureSet[len(gestureSet) - 1]]

            if classifier == type_DA[0] + '_ours_soft_labels':
                _, p = stats.wilcoxon(vect, soft_label_state_vect)

                if p > p_value_control:
                    print('p value', p)
                    print(
                        type_DA[
                            0] + '_state_art_soft_labels and ' + classifier + ' come from the same distribution')

            if classifier == type_DA[0] + '_ours_labels':
                _, p = stats.wilcoxon(vect, label_state_vect)

                if p > p_value_control:
                    print('p value', p)
                    print(
                        type_DA[
                            0] + '_state_art_labels and ' + classifier + ' come from the same distribution')


            data.at[0, classifier] = vect.mean() - weak_vect.mean()

    return data


def get_data_acc_best(dataFrame, gestureSet, best_probs, best_threshold):
    metric = 'acc'
    data = pd.DataFrame()

    for type_DA in [['LDA', 0], ['QDA', 1]]:
        classifierSet = []
        classifierSet_names = []
        for type_model in [['ours', 0], ['state_art', 1]]:
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_soft_labels')
            classifierSet_names.append(type_DA[0] + '_' + type_model[0] + '_soft_labels')

            classifierSet.append(
                type_DA[0] + '_' + type_model[0] + '_probs_' + str(best_probs[type_model[1]][type_DA[1]]))
            classifierSet_names.append(type_DA[0] + '_' + type_model[0] + '_best_probs')

            classifierSet.append(
                type_DA[0] + '_' + type_model[0] + '_threshold_' + str(best_threshold[type_model[1]][type_DA[1]]))
            classifierSet_names.append(type_DA[0] + '_' + type_model[0] + '_best_threshold')

        for i in range(len(classifierSet)):
            vect = dataFrame[metric + '_' + classifierSet[i]].loc[
                dataFrame['number gesture'] == gestureSet[len(gestureSet) - 1]]
            data[classifierSet_names[i]] = vect

    return data


def get_data_acc_mean_batch(dataFrame, gestureSet):
    metric = 'acc'
    p_value_control = 0.05

    data = pd.DataFrame()
    for type_DA in [['LDA', 0], ['QDA', 1]]:
        classifierSet = [type_DA[0] + '_' + 'weak', type_DA[0] + '_' + 'batch']
        for type_model in [['ours', 0]]:
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_labels')

        batch_vect = dataFrame[metric + '_' + type_DA[0] + '_' + 'batch'].loc[
            dataFrame['number gesture'] == gestureSet[len(gestureSet) - 1]]
        for classifier in classifierSet:
            vect = dataFrame[metric + '_' + classifier].loc[
                dataFrame['number gesture'] == gestureSet[len(gestureSet) - 1]]

            if classifier == type_DA[0] + '_' + 'ours' + '_labels':
                _, p = stats.wilcoxon(vect, batch_vect)

                if p < p_value_control:
                    print(
                        type_DA[0]
                        + ' the mean of the accuracy of the batch and our classifier is different (wilcoxon)')
                    if vect.mean() > batch_vect.mean():
                        print('best ours', vect.mean(), batch_vect.mean())
                    else:
                        print('best batch', vect.mean(), batch_vect.mean())

            data.at[0, classifier] = vect.mean()

    return data


def get_data_acc_batch(dataFrame, gestureSet):
    metric = 'acc'
    data = pd.DataFrame()

    for type_DA in [['LDA', 0], ['QDA', 1]]:
        classifierSet = [type_DA[0] + '_' + 'weak', type_DA[0] + '_' + 'batch']
        for type_model in [['ours', 0]]:
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_labels')

        for i in range(len(classifierSet)):
            vect = dataFrame[metric + '_' + classifierSet[i]].loc[
                dataFrame['number gesture'] == gestureSet[len(gestureSet) - 1]]
            data[classifierSet[i]] = vect

    return data


def get_data_weight(dataFrame, gestureSet, classes, database):
    metric = 'weight'

    if database == 'Nina5':
        b_wp = 0.6
        b_t = 0.1
    elif database == 'Cote':
        b_wp = 1.0
        b_t = 0.4
    elif database == 'LongTerm3DC':
        b_wp = 1.0
        b_t = 0.7
    elif database == 'Capgmyo_dbb':
        b_wp = 1.0
        b_t = 0.5

    classifierSet = ['incre_proposed_semi']
    classifierSet.append('incre_weight_postprobability_' + str(b_wp))
    classifierSet.append('incre_threshold_' + str(b_t))
    numberClassifiers = 6
    data = np.zeros((len(gestureSet), numberClassifiers))
    typeDA = 'LDA'
    for classifier in range(numberClassifiers):
        if classifier >= numberClassifiers / 2:
            typeDA = 'QDA'
            classifier_aux = classifier - int(numberClassifiers / 2)
        else:
            classifier_aux = classifier
        idx = 0
        for gesture in gestureSet:
            arrays_string = dataFrame[metric + '_' + typeDA + '_' + classifierSet[classifier_aux]].loc[
                dataFrame['unlabeled Gesture'] == gesture].sum()
            arrays_string = np.array(re.split(']\[|]|\[', arrays_string))
            classifier_weights = np.zeros((len(arrays_string) - 2, classes))
            for i in range(1, len(arrays_string) - 1):
                aux = np.array(arrays_string[i].split())
                classifier_weights[i - 1, :] = aux.astype(np.float)

            arrays_string = dataFrame[metric + '_' + typeDA + '_' + 'incre_proposed_sup'].loc[
                dataFrame['unlabeled Gesture'] == gesture].sum()
            arrays_string = np.array(re.split(']\[|]|\[', arrays_string))
            real_weights = np.zeros((len(arrays_string) - 2, classes))
            for i in range(1, len(arrays_string) - 1):
                aux = np.array(arrays_string[i].split())
                real_weights[i - 1, :] = aux.astype(np.float)

            data[idx, classifier] = models.errorWeights_type2_mse(classifier_weights, real_weights)
            idx += 1
    aux_data = data.copy()
    ###accumulated error
    for i in range(len(data)):
        data[i, :] = aux_data[:i + 1, :].sum(axis=0)
    return data


def get_data_time_weight(dataFrame, gestureSet):
    metric = 'time_weight'
    numberClassifiers = 6

    classifierSet = ['LDA_incre_proposed_semi', 'incre_weight_postprobability_LDA', 'incre_threshold_LDA',
                     'QDA_incre_proposed_semi', 'incre_weight_postprobability_QDA', 'incre_threshold_QDA']

    data = np.zeros((len(gestureSet), numberClassifiers))
    typeDA = 'LDA'
    for classifier in range(numberClassifiers):
        if classifier >= numberClassifiers / 2:
            typeDA = 'QDA'
            classifier_aux = classifier - int(numberClassifiers / 2)
        else:
            classifier_aux = classifier
        idx = 0
        for gesture in gestureSet:
            data[idx, classifier] = \
                dataFrame[metric + '_' + classifierSet[classifier_aux]].loc[
                    dataFrame['unlabeled Gesture'] == gesture].mean()
            idx += 1
    return data


def get_data_time_update(dataFrame, gestureSet):
    metric = 'time_update'
    numberClassifiers = 6

    classifierSet = ['batch', 'incre_proposed_sup', 'incre_proposed_semi']

    data = np.zeros((len(gestureSet), numberClassifiers))
    typeDA = 'LDA'
    for classifier in range(numberClassifiers):
        if classifier >= numberClassifiers / 2:
            typeDA = 'QDA'
            classifier_aux = classifier - int(numberClassifiers / 2)
        else:
            classifier_aux = classifier
        idx = 0
        for gesture in gestureSet:
            data[idx, classifier] = \
                dataFrame[metric + '_' + typeDA + '_' + classifierSet[classifier_aux]].loc[
                    dataFrame['unlabeled Gesture'] == gesture].mean()
            idx += 1
    return data


def get_data_acc_mean_best_weak(dataFrame, gestureSet, best_probs, best_threshold):
    metric = 'acc'

    data = pd.DataFrame()
    for type_DA in [['LDA', 0], ['QDA', 1]]:
        classifierSet = [type_DA[0] + '_' + 'weak']
        for type_model in [['ours', 0], ['state_art', 1]]:
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_soft_labels')
            classifierSet.append(
                type_DA[0] + '_' + type_model[0] + '_probs_' + str(best_probs[type_model[1]][type_DA[1]]))
            classifierSet.append(
                type_DA[0] + '_' + type_model[0] + '_threshold_' + str(best_threshold[type_model[1]][type_DA[1]]))

        for classifier in classifierSet:
            vect = dataFrame[metric + '_' + classifier].loc[
                dataFrame['number gesture'] == gestureSet[len(gestureSet) - 1]]

            data.at[0, classifier] = vect.mean()

    return data


def uploadResultsDatabasesVF1(folder, database, featureSet, times, shotStart, best_probs=0, best_threshold=0):
    if database == 'Nina5':
        samplesTrain = 4
        samplesTest = 2
        classes = 18
        people = 10
        days = 1
    elif database == 'Cote':
        samplesTrain = 9
        samplesTest = 3
        classes = 7
        people = 17
        days = 1
    elif database == 'EPN_120':
        samplesTrain = 25
        samplesTest = 25
        classes = 5
        people = 120
        days = 1
    elif database == 'Capgmyo_dbb':
        samplesTrain = 7
        samplesTest = 3
        classes = 8
        people = 10
        days = 2
    elif database == 'LongTerm3DC':
        samplesTrain = 2
        samplesTest = 1
        classes = 11
        people = 20
        days = 3
    return uploadResults(folder, database, samplesTrain, people, times, featureSet, classes, days, shotStart,
                         best_probs, best_threshold)


def plot_arrays(data, database, ax, type_DA, classifierSet, x_ticks, ylabel, gestureSet):
    for classifier in classifierSet:
        ax.plot(gestureSet, data[type_DA + '_' + classifier], label=type_DA + '_' + classifier)
        ax.set_xticks(x_ticks)
        ax.grid(color='gainsboro', linewidth=1)
    ax.set_ylabel(ylabel + ' (' + database + ')')


def plot_bar(data, database, ax, type_DA, classifierSet, x_ticks, ylabel, gestureSet):
    ax.bar()
    for classifier in classifierSet:
        ax.plot(gestureSet, data[type_DA + '_' + classifier], label=type_DA + '_' + classifier)
        ax.set_xticks(x_ticks)
        ax.grid(color='gainsboro', linewidth=1)
    ax.set_ylabel(ylabel + ' (' + database + ')')


def print_ACC(data, type_DA, classifierSet):
    values = []
    for classifier in classifierSet:
        values.append(data.loc[len(data) - 1, type_DA + '_' + classifier])
    values = np.array(values)

    aux_values = values[1:int((len(values) - 1) / 2) + 1]
    aux_classifierSet = classifierSet[1:int((len(values) - 1) / 2) + 1]
    idx = np.argmax(aux_values)
    print('best classifier using ours updating: ', aux_classifierSet[idx], ', ACC =', aux_values[idx])

    aux_values = values[int((len(values) - 1) / 2) + 1:]
    aux_classifierSet = classifierSet[int((len(values) - 1) / 2) + 1:]
    idx = np.argmax(aux_values)
    print('best classifier using state-of-the-art updating: ', aux_classifierSet[idx], ', ACC = ', aux_values[idx])


def set_title_labeles_legend(ax):
    ax[0, 0].set_title('LDA')
    ax[0, 1].set_title('QDA')
    ax[3, 0].set_xlabel('Number of Labled/Unlabeled sets')
    ax[3, 1].set_xlabel('Number of Labled/Unlabeled sets')
    ax[3, 1].legend()


def vect_bar(vect, data, best_probs, best_threshold):
    for type_DA in [['LDA', 0], ['QDA', 1]]:
        classifierSet = [type_DA[0] + '_' + 'weak']
        for type_model in [['ours', 0], ['state_art', 1]]:
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_soft_labels')
            classifierSet.append(
                type_DA[0] + '_' + type_model[0] + '_probs_' + str(best_probs[type_model[1]][type_DA[1]]))
            classifierSet.append(
                type_DA[0] + '_' + type_model[0] + '_threshold_' + str(best_threshold[type_model[1]][type_DA[1]]))

        vect.append(list(data[classifierSet].loc[len(data) - 1].values * 100))
    return vect


def vect_bar_soft_labeling(vect, data, best_probs, best_threshold):
    for type_DA in [['LDA', 0], ['QDA', 1]]:
        classifierSet = []
        for type_model in [['ours', 0], ['state_art', 1]]:
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_soft_labels')
            classifierSet.append(
                type_DA[0] + '_' + type_model[0] + '_probs_' + str(best_probs[type_model[1]][type_DA[1]]))
            classifierSet.append(
                type_DA[0] + '_' + type_model[0] + '_threshold_' + str(best_threshold[type_model[1]][type_DA[1]]))

        vect.append(list(data[classifierSet].loc[len(data) - 1].values * 100))
    return vect


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


def vect_bar_improve(vect_LDA, vect_QDA, data,type_label):
    for type_DA in [['LDA', 0], ['QDA', 1]]:
        classifierSet = []
        for type_model in [['state_art', 0],['ours', 0]]:
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_'+type_label)
        if type_DA[0] == 'LDA':
            vect_LDA.append(list(data[classifierSet].loc[len(data) - 1].values * 100))
        else:
            vect_QDA.append(list(data[classifierSet].loc[len(data) - 1].values * 100))
    return vect_LDA, vect_QDA




def vect_bar_labels(vect, data):
    for type_DA in [['LDA', 0], ['QDA', 1]]:
        classifierSet = [type_DA[0] + '_' + 'weak', type_DA[0] + '_' + 'batch']
        for type_model in [['ours', 0]]:
            classifierSet.append(type_DA[0] + '_' + type_model[0] + '_labels')
        vect.append(list(data[classifierSet].loc[len(data) - 1].values * 100))
    return vect


def vect_bar_PROBS_batch_best(data_acc):
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
            print('Best classifier (posterior probability) is ' + classifierSet[idx] + '. Improve acc=' + str(
                round(values[idx] * 100, 2)))
            aux_best_parameters.append(list_weight[idx])
        best_parameters.append(aux_best_parameters)
    return best_parameters


def vect_bar_THRESHOLD_batch_best(data_acc):
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
            print('Best classifier (threshold) is ' + classifierSet[idx] + '. Improve acc=' + str(
                round(values[idx] * 100, 2)))
            aux_best_parameters.append(list_threshold[idx])
        best_parameters.append(aux_best_parameters)
    return best_parameters


def vect_bar_PROBS(data_acc):
    best_parameters = []
    for type_DA in [['LDA', 0], ['QDA', 1]]:
        aux_best_parameters = []

        for type_model in [['ours', 0], ['state_art', 1]]:
            classifierSet = []
            list_weight = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            for weight in list_weight:
                classifierSet.append(type_DA[0] + '_' + type_model[0] + '_probs_' + str(weight))

            values = data_acc[classifierSet].loc[len(data_acc) - 1].values
            idx = np.argmax(values)
            print('Best classifier (posterior probability) is ' + classifierSet[idx] + '. Improve acc=' + str(
                round(values[idx] * 100, 2)))
            aux_best_parameters.append(list_weight[idx])
        best_parameters.append(aux_best_parameters)
    return best_parameters


def vect_bar_THRESHOLD(data_acc):
    best_parameters = []
    for type_DA in [['LDA', 0], ['QDA', 1]]:
        aux_best_parameters = []
        for type_model in [['ours', 0], ['state_art', 1]]:
            classifierSet = []
            list_threshold = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for threshold in list_threshold:
                classifierSet.append(type_DA[0] + '_' + type_model[0] + '_threshold_' + str(threshold))

            values = data_acc[classifierSet].loc[len(data_acc) - 1].values
            idx = np.argmax(values)
            print('Best classifier (threshold) is ' + classifierSet[idx] + '. Improve acc=' + str(
                round(values[idx] * 100, 2)))
            aux_best_parameters.append(list_threshold[idx])
        best_parameters.append(aux_best_parameters)
    return best_parameters


# %% set initial variables

featureSet = 1
times = 2
windowSize = 290
shotStart = 1

print(times)
# %% GRAPH PAPER ACC IMPROVE

vect_LDA_soft = []
vect_QDA_soft = []
vect_LDA_label = []
vect_QDA_label = []
for database in ['Nina5', 'Capgmyo_dbb', 'Cote', 'LongTerm3DC', 'EPN_120']:
    print('\n' + database)
    folder = '../Results/final3/'
    data_frame_database = pd.DataFrame()

    print('\nANALYSIS WILCOXON (CONFIDENCE LEVEL 95%): ' + database)

    print('\nFEATURE 1: ' + database)
    _, _, _, data_acc_improve1, _ = uploadResultsDatabasesVF1(
        folder, database, 1, times, shotStart, best_probs=0, best_threshold=0)

    print('\nFEATURE 2: ' + database)
    _, _, _, data_acc_improve2, _ = uploadResultsDatabasesVF1(
        folder, database, 2, times, shotStart, best_probs=0, best_threshold=0)

    print('\nFEATURE 3: ' + database)
    _, _, _, data_acc_improve3, _ = uploadResultsDatabasesVF1(
        folder, database, 3, times, shotStart, best_probs=0, best_threshold=0)

    ##### graph best acc
    type_label='soft_labels'
    vect_LDA_soft, vect_QDA_soft = vect_bar_improve(vect_LDA_soft, vect_QDA_soft, data_acc_improve1,type_label)
    vect_LDA_soft, vect_QDA_soft = vect_bar_improve(vect_LDA_soft, vect_QDA_soft, data_acc_improve2,type_label)
    vect_LDA_soft, vect_QDA_soft = vect_bar_improve(vect_LDA_soft, vect_QDA_soft, data_acc_improve3,type_label)

    type_label = 'labels'
    vect_LDA_label, vect_QDA_label = vect_bar_improve(vect_LDA_label, vect_QDA_label, data_acc_improve1, type_label)
    vect_LDA_label, vect_QDA_label = vect_bar_improve(vect_LDA_label, vect_QDA_label, data_acc_improve2, type_label)
    vect_LDA_label, vect_QDA_label = vect_bar_improve(vect_LDA_label, vect_QDA_label, data_acc_improve3, type_label)


fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(9, 72/7), sharex=True)
# fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(14, 16), sharex=True)
vect_list = [vect_LDA_label, vect_QDA_label]
classifierSet_names = []
list_colors = ['k', 'tab:blue']
for type_model in [['Traditional classifier', 0], ['Weighted classifier', 1]]:
    classifierSet_names.append(type_model[0])
X = np.arange(3 * 5)
slide = 0.25
for j in range(2):
    vect=vect_list[j]

    vect = np.array(vect)
    ax[j].grid(axis='y', color='gainsboro', linewidth=1, zorder=1)


    for i in range(len(classifierSet_names)):
        ax[j].bar(X + (slide + 0.01) * i, vect[:, i], width=slide, zorder=i + 2, color=list_colors[i])

    # ax[i].set_yticks(np.arange(30, 76, step=5))
    # ax[i].set_ylim([30, 76])

    ax[j].set_ylabel('accuracy difference[%]')

ax[0].set_title('Labeled Gestures')
# plt.xticks(X + (slide + 0.01) * 0.5,
#            ('FS1', 'FS2', 'FS3', 'FS1', 'FS2', 'FS3','FS1', 'FS2', 'FS3','FS1', 'FS2', 'FS3','FS1', 'FS2', 'FS3'))
# ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
#              fancybox=True, shadow=True, labels=classifierSet_names, ncol=2)
# plt.savefig('../images/acc_improve1.png', dpi=300)
# plt.show()



vect_list = [vect_LDA_soft, vect_QDA_soft]
classifierSet_names = []
for type_model in [['Traditional classifier', 0], ['Weighted classifier', 1]]:
    classifierSet_names.append(type_model[0])
X = np.arange(3 * 5)
slide = 0.25
for j in range(2,4):
    vect=vect_list[j-2]

    vect = np.array(vect)
    ax[j].grid(axis='y', color='gainsboro', linewidth=1, zorder=1)



    # list_colors = ['tab:blue', 'k']
    for i in range(len(classifierSet_names)):
        ax[j].bar(X + (slide + 0.01) * i, vect[:, i], width=slide, zorder=i + 2, color=list_colors[i])

    # ax[i].set_yticks(np.arange(30, 76, step=5))
    # ax[i].set_ylim([30, 76])

    ax[j].set_ylabel('accuracy difference[%]')

ax[2].set_title('Pseudo-Labeled Gestures')
ax[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
             fancybox=True, shadow=True, labels=classifierSet_names, ncol=2)
plt.xticks(X + (slide + 0.01) * 0.5,
           ('FS1', 'FS2', 'FS3', 'FS1', 'FS2', 'FS3','FS1', 'FS2', 'FS3','FS1', 'FS2', 'FS3','FS1', 'FS2', 'FS3'))
plt.tight_layout()
plt.savefig('../images/acc_improve.png', dpi=300)
plt.show()


# %% GRAPH PAPER ACC all
data_frame_total = pd.DataFrame()
fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(8, 11), sharex=True)
idx = 0
database_names=['NinaPro5', 'Capgmyo_dbb','MyoArmband','Long-Term 3DC','EMG-EPN-120']

for database in ['Nina5', 'Capgmyo_dbb', 'Cote', 'LongTerm3DC', 'EPN_120']:
    print('\n' + database)
    folder = '../Results/final3/'
    data_frame_database = pd.DataFrame()

    data_acc_mean_all_1, _, _, _, _ = uploadResultsDatabasesVF1(
        folder, database, 1, times, shotStart, best_probs=0, best_threshold=0)

    data_acc_mean_all_2, _, _, _, _ = uploadResultsDatabasesVF1(
        folder, database, 2, times, shotStart, best_probs=0, best_threshold=0)

    data_acc_mean_all_3, _, _, _, _ = uploadResultsDatabasesVF1(
        folder, database, 3, times, shotStart, best_probs=0, best_threshold=0)

    ##### graph best acc
    print('\nFEATURE 1: ' + database)
    best_probs1 = vect_bar_PROBS_batch_best(data_acc_mean_all_1)
    best_threshold1 = vect_bar_THRESHOLD_batch_best(data_acc_mean_all_1)
    print('FEATURE 2: ' + database)
    best_probs2 = vect_bar_PROBS_batch_best(data_acc_mean_all_2)
    best_threshold2 = vect_bar_THRESHOLD_batch_best(data_acc_mean_all_2)
    print('FEATURE 3: ' + database)
    best_probs3 = vect_bar_PROBS_batch_best(data_acc_mean_all_3)
    best_threshold3 = vect_bar_THRESHOLD_batch_best(data_acc_mean_all_3)

    print('\nANALYSIS WILCOXON (CONFIDENCE LEVEL 95%): ' + database)

    print('\nFEATURE 1: ' + database)
    _, data_acc_mean_best1, data_acc_best1, _, _ = uploadResultsDatabasesVF1(
        folder, database, 1, times, shotStart, best_probs=best_probs1, best_threshold=best_threshold1)
    data_frame_database = data_frame_database.append(data_acc_best1, ignore_index=True, sort=False)

    print('\nFEATURE 2: ' + database)
    _, data_acc_mean_best2, data_acc_best2, _, _ = uploadResultsDatabasesVF1(
        folder, database, 2, times, shotStart, best_probs=best_probs2, best_threshold=best_threshold2)
    data_frame_database = data_frame_database.append(data_acc_best2, ignore_index=True, sort=False)

    print('\nFEATURE 3: ' + database)
    _, data_acc_mean_best3, data_acc_best3, _, _ = uploadResultsDatabasesVF1(
        folder, database, 3, times, shotStart, best_probs=best_probs3, best_threshold=best_threshold3)
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

    ax[idx].set_title(database_names[idx] )
    ax[idx].set_ylabel('accuracy [%]')

    idx += 1



print('\n\nFRIEDMAN TOTAL (CONFIDENCE LEVEL 95%) soft-labelling techniques')
frame = pd.DataFrame()
for type_DA in ['LDA', 'QDA']:
    print('\nType DA classifier: ' + type_DA)
    for type_model in ['ours']:
        classifierSet_names = []
        for type_updating in ['soft_labels', 'best_probs', 'best_threshold']:
            classifierSet_names.append(type_DA + '_' + type_model + '_' + type_updating)

        aux_frame = data_frame_total[classifierSet_names]
        aux_frame.columns = ['ours_soft_labelling', 'best_probs', 'best_threshold']
        frame = frame.append(aux_frame, ignore_index=True, sort=False)

    friedman_analysis(frame)


print('\n\nFRIEDMAN TOTAL (CONFIDENCE LEVEL 95%) soft-labelling techniques')

for type_updating in ['labels','soft_labels']:
    for type_DA in ['LDA', 'QDA']:
        frame = pd.DataFrame()
        classifierSet_names = []
        print('\nType DA classifier: ' + type_DA)
        for type_model in ['ours', 'state_art']:
            classifierSet_names.append(type_DA + '_' + type_model + '_' + type_updating)
        aux_frame = data_frame_total[classifierSet_names]
        aux_frame.columns = ['ours', 'state_art']
        frame = frame.append(aux_frame, ignore_index=True, sort=False)
        print(type_updating)
        friedman_analysis(frame)

# p_value_control=0.05
# for type_label in ['soft_labels','labels']:
#     for type_DA in ['LDA', 'QDA']:
#         a=type_DA + '_state_art_'+type_label
#         b=type_DA + '_ours_'+type_label
#
#         if data_frame_total[a].mean()>data_frame_total[b].mean():
#
#             _, p = stats.wilcoxon(data_frame_total[a],data_frame_total[b])
#
#             print('\n'+a+' and '+b+': p value', p)
#             if p < p_value_control:
#                 print(a+' is higher that ' +b+', accuracies:',data_frame_total[a].mean(),data_frame_total[b].mean())
#         elif data_frame_total[a].mean()<data_frame_total[b].mean():
#
#             _, p = stats.wilcoxon(data_frame_total[b],data_frame_total[a])
#
#             print('\n'+b+' and '+a+': p value', p)
#             if p < p_value_control:
#                 print(b+' is higher that ' +a+', accuracies:',data_frame_total[b].mean(),data_frame_total[a].mean())
#


plt.xticks(X + (slide + 0.009) * 2,
           ('LDA FS1', 'QDA FS1', 'LDA FS2', 'QDA FS2', 'LDA FS3', 'QDA FS3'))
ax[idx - 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
                   fancybox=True, shadow=True, labels=classifierSet_names_all, ncol=2)
plt.tight_layout()
plt.savefig('../images/acc2.png', dpi=300)
plt.show()


# %% GRAPH PAPER TIME UPDATE
#
# fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(12, 26))
# idx = 0
# numberClassifiers = 24
# for info in ['Nina5', 'Cote', 'LongTerm3DC', 'Capgmyo_dbb','EPN_612']:
#     if info == 'Nina5':
#         x_ticks = np.hstack((np.arange(0, 40 + 1, 10), np.array([54])))
#     elif info == 'Cote':
#         x_ticks = np.hstack((np.arange(0, 40 + 1, 10), np.array([56])))
#     elif info == 'LongTerm3DC':
#         x_ticks = np.hstack((np.arange(0, 8 + 1, 2), np.array([11])))
#     elif info == 'Capgmyo_dbb':
#         x_ticks = np.hstack((np.arange(0, 40 + 1, 10), np.array([48])))
#     elif info == 'EPN_612':
#         x_ticks = np.hstack((np.arange(0, 120+ 1, 30), np.array([144])))
#
#     folder = '../Results/final/'
#     data_acc, data_time_update, data_time_weight, data_weight, gestureSet = uploadResultsDatabasesVF1(
#         folder, info, featureSet, times, shotStart)
#
#     # Compare all classifiers
#     classifierSet = ['batch', 'OnlineSUP', 'OnlineSEMI']
#     idx_classifierSet = np.arange(6)
#     plot_arrays(data_time_update,idx, ax, classifierSet, idx_classifierSet, x_ticks, 'updating time [s]', gestureSet)
#     print('\nAverage of the updating time over time (streaming gestures over time t) of our online classifiers ('+info+' dataset): ')
#     print('LDA OnlineSUP: ',data_time_update[:,1].mean(),'±',data_time_update[:,1].std())
#     print('QDA OnlineSEMI: ', data_time_update[:, 2].mean(), '±', data_time_update[:, 2].std())
#     print('QDA OnlineSUP: ', data_time_update[:, 4].mean(), '±', data_time_update[:, 4].std())
#     print('QDA OnlineSEMI: ', data_time_update[:, 5].mean(), '±', data_time_update[:, 5].std())
#
#     idx += 1
#
# ax[0, 0].set_title('LDA')
# ax[0, 1].set_title('QDA')
# for i in range(2):
#     ax[4, i].set_xlabel('streaming gestures over time $t$')
#
# # ax[3, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
# #                 fancybox=True, shadow=True, ncol=3)
#
# plt.savefig('../images/updating_time.png', dpi=300)
# plt.show()





# %% GRAPH PAPER ACC soft-labelling
# data_frame_total = pd.DataFrame()
# fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(8, 20), sharex=True)
# idx = 0
# for database in ['Nina5','Capgmyo_dbb', 'Cote', 'LongTerm3DC', 'EPN_120']:
#     print('\n' + database)
#     folder = '../Results/final3/'
#     data_frame_database = pd.DataFrame()
#
#     data_acc_mean_all_1, _, _, _, _ = uploadResultsDatabasesVF1(
#         folder, database, 1, times, shotStart, best_probs=0, best_threshold=0)
#
#     data_acc_mean_all_2, _, _, _, _ = uploadResultsDatabasesVF1(
#         folder, database, 2, times, shotStart, best_probs=0, best_threshold=0)
#
#     data_acc_mean_all_3, _, _, _, _ = uploadResultsDatabasesVF1(
#         folder, database, 3, times, shotStart, best_probs=0, best_threshold=0)
#
#     ##### graph best acc
#     print('\nFEATURE 1: ' + database)
#     best_probs1 = vect_bar_PROBS(data_acc_mean_all_1)
#     best_threshold1 = vect_bar_THRESHOLD(data_acc_mean_all_1)
#     print('FEATURE 2: ' + database)
#     best_probs2 = vect_bar_PROBS(data_acc_mean_all_2)
#     best_threshold2 = vect_bar_THRESHOLD(data_acc_mean_all_2)
#     print('FEATURE 3: ' + database)
#     best_probs3 = vect_bar_PROBS(data_acc_mean_all_3)
#     best_threshold3 = vect_bar_THRESHOLD(data_acc_mean_all_3)
#
#     print('\nANALYSIS WILCOXON (CONFIDENCE LEVEL 95%): ' + database)
#
#     _, data_acc_mean_best1, data_acc_best1, _, _ = uploadResultsDatabasesVF1(
#         folder, database, 1, times, shotStart, best_probs=best_probs1, best_threshold=best_threshold1)
#
#     data_frame_database = data_frame_database.append(data_acc_best1, ignore_index=True, sort=False)
#
#     _, data_acc_mean_best2, data_acc_best2, _, _ = uploadResultsDatabasesVF1(
#         folder, database, 2, times, shotStart, best_probs=best_probs2, best_threshold=best_threshold2)
#     data_frame_database = data_frame_database.append(data_acc_best2, ignore_index=True, sort=False)
#
#     _, data_acc_mean_best3, data_acc_best3, _, _ = uploadResultsDatabasesVF1(
#         folder, database, 3, times, shotStart, best_probs=best_probs3, best_threshold=best_threshold3)
#     data_frame_database = data_frame_database.append(data_acc_best3, ignore_index=True, sort=False)
#
#     data_frame_total = data_frame_total.append(data_frame_database, ignore_index=True, sort=False)
#
#     ##### graph best acc
#     vect = []
#     vect = vect_bar_soft_labeling(vect, data_acc_mean_best1, best_probs1, best_threshold1)
#     vect = vect_bar_soft_labeling(vect, data_acc_mean_best2, best_probs2, best_threshold2)
#     vect = vect_bar_soft_labeling(vect, data_acc_mean_best3, best_probs3, best_threshold3)
#
#     classifierSet_names = []
#     for type_model in [['Ours', 0], ['DR updating', 1]]:
#         classifierSet_names.append(type_model[0] + ' and our soft-labelling')
#         classifierSet_names.append(type_model[0] + ' and best Nigam weight')
#         classifierSet_names.append(type_model[0] + ' and best threshold')
#
#     vect = np.array(vect)
#     ax[idx].grid(axis='y', color='gainsboro', linewidth=1, zorder=1)
#     X = np.arange(2 * 3)
#
#     slide = 0.1
#     list_colors = ['tab:blue','dodgerblue','deepskyblue', 'tab:red', 'indianred','lightcoral'  ]
#     for i in range(len(classifierSet_names)):
#         ax[idx].bar(X + slide * i, vect[:, i], width=slide, zorder=i + 2, color=list_colors[i])
#
#     # if idx == 0:
#     #     ax[idx].set_yticks(np.arange(35, 70, step=5))
#     #     ax[idx].set_ylim([35, 70])
#     # elif idx == 1:
#     #     ax[idx].set_yticks(np.arange(75, 100, step=5))
#     #     ax[idx].set_ylim([80, 100])
#     # elif idx == 2:
#     #     ax[idx].set_yticks(np.arange(55, 85, step=5))
#     #     ax[idx].set_ylim([55, 85])
#     # elif idx == 3:
#     #     ax[idx].set_yticks(np.arange(45, 96, step=10))
#     #     ax[idx].set_ylim([45, 100])
#     # elif idx == 4:
#     #     ax[idx].set_yticks(np.arange(55, 85, step=5))
#     #     ax[idx].set_ylim([55, 85])
#
#     # ax[idx].set_xticks(X + (len(classifierSet_names) - 1) * (slide / 2), ('LDA', 'QDA'))
#     ax[idx].set_title(database)
#     ax[idx].set_ylabel('accuracy [%]')
#
#     idx += 1
#
#     print('\n\nFRIEDMAN PER DATABASE (CONFIDENCE LEVEL 95%): '+database)
#     for type_DA in ['LDA', 'QDA']:
#         print('\nType DA classifier: '+type_DA)
#         classifierSet_names = []
#         for type_model in [['ours', 0], ['state_art', 1]]:
#             classifierSet_names.append(type_DA + '_' + type_model[0] + '_soft_labels')
#             classifierSet_names.append(type_DA + '_' + type_model[0] + '_best_probs')
#             classifierSet_names.append(type_DA + '_' + type_model[0] + '_best_threshold')
#
#         friedman_analysis(data_frame_database[classifierSet_names])
#
#     print('\n\nFRIEDMAN PER DATABASE (CONFIDENCE LEVEL 95%) soft-labelling technique: ' + database)
#     frame = pd.DataFrame()
#     for type_DA in ['LDA', 'QDA']:
#         print('\nType DA classifier: ' + type_DA)
#         for type_model in ['ours', 'state_art']:
#             classifierSet_names = []
#             for type_updating in ['soft_labels', 'best_probs', 'best_threshold']:
#                 classifierSet_names.append(type_DA + '_' + type_model + '_'+type_updating)
#
#             aux_frame = data_frame_database[classifierSet_names]
#             aux_frame.columns = ['ours_soft_labelling','best_probs', 'best_threshold']
#             frame = frame.append(aux_frame, ignore_index=True, sort=False)
#
#         friedman_analysis(frame)
#
#     print('\n\nFRIEDMAN PER DATABASE (CONFIDENCE LEVEL 95%) updating method: ' + database)
#     frame = pd.DataFrame()
#     for type_DA in ['LDA', 'QDA']:
#         print('\nType DA classifier: ' + type_DA)
#         for type_updating in ['soft_labels','best_probs', 'best_threshold']:
#             classifierSet_names = []
#             for type_model in ['ours', 'state_art']:
#                 classifierSet_names.append(type_DA + '_' + type_model + '_'+type_updating)
#
#             aux_frame = data_frame_database[classifierSet_names]
#             aux_frame.columns = ['ours', 'state_art']
#             frame = frame.append(aux_frame, ignore_index=True, sort=False)
#
#         friedman_analysis(frame)
#
# plt.xticks(X + (len(classifierSet_names) - 1) * (slide / 2),
#            ('LDA FS1', 'QDA FS1', 'LDA FS2', 'QDA FS2', 'LDA FS3', 'QDA FS3'))
# # ax[idx].legend(labels=classifierSet_names)
# plt.show()
#
# plt.savefig('../images/acc1.png', dpi=300)
#
# print('\n\nFRIEDMAN TOTAL (CONFIDENCE LEVEL 95%)')
# for type_DA in ['LDA', 'QDA']:
#     classifierSet_names = []
#     for type_model in [['ours', 0], ['state_art', 1]]:
#         classifierSet_names.append(type_DA + '_' + type_model[0] + '_soft_labels')
#         classifierSet_names.append(type_DA + '_' + type_model[0] + '_best_probs')
#         classifierSet_names.append(type_DA + '_' + type_model[0] + '_best_threshold')
#
#     friedman_analysis(data_frame_total[classifierSet_names])
#
#
# print('\n\nFRIEDMAN TOTAL (CONFIDENCE LEVEL 95%) soft-labelling technique')
# frame = pd.DataFrame()
# for type_DA in ['LDA', 'QDA']:
#     print('\nType DA classifier: ' + type_DA)
#     for type_model in ['ours', 'state_art']:
#         classifierSet_names = []
#         for type_updating in ['soft_labels', 'best_probs', 'best_threshold']:
#             classifierSet_names.append(type_DA + '_' + type_model + '_'+type_updating)
#
#         aux_frame = data_frame_total[classifierSet_names]
#         aux_frame.columns = ['ours_soft_labelling','best_probs', 'best_threshold']
#         frame = frame.append(aux_frame, ignore_index=True, sort=False)
#
#     friedman_analysis(frame)
#
# print('\n\nFRIEDMAN TOTAL (CONFIDENCE LEVEL 95%) updating method')
# frame = pd.DataFrame()
# for type_DA in ['LDA', 'QDA']:
#     print('\nType DA classifier: ' + type_DA)
#     for type_updating in ['soft_labels','best_probs', 'best_threshold']:
#         classifierSet_names = []
#         for type_model in ['ours', 'state_art']:
#             classifierSet_names.append(type_DA + '_' + type_model + '_'+type_updating)
#
#         aux_frame = data_frame_total[classifierSet_names]
#         aux_frame.columns = ['ours', 'state_art']
#         frame = frame.append(aux_frame, ignore_index=True, sort=False)
#
#     friedman_analysis(frame)


# %% GRAPH PAPER ACC labels
# data_frame_total = pd.DataFrame()
# fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(8, 20), sharex=True)
# idx = 0
# for database in ['Nina5', 'Capgmyo_dbb', 'Cote', 'LongTerm3DC', 'EPN_120']:
#     print('\n' + database)
#     folder = '../Results/final3/'
#     data_frame_database = pd.DataFrame()
#
#     print('\nANALYSIS WILCOXON (CONFIDENCE LEVEL 95%): ' + database)
#
#     print('Feature set 1')
#     _, _, _, data_acc_mean_batch1, data_acc_batch1 = uploadResultsDatabasesVF1(
#         folder, database, 1, times, shotStart)
#     data_frame_database = data_frame_database.append(data_acc_batch1, ignore_index=True, sort=False)
#
#     print('Feature set 2')
#     _, _, _, data_acc_mean_batch2, data_acc_batch2 = uploadResultsDatabasesVF1(
#         folder, database, 2, times, shotStart)
#     data_frame_database = data_frame_database.append(data_acc_batch2, ignore_index=True, sort=False)
#
#     print('Feature set 3')
#     _, _, _, data_acc_mean_batch3, data_acc_batch3 = uploadResultsDatabasesVF1(
#         folder, database, 3, times, shotStart)
#     data_frame_database = data_frame_database.append(data_acc_batch3, ignore_index=True, sort=False)
#
#     data_frame_total = data_frame_total.append(data_frame_database, ignore_index=True, sort=False)
#
#     ##### graph best acc
#     vect = []
#     vect = vect_bar_labels(vect, data_acc_mean_batch1)
#     vect = vect_bar_labels(vect, data_acc_mean_batch2)
#     vect = vect_bar_labels(vect, data_acc_mean_batch3)
#
#     classifierSet_names = ['weak',  'batch']
#     for type_model in [['Ours', 0]]:
#         classifierSet_names.append(type_model[0] + ' and ours with labels ')
#
#     vect = np.array(vect)
#     ax[idx].grid(axis='y', color='gainsboro', linewidth=1, zorder=1)
#     X = np.arange(2 * 3)
#
#     slide = 0.1
#     list_colors = ['tab:orange','tab:green','tab:blue']
#     for i in range(len(classifierSet_names)):
#         ax[idx].bar(X + slide * i, vect[:, i], width=slide, zorder=i + 2,color=list_colors[i])
#
#     if idx == 0:
#         ax[idx].set_yticks(np.arange(45, 76, step=5))
#         ax[idx].set_ylim([45, 76])
#     elif idx == 1:
#         ax[idx].set_yticks(np.arange(75, 101, step=5))
#         ax[idx].set_ylim([75, 101])
#     elif idx == 2:
#         ax[idx].set_yticks(np.arange(80, 101, step=5))
#         ax[idx].set_ylim([80, 101])
#     elif idx == 3:
#         ax[idx].set_yticks(np.arange(65, 91, step=5))
#         ax[idx].set_ylim([65, 91])
#     elif idx == 4:
#         ax[idx].set_yticks(np.arange(55, 91, step=5))
#         ax[idx].set_ylim([55, 91])
#
#     # ax[idx].set_xticks(X + (len(classifierSet_names) - 1) * (slide / 2), ('LDA', 'QDA'))
#     ax[idx].set_title(database)
#     ax[idx].set_ylabel('accuracy [%]')
#
#     idx += 1
#
#
# plt.xticks(X + (len(classifierSet_names) - 1) * (slide / 2),
#            ('LDA FS1', 'QDA FS1', 'LDA FS2', 'QDA FS2', 'LDA FS3', 'QDA FS3'))
# # ax[idx].legend(labels=classifierSet_names)
# plt.show()
#
# plt.savefig('../images/acc_batch.png', dpi=300)
#
#
# ### Wilcoxon Test all datasets
# p_value_control=0.05
# for type_DA in [['LDA', 0], ['QDA', 1]]:
#     classifierSet = [type_DA[0] + '_' + 'batch',type_DA[0] + '_ours_labels']
#
#     _, p = stats.wilcoxon(data_frame_total[type_DA[0] + '_' + 'batch'], data_frame_total[type_DA[0] + '_ours_labels'], alternative='greater')
#
#     if p < p_value_control:
#         print(
#             type_DA[0]
#             + ' the mean of the accuracy of the batch and our classifier is different (wilcoxon)')
#     else:
#         print(
#             type_DA[0]
#             + ' the mean of the accuracy of the batch and our classifier is same (wilcoxon)')
#     print('accuracy of batch:',data_frame_total[type_DA[0] + '_' + 'batch'].mean())
#     print('accuracy of ours:', data_frame_total[type_DA[0] + '_ours_labels'].mean())

# %% GRAPH PAPER ACC weak and batch labeled

# data_frame_total = pd.DataFrame()
# fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(8, 20), sharex=True)
# idx = 0
# for database in ['Nina5', 'Cote', 'LongTerm3DC', 'Capgmyo_dbb', 'EPN_120']:
#     print('\n' + database)
#     folder = '../Results/final1/'
#     data_frame_database = pd.DataFrame()
#
#     data_acc_mean_all_1, _, _, _, _ = uploadResultsDatabasesVF1(
#         folder, database, 1, times, shotStart, best_probs=0, best_threshold=0)
#
#     data_acc_mean_all_2, _, _, _, _ = uploadResultsDatabasesVF1(
#         folder, database, 2, times, shotStart, best_probs=0, best_threshold=0)
#
#     data_acc_mean_all_3, _, _, _, _ = uploadResultsDatabasesVF1(
#         folder, database, 3, times, shotStart, best_probs=0, best_threshold=0)
#
#     ##### graph best acc
#     print('\nFEATURE 1: ' + database)
#     best_probs1 = vect_bar_PROBS(data_acc_mean_all_1)
#     best_threshold1 = vect_bar_THRESHOLD(data_acc_mean_all_1)
#     print('FEATURE 2: ' + database)
#     best_probs2 = vect_bar_PROBS(data_acc_mean_all_2)
#     best_threshold2 = vect_bar_THRESHOLD(data_acc_mean_all_2)
#     print('FEATURE 3: ' + database)
#     best_probs3 = vect_bar_PROBS(data_acc_mean_all_3)
#     best_threshold3 = vect_bar_THRESHOLD(data_acc_mean_all_3)
#
#     print('\nANALYSIS WILCOXON (CONFIDENCE LEVEL 95%): ' + database)
#
#     _, data_acc_mean_best1, data_acc_best1, _, _ = uploadResultsDatabasesVF1(
#         folder, database, 1, times, shotStart, best_probs=best_probs1, best_threshold=best_threshold1)
#
#     data_frame_database = data_frame_database.append(data_acc_best1, ignore_index=True, sort=False)
#
#     _, data_acc_mean_best2, data_acc_best2, _, _ = uploadResultsDatabasesVF1(
#         folder, database, 2, times, shotStart, best_probs=best_probs2, best_threshold=best_threshold2)
#     data_frame_database = data_frame_database.append(data_acc_best2, ignore_index=True, sort=False)
#
#     _, data_acc_mean_best3, data_acc_best3, _, _ = uploadResultsDatabasesVF1(
#         folder, database, 3, times, shotStart, best_probs=best_probs3, best_threshold=best_threshold3)
#     data_frame_database = data_frame_database.append(data_acc_best3, ignore_index=True, sort=False)
#
#     data_frame_total = data_frame_total.append(data_frame_database, ignore_index=True, sort=False)
#
#     ##### graph best acc
#     vect = []
#     vect = vect_bar(vect, data_acc_mean_best1, best_probs1, best_threshold1)
#     vect = vect_bar(vect, data_acc_mean_best2, best_probs2, best_threshold2)
#     vect = vect_bar(vect, data_acc_mean_best3, best_probs3, best_threshold3)
#
#     classifierSet_names = ['weak']
#     for type_model in [['Ours', 0], ['DR updating', 1]]:
#         classifierSet_names.append(type_model[0] + ' and our soft-labelling')
#         # classifierSet_names.append(type_model[0] + '_labels')
#         classifierSet_names.append(type_model[0] + ' and best Nigam weight')
#         classifierSet_names.append(type_model[0] + ' and best threshold')
#
#     vect = np.array(vect)
#     # ax = plt.subplot(idx, 1, 1)
#
#     # plt.figure(figsize=(8, 5))
#     ax[idx].grid(axis='y', color='gainsboro', linewidth=1, zorder=1)
#     X = np.arange(2 * 3)
#
#     slide = 0.1
#     list_colors = ['tab:green', 'deepskyblue', 'dodgerblue','tab:blue','lightcoral', 'indianred', 'tab:red' ]
#     for i in range(len(classifierSet_names)):
#         ax[idx].bar(X + slide * i, vect[:, i], width=slide, zorder=i + 2, color=list_colors[i])
#
#     # ax[idx].legend(labels=classifierSet_names)
#     if idx == 0:
#         ax[idx].set_yticks(np.arange(35, 70, step=5))
#         ax[idx].set_ylim([35, 70])
#     elif idx == 1:
#         ax[idx].set_yticks(np.arange(75, 100, step=5))
#         ax[idx].set_ylim([80, 100])
#     elif idx == 2:
#         ax[idx].set_yticks(np.arange(55, 85, step=5))
#         ax[idx].set_ylim([55, 85])
#     elif idx == 3:
#         ax[idx].set_yticks(np.arange(45, 96, step=10))
#         ax[idx].set_ylim([45, 100])
#     elif idx == 4:
#         ax[idx].set_yticks(np.arange(55, 85, step=5))
#         ax[idx].set_ylim([55, 85])
#
#     # ax[idx].set_xticks(X + (len(classifierSet_names) - 1) * (slide / 2), ('LDA', 'QDA'))
#     ax[idx].set_title(database)
#     ax[idx].set_ylabel('accuracy [%]')
#
#     idx += 1
#
#     # print('\n\nFRIEDMAN PER DATABASE (CONFIDENCE LEVEL 95%): '+database)
#     # for type_DA in ['LDA', 'QDA']:
#     #     print('\nType DA classifier: '+type_DA)
#     #     classifierSet_names = []
#     #     for type_model in [['ours', 0], ['state_art', 1]]:
#     #         classifierSet_names.append(type_DA + '_' + type_model[0] + '_soft_labels')
#     #         classifierSet_names.append(type_DA + '_' + type_model[0] + '_best_probs')
#     #         classifierSet_names.append(type_DA + '_' + type_model[0] + '_best_threshold')
#     #
#     #     friedman_analysis(data_frame_database[classifierSet_names])
#
#     # print('\n\nFRIEDMAN PER DATABASE (CONFIDENCE LEVEL 95%) soft-labelling technique: ' + database)
#     # frame = pd.DataFrame()
#     # for type_DA in ['LDA', 'QDA']:
#     #     print('\nType DA classifier: ' + type_DA)
#     #     for type_model in ['ours', 'state_art']:
#     #         classifierSet_names = []
#     #         for type_updating in ['soft_labels', 'best_probs', 'best_threshold']:
#     #             classifierSet_names.append(type_DA + '_' + type_model + '_'+type_updating)
#     #
#     #         aux_frame = data_frame_database[classifierSet_names]
#     #         aux_frame.columns = ['ours_soft_labelling','best_probs', 'best_threshold']
#     #         frame = frame.append(aux_frame, ignore_index=True, sort=False)
#     #
#     #     friedman_analysis(frame)
#     #
#     # print('\n\nFRIEDMAN PER DATABASE (CONFIDENCE LEVEL 95%) updating method: ' + database)
#     # frame = pd.DataFrame()
#     # for type_DA in ['LDA', 'QDA']:
#     #     print('\nType DA classifier: ' + type_DA)
#     #     for type_updating in ['soft_labels','best_probs', 'best_threshold']:
#     #         classifierSet_names = []
#     #         for type_model in ['ours', 'state_art']:
#     #             classifierSet_names.append(type_DA + '_' + type_model + '_'+type_updating)
#     #
#     #         aux_frame = data_frame_database[classifierSet_names]
#     #         aux_frame.columns = ['ours', 'state_art']
#     #         frame = frame.append(aux_frame, ignore_index=True, sort=False)
#     #
#     #     friedman_analysis(frame)
#
# plt.xticks(X + (len(classifierSet_names) - 1) * (slide / 2),
#            ('LDA FS1', 'QDA FS1', 'LDA FS2', 'QDA FS2', 'LDA FS3', 'QDA FS3'))
# # ax[idx].legend(labels=classifierSet_names)
# plt.show()
#
# plt.savefig('../images/acc_batch.png', dpi=300)
#
# # print('\n\nFRIEDMAN TOTAL (CONFIDENCE LEVEL 95%)')
# # for type_DA in ['LDA', 'QDA']:
# #     classifierSet_names = []
# #     for type_model in [['ours', 0], ['state_art', 1]]:
# #         classifierSet_names.append(type_DA + '_' + type_model[0] + '_soft_labels')
# #         classifierSet_names.append(type_DA + '_' + type_model[0] + '_best_probs')
# #         classifierSet_names.append(type_DA + '_' + type_model[0] + '_best_threshold')
# #
# #     friedman_analysis(data_frame_total[classifierSet_names])
#
#
# # print('\n\nFRIEDMAN TOTAL (CONFIDENCE LEVEL 95%) soft-labelling technique')
# # frame = pd.DataFrame()
# # for type_DA in ['LDA', 'QDA']:
# #     print('\nType DA classifier: ' + type_DA)
# #     for type_model in ['ours', 'state_art']:
# #         classifierSet_names = []
# #         for type_updating in ['soft_labels', 'best_probs', 'best_threshold']:
# #             classifierSet_names.append(type_DA + '_' + type_model + '_'+type_updating)
# #
# #         aux_frame = data_frame_total[classifierSet_names]
# #         aux_frame.columns = ['ours_soft_labelling','best_probs', 'best_threshold']
# #         frame = frame.append(aux_frame, ignore_index=True, sort=False)
# #
# #     friedman_analysis(frame)
# #
# # print('\n\nFRIEDMAN TOTAL (CONFIDENCE LEVEL 95%) updating method')
# # frame = pd.DataFrame()
# # for type_DA in ['LDA', 'QDA']:
# #     print('\nType DA classifier: ' + type_DA)
# #     for type_updating in ['soft_labels','best_probs', 'best_threshold']:
# #         classifierSet_names = []
# #         for type_model in ['ours', 'state_art']:
# #             classifierSet_names.append(type_DA + '_' + type_model + '_'+type_updating)
# #
# #         aux_frame = data_frame_total[classifierSet_names]
# #         aux_frame.columns = ['ours', 'state_art']
# #         frame = frame.append(aux_frame, ignore_index=True, sort=False)
# #
# #     friedman_analysis(frame)



# %% GRAPH PAPER WEIGHT ERROR
#
# fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(12, 22))
# idx = 0
# numberClassifiers = 24
# for info in ['Nina5', 'Cote', 'LongTerm3DC', 'Capgmyo_dbb']:
#     if info == 'Nina5':
#         x_ticks = np.hstack((np.arange(0, 40 + 1, 10), np.array([54])))
#         b_wp = 0.6
#         b_t = 0.1
#     elif info == 'Cote':
#         x_ticks = np.hstack((np.arange(0, 40 + 1, 10), np.array([56])))
#         b_wp = 1.0
#         b_t = 0.4
#     elif info == 'LongTerm3DC':
#         x_ticks = np.hstack((np.arange(0, 8 + 1, 2), np.array([11])))
#         b_wp = 1.0
#         b_t = 0.7
#     elif info == 'Capgmyo_dbb':
#         x_ticks = np.hstack((np.arange(0, 40 + 1, 10), np.array([48])))
#         b_wp = 1.0
#         b_t = 0.5
#
#     folder = '../Results/final/'
#     data_acc, data_time_update, data_time_weight, data_weight, gestureSet = uploadResultsDatabasesVF1(
#         folder, info, featureSet, times, shotStart)
#
#     # Compare all classifiers
#     classifierSet = ['incre_proposed_semi']
#     classifierSet.append('incre_weight_postprobability_' + str(b_wp))
#     classifierSet.append('incre_threshold_' + str(b_t))
#     idx_classifierSet = np.arange(6)
#     plot_arrays(data_weight, idx, ax, classifierSet, idx_classifierSet, x_ticks, 'RMSE', gestureSet)
#
#     idx += 1
#
# ax[0, 0].set_title('LDA')
# ax[0, 1].set_title('QDA')
# for i in range(2):
#     ax[3, i].set_xlabel('streaming gestures')
#
# # ax[3, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
# #                 fancybox=True, shadow=True, ncol=3)
#
# plt.savefig('../images/rmse.png', dpi=300)
# plt.show()
