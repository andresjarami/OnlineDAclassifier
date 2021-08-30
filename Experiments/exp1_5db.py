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


# %% Upload results of the three databases
def uploadResults2(folder, database, samplesTrain, people, times, featureSet, classes, days, shotStart):
    folder = folder + '_' + database

    all_samples = (samplesTrain - shotStart) * classes * days
    number_gestures = 5
    gestureSet = np.append(np.arange(np.floor(all_samples / number_gestures), all_samples,
                                     np.floor(all_samples / number_gestures)), all_samples)
    numberClassifiers = 48
    data = np.zeros((len(gestureSet), numberClassifiers))

    classifierSet = ['weak_model', 'batch_model', 'incre_proposed_sup', 'incre_proposed_semi']
    for weight in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        classifierSet.append('incre_weight_postprobability_' + str(weight))
    for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        classifierSet.append('incre_threshold_' + str(threshold))
    classifierSet.append(classifierSet)
    count = 0
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

                if len(resultsTest) != len(gestureSet):
                    print('error' + '_time' + str(time) + '_person' + str(person) + '_FS' + str(featureSet))
                    print(len(resultsTest))

                typeDA = 'LDA'
                for classifier in range(numberClassifiers):
                    for metric in ['acc']:
                        if classifier >= 24:
                            typeDA = 'QDA'
                            classifier_aux = classifier - 24
                        else:
                            classifier_aux = classifier
                        idx = 0
                        for gesture in gestureSet:
                            data[idx, classifier] += \
                                resultsTest[metric + '_' + typeDA + '_' + classifierSet[classifier_aux]].loc[
                                    resultsTest['unlabeled Gesture'] == gesture].sum()
                            idx += 1
                count += 1
            except:
                print('error' + '_time' + str(time) + '_person' + str(person) + '_FS' + str(featureSet))
        print(time)
    return data / (count * days)


def uploadResults(folder, database, samplesTrain, people, times, featureSet, classes, days, shotStart):
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

                if len(resultsTest) != len(gestureSet) * days:
                    print('error' + '_time' + str(time) + '_person' + str(person) + '_FS' + str(featureSet))
                    print(len(resultsTest), '/', len(gestureSet))

                dataFrame = dataFrame.append(resultsTest)

            except:
                print('error' + '_time' + str(time) + '_person' + str(person) + '_FS' + str(featureSet))

    data_acc = get_data_acc(dataFrame, gestureSet)
    aux_weak = np.hstack((np.ones((1, int(np.size(data_acc, axis=1) / 2))) * data_acc[0, 0],
                          np.ones((1, int(np.size(data_acc, axis=1) / 2))) * data_acc[0, 24]))
    data_acc = np.vstack((aux_weak, data_acc))


    data_time_update = get_data_time_update(dataFrame, gestureSet)
    data_time_update = np.vstack((np.ones((1, np.size(data_time_update, axis=1))) * 0, data_time_update))


    # data_time_weight = get_data_time_weight(dataFrame, gestureSet)
    # data_weight = get_data_weight(dataFrame, gestureSet, classes, database)
    # data_weight = np.vstack((np.ones((1, np.size(data_weight, axis=1))) * 0, data_weight))

    return data_acc, data_time_update, 0, 0, np.hstack((np.array([0]), gestureSet))


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


def get_data_acc(dataFrame, gestureSet):
    metric = 'acc'
    numberClassifiers = 48

    classifierSet = ['weak_model', 'batch_model', 'incre_proposed_sup', 'incre_proposed_semi']
    for weight in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        classifierSet.append('incre_weight_postprobability_' + str(weight))
    for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        classifierSet.append('incre_threshold_' + str(threshold))

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


def uploadResultsDatabasesVF1(folder, database, featureSet, times, shotStart):
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
    elif database == 'EPN_612':
        samplesTrain = 25
        samplesTest = 25
        classes = 6
        people = 612
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
    return uploadResults(folder, database, samplesTrain, people, times, featureSet, classes, days, shotStart)


def plot_arrays(data, idx, ax, classifierSet, idx_classifierSet, x_ticks, ylabel, gestureSet, slide=0):
    idx_classifier = 0
    for typeDA in [['LDA', 0], ['QDA', 1]]:
        for classifier in classifierSet:
            ax[idx, typeDA[1] + slide].plot(gestureSet, data[:, idx_classifierSet[idx_classifier]],
                                            label=classifier)
            idx_classifier += 1
        ax[idx, typeDA[1]].set_xticks(x_ticks)
        ax[idx, typeDA[1]].grid(color='gainsboro', linewidth=1)
    ax[idx, 0].set_ylabel(ylabel + ' (' + info + ')')



def set_title_labeles_legend(ax):
    ax[0, 0].set_title('LDA')
    ax[0, 1].set_title('QDA')
    ax[3, 0].set_xlabel('Number of Labled/Unlabeled sets')
    ax[3, 1].set_xlabel('Number of Labled/Unlabeled sets')
    ax[3, 1].legend()


# %% set initial variables

featureSet = 1
times = 20
windowSize = 290
shotStart = 1

# %% all models accuracy comparison
# fig, ax = plt.subplots(nrows=5, ncols=4, figsize=(36, 22))
# idx = 0
# for info in ['Nina5', 'Cote', 'LongTerm3DC', 'Capgmyo_dbb', 'EPN_612']:
#
#     if info == 'Nina5':
#         x_ticks = np.hstack((np.arange(0, 40 + 1, 10), np.array([54])))
#     elif info == 'Cote':
#         x_ticks = np.hstack((np.arange(0, 40 + 1, 10), np.array([56])))
#     elif info == 'LongTerm3DC':
#         x_ticks = np.hstack((np.arange(0, 8 + 1, 2), np.array([11])))
#     elif info == 'Capgmyo_dbb':
#         x_ticks = np.hstack((np.arange(0, 40 + 1, 10), np.array([48])))
#
#     folder = '../Results/final/'
#     data_acc, data_time_update, data_time_weight, data_weight, gestureSet = uploadResultsDatabasesVF1(
#         folder, info, featureSet, times, shotStart)
#
#     # best weight_postprobability
#     classifierSet = []
#     for weight in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#         classifierSet.append('incre_weight_postprobability_' + str(weight))
#     idx_classifierSet = np.hstack((np.arange(4, 4 + 10), np.arange(28, 28 + 10)))
#     plot_arrays(data_acc, idx, ax, classifierSet, idx_classifierSet, x_ticks, 'accuracy', gestureSet, slide=0)
#
#     # best threshold
#     classifierSet = []
#     for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#         classifierSet.append('incre_threshold_' + str(threshold))
#     idx_classifierSet = np.hstack((np.arange(14, 14 + 10), np.arange(38, 38 + 10)))
#     plot_arrays(data_acc, idx, ax, classifierSet, idx_classifierSet, x_ticks, 'accuracy', gestureSet, slide=2)
#
#     idx += 1
#
# ax[0, 0].set_title('LDA_weight_postprobability')
# ax[0, 1].set_title('QDA_weight_postprobability')
# ax[0, 2].set_title('LDA_threshold')
# ax[0, 3].set_title('QDA_threshold')
# for i in range(6):
#     ax[4, i].set_xlabel('Number of Labled/Unlabeled sets')
#     ax[4, i].legend()
#
# plt.show()

# %%GRAPH PAPER ACC

fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(12, 26))
idx = 0
for info in ['Nina5', 'Cote', 'LongTerm3DC', 'Capgmyo_dbb','EPN_612']:
    if info == 'Nina5':
        x_ticks = np.hstack((np.arange(0, 40 + 1, 10), np.array([54])))
    elif info == 'Cote':
        x_ticks = np.hstack((np.arange(0, 40 + 1, 10), np.array([56])))
    elif info == 'LongTerm3DC':
        x_ticks = np.hstack((np.arange(0, 8 + 1, 2), np.array([11])))
    elif info == 'Capgmyo_dbb':
        x_ticks = np.hstack((np.arange(0, 40 + 1, 10), np.array([48])))
    elif info == 'EPN_612':
        x_ticks = np.hstack((np.arange(0, 120+ 1, 30), np.array([144])))

    folder = '../Results/final/'
    data_acc, data_time_update, data_time_weight, data_weight, gestureSet = uploadResultsDatabasesVF1(
        folder, info, featureSet, times, shotStart)

    # Compare all classifiers
    classifierSet = ['weak', 'OnlineSUP or batch', 'OnlineSUP with our weight']
    best_weight_postprobability = [0.6, 1.0, 1.0, 1.0, 0.4]
    idx_best_wp = [9, 13, 13, 13, 7]
    classifierSet.append(
        'OnlineSUP with Nigam weight')
    best_threshold = [0.1, 0.4, 0.7, 0.5, 0.5]
    idx_best_t = [15, 18, 21, 19, 19]
    classifierSet.append('OnlineSUP with threshold' )
    idx_classifierSet = np.hstack(
        (np.array([0, 1, 3]), np.array([idx_best_wp[idx]]), np.array([idx_best_t[idx]]), np.array([24, 25, 27]),
         np.array([idx_best_wp[idx] + 24]), np.array([idx_best_t[idx] + 24])))
    plot_arrays(data_acc,idx, ax, classifierSet, idx_classifierSet, x_ticks, 'accuracy', gestureSet)

    idx += 1

ax[0, 0].set_title('LDA')
ax[0, 1].set_title('QDA')
for i in range(2):
    ax[4, i].set_xlabel('streaming gestures over time $t$')


# ax[3, 0].legend(loc='upper center', bbox_to_anchor=(1, -0.05),
#                 fancybox=True, shadow=True, ncol=3)

plt.savefig('../images/acc.png', dpi=300)
plt.show()

# %% GRAPH PAPER TIME UPDATE

fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(12, 26))
idx = 0
numberClassifiers = 24
for info in ['Nina5', 'Cote', 'LongTerm3DC', 'Capgmyo_dbb','EPN_612']:
    if info == 'Nina5':
        x_ticks = np.hstack((np.arange(0, 40 + 1, 10), np.array([54])))
    elif info == 'Cote':
        x_ticks = np.hstack((np.arange(0, 40 + 1, 10), np.array([56])))
    elif info == 'LongTerm3DC':
        x_ticks = np.hstack((np.arange(0, 8 + 1, 2), np.array([11])))
    elif info == 'Capgmyo_dbb':
        x_ticks = np.hstack((np.arange(0, 40 + 1, 10), np.array([48])))
    elif info == 'EPN_612':
        x_ticks = np.hstack((np.arange(0, 120+ 1, 30), np.array([144])))

    folder = '../Results/final/'
    data_acc, data_time_update, data_time_weight, data_weight, gestureSet = uploadResultsDatabasesVF1(
        folder, info, featureSet, times, shotStart)

    # Compare all classifiers
    classifierSet = ['batch', 'OnlineSUP', 'OnlineSEMI']
    idx_classifierSet = np.arange(6)
    plot_arrays(data_time_update,idx, ax, classifierSet, idx_classifierSet, x_ticks, 'updating time [s]', gestureSet)
    print('\nAverage of the updating time over time (streaming gestures over time t) of our online classifiers ('+info+' dataset): ')
    print('LDA OnlineSUP: ',data_time_update[:,1].mean(),'±',data_time_update[:,1].std())
    print('QDA OnlineSEMI: ', data_time_update[:, 2].mean(), '±', data_time_update[:, 2].std())
    print('QDA OnlineSUP: ', data_time_update[:, 4].mean(), '±', data_time_update[:, 4].std())
    print('QDA OnlineSEMI: ', data_time_update[:, 5].mean(), '±', data_time_update[:, 5].std())

    idx += 1

ax[0, 0].set_title('LDA')
ax[0, 1].set_title('QDA')
for i in range(2):
    ax[4, i].set_xlabel('streaming gestures over time $t$')

# ax[3, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#                 fancybox=True, shadow=True, ncol=3)

plt.savefig('../images/updating_time.png', dpi=300)
plt.show()


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
