# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FormatStrFormatter
from scipy import stats
import matplotlib.pyplot as plt


# %% Upload results of the three databases
def uploadResults(folder, database, samples, people, times, featureSet, classes, days, shotStart, typeDA):
    folder = folder + '_' + database
    average = pd.DataFrame()
    detail = pd.DataFrame()
    for metric in ['acc']:
        average.loc[0, metric + '_' + typeDA + '_' + 'weak'] = 0
        average.loc[0, metric + '_' + typeDA + '_' + 'incre_proposed'] = 0
        average.loc[0, metric + '_' + typeDA + '_' + 'incre_proposed2'] = 0
        average.loc[0, metric + '_' + typeDA + '_' + 'incre_proposed3'] = 0
        average.loc[0, metric + '_' + typeDA + '_' + 'incre_labels'] = 0
        average.loc[0, metric + '_' + typeDA + '_' + 'incre_labels_SA'] = 0
        average.loc[0, metric + '_' + typeDA + '_' + 'incre_sequential_SA'] = 0
        average.loc[0, metric + '_' + typeDA + '_' + 'incre_supervised'] = 0
        average.loc[0, metric + '_' + typeDA + '_' + 'incre_supervised_SA'] = 0
        for l in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
            average.loc[0, metric + '_' + typeDA + '_' + 'incre_Nigam_' + str(l)] = 0
        for l in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            average.loc[0, metric + '_' + typeDA + '_' + 'incre_threshold_' + str(l)] = 0

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

                if len(resultsTest) != samples * classes * days:
                    print('error' + '_time' + str(time) + '_person' + str(person) + '_FS' + str(featureSet))
                    print(len(resultsTest))

                for day in range(1, days + 1):
                    for metric in ['acc']:
                        average.loc[0, metric + '_' + typeDA + '_' + 'weak'] += resultsTest.loc[
                            samples * classes * day - 1, metric + '_' + typeDA + '_' + 'weak']
                        average.loc[0, metric + '_' + typeDA + '_' + 'incre_proposed'] += resultsTest.loc[
                            samples * classes * day - 1, metric + '_' + typeDA + '_' + 'incre_proposed']
                        average.loc[0, metric + '_' + typeDA + '_' + 'incre_proposed2'] += resultsTest.loc[
                            samples * classes * day - 1, metric + '_' + typeDA + '_' + 'incre_proposed2']
                        average.loc[0, metric + '_' + typeDA + '_' + 'incre_proposed3'] += resultsTest.loc[
                            samples * classes * day - 1, metric + '_' + typeDA + '_' + 'incre_proposed3']
                        average.loc[0, metric + '_' + typeDA + '_' + 'incre_labels'] += resultsTest.loc[
                            samples * classes * day - 1, metric + '_' + typeDA + '_' + 'incre_labels']
                        average.loc[0, metric + '_' + typeDA + '_' + 'incre_labels_SA'] += resultsTest.loc[
                            samples * classes * day - 1, metric + '_' + typeDA + '_' + 'incre_labels_SA']
                        average.loc[0, metric + '_' + typeDA + '_' + 'incre_sequential_SA'] += resultsTest.loc[
                            samples * classes * day - 1, metric + '_' + typeDA + '_' + 'incre_sequential_SA']
                        average.loc[0, metric + '_' + typeDA + '_' + 'incre_supervised'] += resultsTest.loc[
                            samples * classes * day - 1, metric + '_' + typeDA + '_' + 'incre_supervised']
                        average.loc[0, metric + '_' + typeDA + '_' + 'incre_supervised_SA'] += resultsTest.loc[
                            samples * classes * day - 1, metric + '_' + typeDA + '_' + 'incre_supervised_SA']
                        for l in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
                            average.loc[0, metric + '_' + typeDA + '_' + 'incre_Nigam_' + str(l)] += resultsTest.loc[
                                samples * classes * day - 1, metric + '_' + typeDA + '_' + 'incre_Nigam_' + str(l)]
                        for l in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                            average.loc[0, metric + '_' + typeDA + '_' + 'incre_threshold_' + str(l)] += \
                                resultsTest.loc[samples * classes * day - 1,
                                                metric + '_' + typeDA + '_' + 'incre_threshold_' + str(l)]
                        detail.loc[count, metric + '_' + typeDA + '_' + 'incre_proposed'] = resultsTest.loc[
                            samples * classes * day - 1, metric + '_' + typeDA + '_' + 'incre_proposed']
                        detail.loc[count, metric + '_' + typeDA + '_' + 'weak'] = resultsTest.loc[
                            samples * classes * day - 1, metric + '_' + typeDA + '_' + 'weak']

                    count += 1
            except:
                print('error' + '_time' + str(time) + '_person' + str(person) + '_FS' + str(featureSet))
    return average / count, detail


def uploadResultsDatabasesVF1(folder, database, featureSet, times, shotStart, typeDA):
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
    return uploadResults(folder, database, samplesTrain - 1, people, times, featureSet, classes, days, shotStart,
                         typeDA)


featureSet = 1
times = 20
windowSize = 290
shotStart = 1
plt.rcParams["figure.figsize"] = (22, 10)
# for info in [['Cote', 'LDA', 0.95, 1], ['Cote', 'QDA', 0.91, 1], ['Nina5', 'LDA', 0.56, 0.75],
#              ['Nina5', 'QDA', 0.45, 0.85], ['LongTerm3DC', 'LDA', 0.64, 0.84], ['LongTerm3DC', 'QDA', 0.4, 0.85],
#              ['Capgmyo_dbb', 'LDA', 0.83, 1], ['Capgmyo_dbb', 'QDA', 0.65, 1]]:
for info in [['EPN_612', 'LDA', 0.56, 0.75],['EPN_612', 'QDA', 0.40, 0.80]]:
    for folder_idx in range(2,3):
        folder = 'results' + str(folder_idx) + '/'
        result, detail = uploadResultsDatabasesVF1(folder, info[0], featureSet, times, shotStart, info[1])
        result = result.T
        # my_colors = 'brbbbgkkkkkkyyyyyyyy'
        my_colors = 'brmcgkkkkkkyyyyyyyy'
        ax = result.plot(kind='bar', ylim=(info[2], info[3]), zorder=2, color=my_colors, )

        plt.ylabel("acc")
        plt.xlabel("models")
        plt.title(info[0] + ' ' + info[1] + ' folder' + str(folder_idx))
        plt.grid(color='gainsboro', linewidth=1, zorder=1)
        plt.xticks(rotation=80, horizontalalignment="center")
        for p in ax.patches:
            b = p.get_bbox()
            ax.annotate("{:.3f}".format(b.y1 + b.y0), (p.get_x() , p.get_height() * 1.005))

        plt.show()
