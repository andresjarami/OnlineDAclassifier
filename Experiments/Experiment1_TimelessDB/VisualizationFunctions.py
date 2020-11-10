import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.ticker as mtick
from scipy import stats


def graphSyntheticData(resultsData, numberShots, iSample):
    shot = np.arange(numberShots)
    AccLDAInd = np.zeros(numberShots)
    AccQDAInd = np.zeros(numberShots)

    AccLDAMulti = np.zeros(numberShots)
    AccQDAMulti = np.zeros(numberShots)

    AccLDAProp = np.zeros(numberShots)
    AccQDAProp = np.zeros(numberShots)
    AccLDALiu = np.zeros(numberShots)
    AccQDALiu = np.zeros(numberShots)
    AccLDAVidovic = np.zeros(numberShots)
    AccQDAVidovic = np.zeros(numberShots)

    for i in range(numberShots):
        AccLDAInd[i] = resultsData['AccLDAInd'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccQDAInd[i] = resultsData['AccQDAInd'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()

        AccLDAMulti[i] = resultsData['AccLDAMulti'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccQDAMulti[i] = resultsData['AccQDAMulti'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()

        AccLDAProp[i] = resultsData['AccLDAProp'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccQDAProp[i] = resultsData['AccQDAProp'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccLDALiu[i] = resultsData['AccLDALiu'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccQDALiu[i] = resultsData['AccQDALiu'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccLDAVidovic[i] = resultsData['AccLDAVidovic'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccQDAVidovic[i] = resultsData['AccQDAVidovic'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()

    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(9, 3))
    sizeM = 5
    ax1.plot(shot + iSample, AccLDAInd, label='Individual', markersize=sizeM, color='tab:orange', linestyle='--')
    ax1.plot(shot + iSample, AccLDAMulti, label='Multi-user', markersize=sizeM, color='tab:purple',
             linestyle=(0, (3, 3, 1, 3, 1, 3)))
    ax1.plot(shot + iSample, AccLDALiu, label='Liu', markersize=sizeM, color='tab:green', linestyle=':')
    ax1.plot(shot + iSample, AccLDAVidovic, label='Vidovic', markersize=sizeM, color='tab:red',
             linestyle=(0, (3, 3, 1, 3)))
    ax1.plot(shot + iSample, AccLDAProp, label='Our technique', color='tab:blue')
    ax1.set_title('LDA')
    ax1.grid()
    ax1.set_ylabel('accuracy')
    ax2.plot(shot + iSample, AccQDAInd, label='Individual', markersize=sizeM, color='tab:orange', linestyle='--')
    ax2.plot(shot + iSample, AccQDAMulti, label='Multi-user', markersize=sizeM, color='tab:purple',
             linestyle=(0, (3, 3, 1, 3, 1, 3)))
    ax2.plot(shot + iSample, AccQDALiu, label='Liu', markersize=sizeM, color='tab:green', linestyle=':')
    ax2.plot(shot + iSample, AccQDAVidovic, label='Vidovic', markersize=sizeM, color='tab:red',
             linestyle=(0, (3, 3, 1, 3)))
    ax2.plot(shot + iSample, AccQDAProp, label='Our technique', color='tab:blue')
    ax2.set_title('QDA')
    ax2.grid()
    ax2.set_ylabel('accuracy')
    ax2.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0.)
    ax2.set_xlabel('samples')

    fig1.tight_layout()
    plt.show()


#####################################################################

# Graph a ellipse of a normal distribution
# Taken from https://matplotlib.org/3.1.0/gallery/statistics/confidence_ellipse.html

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


###################################################################################################
#### Functions used in the Jupyters Notebooks


def uploadResults(place, samples, people):
    resultsTest = pd.read_csv(place + "_FeatureSet_1_startPerson_1_endPerson_1.csv")
    if len(resultsTest) != samples:
        print('error' + ' 1' + ' 1')
        print(len(resultsTest))

    for i in range(2, people + 1):
        auxFrame = pd.read_csv(place + "_FeatureSet_1_startPerson_" + str(i) + "_endPerson_" + str(i) + ".csv")
        resultsTest = pd.concat([resultsTest, auxFrame], ignore_index=True)
        if len(auxFrame) != samples:
            print('error' + ' 1 ' + str(i))
            print(len(auxFrame))
    for j in range(2, 4):
        for i in range(1, people + 1):
            auxFrame = pd.read_csv(
                place + "_FeatureSet_" + str(j) + "_startPerson_" + str(i) + "_endPerson_" + str(i) + ".csv")
            resultsTest = pd.concat([resultsTest, auxFrame], ignore_index=True)

            if len(auxFrame) != samples:
                print('error' + ' ' + str(j) + ' ' + str(i))
                print(len(auxFrame))

    return resultsTest.drop(columns='Unnamed: 0')


def uploadResultsDatabase(folder, database):
    if database == 'NinaPro5':
        samples = 4
        people = 10
        shots = 4
    elif database == 'Cote':
        samples = 4
        people = 17
        shots = 4
    elif database == 'EPN':
        samples = 25
        people = 30
        shots = 25
    place = folder + database
    resultsTest = pd.read_csv(place + "_FeatureSet_1_startPerson_1_endPerson_1.csv")
    if len(resultsTest) != samples:
        print('error' + ' 1' + ' 1')
        print(len(resultsTest))

    for i in range(2, people + 1):
        auxFrame = pd.read_csv(place + "_FeatureSet_1_startPerson_" + str(i) + "_endPerson_" + str(i) + ".csv")
        resultsTest = pd.concat([resultsTest, auxFrame], ignore_index=True)
        if len(auxFrame) != samples:
            print('error' + ' 1 ' + str(i))
            print(len(auxFrame))
    for j in range(2, 4):
        for i in range(1, people + 1):
            auxFrame = pd.read_csv(
                place + "_FeatureSet_" + str(j) + "_startPerson_" + str(i) + "_endPerson_" + str(i) + ".csv")
            resultsTest = pd.concat([resultsTest, auxFrame], ignore_index=True)

            if len(auxFrame) != samples:
                print('error' + ' ' + str(j) + ' ' + str(i))
                print(len(auxFrame))

    return analysisResults(resultsTest.drop(columns='Unnamed: 0'), shots)


def analysisTime(extractionT, timeM):
    extractionT = extractionT * 1000

    for featureSet in range(3):
        print('\nFeature set: ' + str(featureSet+1))
        print('Training Time of Our Technique [s]: ',
              round(timeM.loc[featureSet + 1, 'trainingTimeMean'], 2), '+-',
              round(timeM.loc[featureSet + 1, 'trainingTimeStd'], 2))

        print('LDA')
        CL = 'LDA'
        print('Extraction time [ms]: ', round(extractionT.loc[featureSet, :].mean(), 2),
              round(extractionT.loc[featureSet, :].std(), 2),
              'Classification time [ms]: ', timeM.loc[featureSet + 1, 'mean' + CL],
              timeM.loc[featureSet + 1, 'std' + CL],
              'Analysis time [ms]: ',
              round(extractionT.loc[featureSet, :].mean() + timeM.loc[featureSet + 1, 'mean' + CL], 2),
              round(np.sqrt((extractionT.loc[featureSet, :].var() + timeM.loc[featureSet + 1, 'var' + CL])), 2))

        print('QDA')
        CL = 'QDA'
        print('Extraction time [ms]: ', round(extractionT.loc[featureSet, :].mean(), 2),
              round(extractionT.loc[featureSet, :].std(), 2),
              'Classification time [ms]: ', timeM.loc[featureSet + 1, 'mean' + CL],
              timeM.loc[featureSet + 1, 'std' + CL],
              'Analysis time [ms]: ',
              round(extractionT.loc[featureSet, :].mean() + timeM.loc[featureSet + 1, 'mean' + CL], 2),
              round(np.sqrt((extractionT.loc[featureSet, :].var() + timeM.loc[featureSet + 1, 'var' + CL])), 2))


def analysisResults(resultDatabase, shots):
    results = pd.DataFrame(columns=['Feature Set', '# shots'])
    timeOurTechnique = pd.DataFrame(columns=[])

    idx = 0
    for j in range(1, 4):
        for i in range(1, shots + 1):
            results.at[idx, 'Feature Set'] = j
            results.at[idx, '# shots'] = i

            subset = str(tuple(range(1, i + 1)))

            LDAmulti = resultDatabase['AccLDAMulti'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            QDAmulti = resultDatabase['AccQDAMulti'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]

            LDAInd = resultDatabase['AccLDAInd'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            QDAInd = resultDatabase['AccQDAInd'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            PropQ = resultDatabase['AccQDAProp'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            PropQ_L = resultDatabase['AccLDAPropQ'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]

            LiuL = resultDatabase['AccLDALiu'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            LiuQ = resultDatabase['AccQDALiu'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            VidL = resultDatabase['AccLDAVidovic'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            VidQ = resultDatabase['AccQDAVidovic'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]

            wmQ = resultDatabase['wTargetMeanQDAm'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            wcQ = resultDatabase['wTargetCovQDAm'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]

            trainingPropT = resultDatabase['tPropQ'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            classificationPropQDAT = resultDatabase['tCLPropQ'].loc[(resultDatabase['Feature Set'] == j)]
            classificationPropLDAT = resultDatabase['tCLPropL'].loc[(resultDatabase['Feature Set'] == j)]

            results.at[idx, 'LDA_Ind'] = LDAInd.mean(axis=0)
            results.at[idx, 'QDA_Ind'] = QDAInd.mean(axis=0)
            results.at[idx, 'LDA_Multi'] = LDAmulti.mean(axis=0)
            results.at[idx, 'QDA_Multi'] = QDAmulti.mean(axis=0)
            results.at[idx, 'LiuL'] = LiuL.mean(axis=0)
            results.at[idx, 'LiuQ'] = LiuQ.mean(axis=0)
            results.at[idx, 'VidL'] = VidL.mean(axis=0)
            results.at[idx, 'VidQ'] = VidQ.mean(axis=0)
            results.at[idx, 'PropQ'] = PropQ.mean(axis=0)
            results.at[idx, 'PropQ_L'] = PropQ_L.mean(axis=0)

            # results.at[idx, 'LDA_Ind'] = LDAInd.median(axis=0)
            # results.at[idx, 'QDA_Ind'] = QDAInd.median(axis=0)
            # results.at[idx, 'LDA_Multi'] = LDAmulti.median(axis=0)
            # results.at[idx, 'QDA_Multi'] = QDAmulti.median(axis=0)
            # results.at[idx, 'LiuL'] = LiuL.median(axis=0)
            # results.at[idx, 'LiuQ'] = LiuQ.median(axis=0)
            # results.at[idx, 'VidL'] = VidL.median(axis=0)
            # results.at[idx, 'VidQ'] = VidQ.median(axis=0)
            # results.at[idx, 'PropQ'] = PropQ.median(axis=0)
            # results.at[idx, 'PropQ_L'] = PropQ_L.median(axis=0)

            results.at[idx, 'wmQ'] = wmQ.mean(axis=0)
            results.at[idx, 'wcQ'] = wcQ.mean(axis=0)

            results.at[idx, 'trainingPropT'] = trainingPropT.mean(axis=0)
            results.at[idx, 'classificationPropQDAT'] = classificationPropQDAT.mean(axis=0)
            results.at[idx, 'classificationPropLDAT'] = classificationPropLDAT.mean(axis=0)
            results.at[idx, 'std_tPropQ'] = trainingPropT.std(axis=0)
            results.at[idx, 'std_tIndQ'] = classificationPropQDAT.std(axis=0)
            results.at[idx, 'std_tIndL'] = classificationPropLDAT.std(axis=0)

            results.at[idx, 'stdLDA_Ind'] = LDAInd.std(axis=0)
            results.at[idx, 'stdQDA_Ind'] = QDAInd.std(axis=0)
            results.at[idx, 'stdPropQ_L'] = PropQ_L.std(axis=0)
            results.at[idx, 'stdPropQ'] = PropQ.std(axis=0)
            results.at[idx, 'stdLiuL'] = LiuL.std(axis=0)
            results.at[idx, 'stdLiuQ'] = LiuQ.std(axis=0)

            confidence = 0.05

            p = stats.wilcoxon(PropQ_L.values, LDAInd.values, alternative='greater', zero_method='zsplit')[1]
            if p < confidence:
                results.at[idx, 'T-test (LDA_Ind)'] = p
            else:
                results.at[idx, 'T-test (LDA_Ind)'] = 1

            p = stats.wilcoxon(PropQ.values, QDAInd.values, alternative='greater', zero_method='zsplit')[1]
            if p < confidence:
                results.at[idx, 'T-test (QDA_Ind)'] = p
            else:
                results.at[idx, 'T-test (QDA_Ind)'] = 1

            p = stats.wilcoxon(PropQ_L.values, LDAmulti.values, alternative='greater', zero_method='zsplit')[1]
            if p < confidence:
                results.at[idx, 'T-test (LDA_Multi)'] = p
            else:
                results.at[idx, 'T-test (LDA_Multi)'] = 1

            p = stats.wilcoxon(PropQ.values, QDAmulti.values, alternative='greater', zero_method='zsplit')[1]
            if p < confidence:
                results.at[idx, 'T-test (QDA_Multi)'] = p
            else:
                results.at[idx, 'T-test (QDA_Multi)'] = 1

            idx += 1

        timeOurTechnique.at[j, 'meanLDA'] = round(classificationPropLDAT.mean(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'stdLDA'] = round(classificationPropLDAT.std(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'varLDA'] = round(classificationPropLDAT.var(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'meanQDA'] = round(classificationPropQDAT.mean(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'stdQDA'] = round(classificationPropQDAT.std(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'varQDA'] = round(classificationPropQDAT.var(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'trainingTimeMean'] = round(trainingPropT.mean(axis=0), 2)
        timeOurTechnique.at[j, 'trainingTimeStd'] = round(trainingPropT.std(axis=0), 2)

    return results, timeOurTechnique


def graphACC(resultsNina5T, resultsCoteT, resultsEPNT):
    FeatureSetM = 3
    fig, ax = plt.subplots(nrows=3, ncols=6, sharey='row', figsize=(13, 6))
    #     shot=np.arange(1,5)
    for classifier in range(2):
        for FeatureSet in range(FeatureSetM):
            for Data in range(FeatureSetM):
                if Data == 0:
                    shot = np.arange(1, 5)
                    results = resultsNina5T

                    ax[Data, FeatureSet].yaxis.set_ticks(np.arange(0.30, 1, .06))
                elif Data == 1:
                    shot = np.arange(1, 5)
                    results = resultsCoteT

                    ax[Data, FeatureSet].yaxis.set_ticks(np.arange(0.62, 1, 0.06))
                elif Data == 2:
                    results = resultsEPNT
                    shot = np.arange(1, 26)

                    ax[Data, FeatureSet].yaxis.set_ticks(np.arange(0.50, 1, 0.06))

                if classifier == 0:

                    Model = 'LDA_Ind'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1])
                    ax[Data, FeatureSet].plot(shot, Y, label='Individual', color='tab:orange', linestyle='--')

                    Model = 'LDA_Multi'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1])
                    ax[Data, FeatureSet].plot(shot, Y, label='Multi-user', color='tab:purple',
                                              linestyle=(0, (3, 3, 1, 3, 1, 3)))

                    Model = 'LiuL'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1])
                    ax[Data, FeatureSet].plot(shot, Y, label='Liu', color='tab:green', linestyle=':')

                    Model = 'VidL'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1])
                    ax[Data, FeatureSet].plot(shot, Y, label='Vidovic', color='tab:red', linestyle=(0, (3, 3, 1, 3)))

                    Model = 'PropQ_L'
                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1])
                    ax[Data, FeatureSet].plot(shot, Y, label='Our technique', color='tab:blue')

                    ax[Data, FeatureSet].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

                    if len(shot) == 25:
                        ax[Data, FeatureSet].xaxis.set_ticks([1, 5, 10, 15, 20, 25])
                    else:

                        ax[Data, FeatureSet].xaxis.set_ticks(np.arange(1, len(shot) + .2, 1))
                    ax[Data, FeatureSet].grid()




                elif classifier == 1:

                    Model = 'QDA_Ind'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1])
                    ax[Data, FeatureSet + 3].plot(shot, Y, label='Individual', color='tab:orange', linestyle='--')

                    Model = 'QDA_Multi'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1])
                    ax[Data, FeatureSet + 3].plot(shot, Y, label='Multi-user', color='tab:purple',
                                                  linestyle=(0, (3, 3, 1, 3, 1, 3)))

                    Model = 'LiuQ'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1])
                    ax[Data, FeatureSet + 3].plot(shot, Y, label='Liu', color='tab:green', linestyle=':')

                    Model = 'VidQ'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1])
                    ax[Data, FeatureSet + 3].plot(shot, Y, label='Vidovic', color='tab:red',
                                                  linestyle=(0, (3, 3, 1, 3)))

                    Model = 'PropQ'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1])

                    ax[Data, FeatureSet + 3].plot(shot, Y, label='Our technique', color='tab:blue')

                    ax[Data, FeatureSet + 3].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

                    if len(shot) == 25:
                        ax[Data, FeatureSet + 3].xaxis.set_ticks([1, 5, 10, 15, 20, 25])
                    else:
                        ax[Data, FeatureSet + 3].xaxis.set_ticks(np.arange(1, len(shot) + .2, 1))
                    ax[Data, FeatureSet + 3].grid()

    ax[2, 0].set_xlabel('repetitions')
    ax[2, 1].set_xlabel('repetitions')
    ax[2, 2].set_xlabel('repetitions')
    ax[2, 3].set_xlabel('repetitions')
    ax[2, 4].set_xlabel('repetitions')
    ax[2, 5].set_xlabel('repetitions')

    ax[0, 0].set_title('LDA\n Feature Set 1')
    ax[0, 1].set_title('LDA\n Feature Set 2')
    ax[0, 2].set_title('LDA\n Feature Set 3')

    ax[0, 3].set_title('QDA\n Feature Set 1')
    ax[0, 4].set_title('QDA\n Feature Set 2')
    ax[0, 5].set_title('QDA\n Feature Set 3')
    ax[0, 0].set_ylabel('NinaPro5\n\naccuracy')
    ax[1, 0].set_ylabel('Cote Allard\n\naccuracy')
    ax[2, 0].set_ylabel('EPN \n\naccuracy')

    ax[2, 5].legend(loc='lower center', bbox_to_anchor=(2, -0.7), ncol=5)

    fig.tight_layout(pad=0.1)
    # plt.savefig("databaseACC.png", bbox_inches='tight', dpi=600)
    plt.show()


def graphWeights(resultsNina5T, resultsCoteT, resultsEPNT):
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey='row', sharex='col', figsize=(9, 5))
    for Data in range(3):

        if Data == 0:
            shot = np.arange(1, 5)
            shots = 4
            results = resultsNina5T

            ax[Data].xaxis.set_ticks(np.arange(1, 4.1, 1))
            title = 'NinaPro5'
        elif Data == 1:
            shot = np.arange(1, 5)
            shots = 4
            results = resultsCoteT
            ax[Data].xaxis.set_ticks(np.arange(1, 4.1, 1))
            title = 'Cote-Allard'
        elif Data == 2:

            results = resultsEPNT
            shot = np.arange(1, 26)
            shots = 25

            ax[Data].xaxis.set_ticks([1, 5, 10, 15, 20, 25])
            title = 'EPN'

        wm = np.array(results['wmQ'])
        wc = np.array(results['wcQ'])

        ax[Data].plot(shot, np.mean(wm.reshape((3, shots)), axis=0), label='$\hat{\omega}_c$', marker='.',
                      color='tab:blue', markersize=5)
        ax[Data].plot(shot, np.mean(wc.reshape((3, shots)), axis=0), label='$\hat{\lambda}_c$', marker='^',
                      color='tab:blue', markersize=5)

        ax[Data].set_title(title)
        ax[Data].grid()

    ax[0].set_xlabel('repetitions')
    ax[1].set_xlabel('repetitions')
    ax[2].set_xlabel('repetitions')
    ax[0].set_ylabel('Weight value')
    fig.tight_layout(pad=1)
    # lgd = ax[2].legend(loc='lower center', bbox_to_anchor=(1.2, -0.8), ncol=2)
    # plt.savefig("weights.png", bbox_inches='tight', dpi=600, bbox_extra_artists=(lgd,))
    plt.show()


def AnalysisWilcoxon(folder):
    bases = ['NinaPro5', 'Cote', 'EPN']
    confidence = 0.05
    results = pd.DataFrame()

    idxD = 0

    for base in bases:

        if base == 'NinaPro5':
            samples = 4
            people = 10
            shots = 5
        elif base == 'Cote':
            samples = 4
            people = 17
            shots = 5
        elif base == 'EPN':
            samples = 25
            people = 30
            shots = 5
        idx = 0
        for f in range(1, 4):

            for s in range(1, shots):

                place = folder + base
                DataFrame = uploadResults(place, samples, people)

                OurResultsQDA = DataFrame['AccQDAProp'].loc[
                                    (DataFrame['Feature Set'] == f) & (DataFrame['# shots'] == s)].values * 100
                propL = DataFrame['AccLDAPropQ'].loc[
                            (DataFrame['Feature Set'] == f) & (DataFrame['# shots'] == s)].values * 100
                indQ = DataFrame['AccQDAInd'].loc[
                           (DataFrame['Feature Set'] == f) & (DataFrame['# shots'] == s)].values * 100
                indL = DataFrame['AccLDAInd'].loc[
                           (DataFrame['Feature Set'] == f) & (DataFrame['# shots'] == s)].values * 100
                liuL = DataFrame['AccLDALiu'].loc[
                           (DataFrame['Feature Set'] == f) & (DataFrame['# shots'] == s)].values * 100
                liuQ = DataFrame['AccQDALiu'].loc[
                           (DataFrame['Feature Set'] == f) & (DataFrame['# shots'] == s)].values * 100
                vidL = DataFrame['AccLDAVidovic'].loc[
                           (DataFrame['Feature Set'] == f) & (DataFrame['# shots'] == s)].values * 100
                vidQ = DataFrame['AccQDAVidovic'].loc[
                           (DataFrame['Feature Set'] == f) & (DataFrame['# shots'] == s)].values * 100

                iL = np.median(indL)
                iQ = np.median(indQ)
                pL = np.median(propL)
                OurQDAMedian = np.median(OurResultsQDA)
                lL = np.median(liuL)
                lQ = np.median(liuQ)
                vL = np.median(vidL)
                vQ = np.median(vidQ)

                WilcoxonMethod = 'wilcox'
                alternativeMethod = 'greater'

                results.at['LDA: Accuracy difference (Our and Individual classifiers) [%] ' + base, idx] = round(
                    pL - iL, 2)
                results.at['LDA: p-value (Our and Individual classifiers) [%] ' + base, idx] = 1
                if pL > iL:
                    p = stats.wilcoxon(propL, indL, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                    if p < confidence:
                        results.at['LDA: p-value (Our and Individual classifiers) [%] ' + base, idx] = p
                elif iL > pL:
                    p = stats.wilcoxon(indL, propL, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                    if p < confidence:
                        results.at['LDA: p-value (Our and Individual classifiers) [%] ' + base, idx] = p
                        print(1)

                results.at['LDA: Accuracy difference (Our and Liu classifiers) [%] ' + base, idx] = round(pL - lL, 2)
                results.at['LDA: p-value (Our and Liu classifiers) [%] ' + base, idx] = 1
                if pL > lL:
                    p = stats.wilcoxon(propL, liuL, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                    if p < confidence:
                        results.at['LDA: p-value (Our and Liu classifiers) [%] ' + base, idx] = p
                elif lL > pL:
                    p = stats.wilcoxon(liuL, propL, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                    if p < confidence:
                        results.at['LDA: p-value (Our and Liu classifiers) [%] ' + base, idx] = p
                        print(1)

                results.at['LDA: Accuracy difference (Our and Vidovic classifiers) [%] ' + base, idx] = round(pL - vL,
                                                                                                              2)
                results.at['LDA: p-value (Our and Vidovic classifiers) [%] ' + base, idx] = 1
                if pL > vL:
                    p = stats.wilcoxon(propL, vidL, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                    if p < confidence:
                        results.at['LDA: p-value (Our and Vidovic classifiers) [%] ' + base, idx] = p
                elif vL > pL:
                    p = stats.wilcoxon(vidL, propL, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                    if p < confidence:
                        results.at['LDA: p-value (Our and Vidovic classifiers) [%] ' + base, idx] = p
                        print(1)

                results.at['QDA: Accuracy difference (Our and Individual classifiers) [%] ' + base, idx] = round(
                    OurQDAMedian - iQ, 2)
                results.at['QDA: p-value (Our and Individual classifiers) [%] ' + base, idx] = 1
                if OurQDAMedian > iQ:
                    p = stats.wilcoxon(OurResultsQDA, indQ, alternative=alternativeMethod, zero_method=WilcoxonMethod)[
                        1]
                    if p < confidence:
                        results.at['QDA: p-value (Our and Individual classifiers) [%] ' + base, idx] = p

                elif iQ > OurQDAMedian:
                    p = stats.wilcoxon(indQ, OurResultsQDA, alternative=alternativeMethod, zero_method=WilcoxonMethod)[
                        1]
                    if p < confidence:
                        results.at['QDA: p-value (Our and Individual classifiers) [%] ' + base, idx] = p
                        print(1)

                results.at['QDA: Accuracy difference (Our and Liu classifiers) [%] ' + base, idx] = round(
                    OurQDAMedian - lQ, 2)
                results.at['QDA: p-value (Our and Liu classifiers) [%] ' + base, idx] = 1
                if OurQDAMedian > lQ:
                    p = stats.wilcoxon(OurResultsQDA, liuQ, alternative=alternativeMethod, zero_method=WilcoxonMethod)[
                        1]
                    if p < confidence:
                        results.at['QDA: p-value (Our and Liu classifiers) [%] ' + base, idx] = p
                elif lQ > OurQDAMedian:
                    p = stats.wilcoxon(liuQ, OurResultsQDA, alternative=alternativeMethod, zero_method=WilcoxonMethod)[
                        1]
                    if p < confidence:
                        results.at['QDA: p-value (Our and Liu classifiers) [%] ' + base, idx] = p
                        print(1)

                results.at['QDA: Accuracy difference (Our and Vidovic classifiers) [%] ' + base, idx] = round(
                    OurQDAMedian - vQ, 2)
                results.at['QDA: p-value (Our and Vidovic classifiers) [%] ' + base, idx] = 1
                if OurQDAMedian > vQ:
                    p = stats.wilcoxon(OurResultsQDA, vidQ, alternative=alternativeMethod, zero_method=WilcoxonMethod)[
                        1]
                    if p < confidence:
                        results.at['QDA: p-value (Our and Vidovic classifiers) [%] ' + base, idx] = p
                elif vQ > OurQDAMedian:
                    p = stats.wilcoxon(vidQ, OurResultsQDA, alternative=alternativeMethod, zero_method=WilcoxonMethod)[
                        1]
                    if p < confidence:
                        results.at['QDA: p-value (Our and Vidovic classifiers) [%] ' + base, idx] = p

                idx += 1
        idxD += 12
    return results


def friedman_test(*args):
    """
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

    chi2 = ((12 * n) / float((k * (k + 1)))) * (
            (np.sum(r ** 2 for r in rankings_avg)) - ((k * (k + 1) ** 2) / float(4)))
    iman_davenport = ((n - 1) * chi2) / float((n * (k - 1) - chi2))

    p_value = 1 - stats.f.cdf(iman_davenport, k - 1, (k - 1) * (n - 1))

    return iman_davenport, p_value, rankings_avg, rankings_cmp


def holm_test(ranks, control=None):
    """
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
        control_i = keys.index(control)

    comparisons = [keys[control_i] + " vs " + keys[i] for i in range(k) if i != control_i]
    z_values = [abs(values[control_i] - values[i]) for i in range(k) if i != control_i]
    p_values = [2 * (1 - stats.norm.cdf(abs(z))) for z in z_values]
    # Sort values by p_value so that p_0 < p_1
    p_values, z_values, comparisons = map(list, zip(*sorted(zip(p_values, z_values, comparisons), key=lambda t: t[0])))
    adj_p_values = [min(max((k - (j + 1)) * p_values[j] for j in range(i + 1)), 1) for i in range(k - 1)]

    return comparisons, z_values, p_values, adj_p_values, keys[control_i]


def AnalysisFriedman(folder):
    dataFrame = pd.DataFrame()

    base = 'NinaPro5'
    samples = 4
    people = 10
    place = folder + base
    DataFrameN = uploadResults(place, samples, people)
    base = 'Cote'
    samples = 4
    people = 17
    place = folder + base
    DataFrameC = uploadResults(place, samples, people)
    base = 'EPN'
    samples = 25
    people = 30
    place = folder + base
    DataFrameE = uploadResults(place, samples, people)

    TotalDataframe = pd.concat([DataFrameN, DataFrameC, DataFrameE])

    dataFrame['Our LDA classifier'] = TotalDataframe['AccLDAPropQ'].values
    dataFrame['Individual LDA classifier'] = TotalDataframe['AccLDAInd'].values
    dataFrame['Liu LDA classifier'] = TotalDataframe['AccLDALiu'].values
    dataFrame['Vidovic LDA classifier'] = TotalDataframe['AccLDAVidovic'].values

    dataFrame['Our QDA classifier'] = TotalDataframe['AccQDAProp'].values
    dataFrame['Individual QDA classifier'] = TotalDataframe['AccQDAInd'].values
    dataFrame['Liu QDA classifier'] = TotalDataframe['AccQDALiu'].values
    dataFrame['Vidovic QDA classifier'] = TotalDataframe['AccQDAVidovic'].values

    data = np.asarray(dataFrame)
    num_datasets, num_methods = data.shape
    print("Number of classifiers: ", num_methods,
          "Number of evaluations (3(feature sets) x [10(people NinaPro5) x 4(shots) + 17(people Cote) x 4(shots) 30(people EPN) x 25(shots)]): ",
          num_datasets, '\n')

    alpha = 0.05  # Set this to the desired alpha/signifance level

    stat, p = stats.friedmanchisquare(*data)

    reject = p <= alpha
    print("Should we reject H0 (i.e. is there a difference in the means) at the", (1 - alpha) * 100,
          "% confidence level?", reject, '\n')

    if not reject:
        print(
            "Exiting early. The rankings are only relevant if there was a difference in the means i.e. if we rejected h0 above")
    else:
        statistic, p_value, ranking, rank_cmp = friedman_test(*np.transpose(data))
        ranks = {key: rank_cmp[i] for i, key in enumerate(list(dataFrame.columns))}

        comparisons, z_values, p_values, adj_p_values, best = holm_test(ranks)

        adj_p_values = np.asarray(adj_p_values)

        for method, rank in ranks.items():
            print(method + ":", "%.2f" % rank)
        print('\n The best classifier is: ', best)
        holm_scores = pd.DataFrame({"p": adj_p_values, "sig": adj_p_values < alpha}, index=comparisons)
        print(holm_scores)

    return
