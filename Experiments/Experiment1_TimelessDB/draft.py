#
# for
#     c = dot(X, X_T.conj())
#     c *= np.true_divide(1, fact)
#
# X=data.T
# X=X-mean[:, None]
# c = np.dot(X, X.T.conj())
# fact=len(data[:,0])-1
# c=c*np.true_divide(1, fact)
#
# X=data.T
# X=X-mean[:, None]
# a = np.dot(X, X.T.conj())
# fact=len(data[:,0])-1
# a*np.true_divide(1, fact)
#
# X=x.T
# X=X-mean
# X=np.resize(X,(len(X),1))
# CK=np.dot(X, X.T.conj())*(((len(data[:,0])))/(len(data[:,0])+1))
# cov_1=((len(data[:,0])-1)/(len(data[:,0])))*cov+(1/(len(data[:,0])))*CK

# ###############COV
# labeledGesturesFeatures_1 = labeledGesturesFeatures[labeledGesturesLabels == 1]
# covNew = np.zeros((8, 8))
# for i in range(len(labeledGesturesFeatures_1)):
#     xNew = labeledGesturesFeatures_1[i, :]
#     x_mean = np.resize(xNew - adaptiveModel['mean'].loc[cla], (len(xNew), 1))
#     covNew += np.dot(x_mean, x_mean.T.conj())
#
# x_mean = np.resize(x - adaptiveModel['mean'].loc[cla], (len(x), 1))
# covNew += p * np.dot(x_mean, x_mean.T.conj())
# covNew = covNew / (N + p - 1)



def model_semi_gestures_labels(weakModel, classes, trainFeatures, postProb_trainFeatures, fewModel,
                              labeledGesturesFeatures, labeledGesturesLabels, type_DA, samplesInMemory, shotStart,
                              unlabeledGesturesTotal, dataTotal):
    t = time.time()

    adaptiveModel = weakModel.copy()
    for cla in range(classes):
        adaptiveModel.at[cla, '# gestures'] = adaptiveModel.loc[0, '# gestures']

    unlabeledGesturesTotal = np.array(unlabeledGesturesTotal)
    numberFeatures = np.size(labeledGesturesFeatures, axis=1)
    labelsList = list(labeledGesturesLabels)
    featuresList = list(labeledGesturesFeatures)
    for sample in range(len(unlabeledGesturesTotal[:, 0])):
        x = unlabeledGesturesTotal[sample, :numberFeatures]
        if type_DA == 'LDA':
            cla = DA_Classifiers.predictedModelLDA(x, weakModel, classes,
                                                   DA_Classifiers.LDA_Cov(weakModel, classes))
        elif type_DA == 'QDA':
            cla = DA_Classifiers.predictedModelQDA(x, weakModel, classes)
        labelsList.append(cla)
        featuresList.append(x)
    return modelCalculation_labels(np.array(featuresList), np.array(labelsList), classes), time.time() - t, 0
