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
# covsample = np.zeros((8, 8))
# for i in range(len(labeledGesturesFeatures_1)):
#     xsample = labeledGesturesFeatures_1[i, :]
#     x_mean = np.resize(xsample - adaptiveModel['mean'].loc[cla], (len(xsample), 1))
#     covsample += np.dot(x_mean, x_mean.T.conj())
#
# x_mean = np.resize(x - adaptiveModel['mean'].loc[cla], (len(x), 1))
# covsample += p * np.dot(x_mean, x_mean.T.conj())
# covsample = covsample / (N + p - 1)
import numpy as np

data = np.array([[1, 2, 3, 4], [2, 5, 6, 3], [2, 7, 8, 9], [12, 34, 5, 6], [2, 5, 6, 2], [34, 5, 2, 7], [3, 5, 6, 3]])
cov_data = np.cov(data, rowvar=False)
mean_data = np.mean(data, axis=0)
N_data = 7

##################sample
data_sample = np.array(
    [[1, 2, 3, 4], [2, 5, 6, 3], [2, 7, 8, 9], [12, 34, 5, 6], [2, 5, 6, 2], [34, 5, 2, 7], [3, 5, 6, 3],
     [2, 3, 1, 78]])
cov_data_sample = np.cov(data_sample, rowvar=False)
mean_data_sample = np.mean(data_sample, axis=0)
N_data_sample = 8

sample = np.array([2, 3, 1, 78])

# mean_data_sample_andres = (N_data * mean_data + sample) / (N_data + 1)
# print('error mean sample', mean_data_sample - mean_data_sample_andres)
#
# x_mean = np.resize(sample - mean_data, (len(sample), 1))
# cov_data_sample_andres = ((N_data - 1) / N_data) * cov_data + (1 / (N_data + 1)) * np.dot(x_mean, x_mean.T.conj())
# print('error cov sample', cov_data_sample - cov_data_sample_andres)
#
# cov_data_sample_pang = cov_data + (N_data / (N_data + 1)) * np.dot(x_mean, x_mean.T.conj())
# print('error cov sample pang', cov_data_sample - cov_data_sample_pang)
#
# cov_data_sample_pang_n = (cov_data + (N_data / (N_data + 1)) * np.dot(x_mean, x_mean.T.conj())) / (N_data)
# print('error cov sample pang n', cov_data_sample - cov_data_sample_pang_n)
#
# cov_data_sample_pang_n1 = (cov_data + (N_data / (N_data + 1)) * np.dot(x_mean, x_mean.T.conj())) / (N_data - 1)
# print('error cov sample pang n-1', cov_data_sample - cov_data_sample_pang_n1)
#
# cov_data_sample_chen = (N_data / (N_data + 1)) * cov_data + (1 / (N_data + 1)) * (N_data / (N_data + 1)) * np.dot(
#     x_mean, x_mean.T.conj())
# print('error cov sample chen', cov_data_sample - cov_data_sample_chen)
#
# #################chunk
# chunk = np.array([[2, 3, 1, 78], [3, 4, 5, 6], [3, 5, 1, 8]])
# N_chunk = 3
# pr = 1
# prob = [1, 1, 1, 1, 1, 1, 1, pr, pr, pr]
#
# cov_chunk = np.cov(chunk, rowvar=False)
# mean_chunk = np.mean(chunk, axis=0)
#
# data_chunk = np.vstack((data, chunk))
# cov_data_chunk = np.cov(data_chunk, rowvar=False)
# mean_data_chunk = np.mean(data_chunk, axis=0)
#
# mean_data_chunk_andres = (N_data * mean_data + N_chunk * mean_chunk) / (N_data + N_chunk)
# print('error mean chunk', mean_data_chunk - mean_data_chunk_andres)
#
# x_mean = np.resize(mean_chunk - mean_data, (len(sample), 1))
# cov_data_chunk_andres = ((N_data - 1) / (N_data + N_chunk - 1)) * cov_data + \
#                         ((N_chunk - 1) / (N_data + N_chunk - 1)) * cov_chunk + \
#                         (N_chunk * N_data / ((N_data + N_chunk) * (N_data + N_chunk - 1))) * np.dot(x_mean,
#                                                                                                     x_mean.T.conj())
# print('error cov chunk', cov_data_chunk - cov_data_chunk_andres)
#
# mean_data_chunk_w = np.zeros(4)
# for i in range(10):
#     mean_data_chunk_w += data_chunk[i, :] * prob[i]
# mean_data_chunk_w /= np.sum(prob)
#
# mean_data_chunk_w_andres = (N_data * mean_data + N_chunk * mean_chunk * pr) / (N_data + N_chunk * pr)
# print('error mean chunk w', mean_data_chunk_w - mean_data_chunk_w_andres)
#
# cov_data_chunk_w = np.zeros((4, 4))
# for i in range(10):
#     x_mean = np.resize(data_chunk[i, :] - mean_data_chunk_w, (len(data_chunk[i, :]), 1))
#     cov_data_chunk_w += np.dot(x_mean, x_mean.T.conj()) * prob[i]
# cov_data_chunk_w /= (np.sum(prob) - 1)
#
# x_mean = np.resize(mean_chunk - mean_data, (len(sample), 1))
# cov_data_chunk_w_andres = ((N_data - 1) / (N_data + N_chunk * pr - 1)) * cov_data + \
#                           (pr * (N_chunk - 1) / (N_data + N_chunk * pr - 1)) * cov_chunk + \
#                           (N_chunk * N_data * pr / ((N_data + N_chunk * pr) * (N_data + N_chunk * pr - 1))) * \
#                           np.dot(x_mean, x_mean.T.conj())
# cov_data_chunk_w_andres_mean = (N_data * cov_data + N_chunk * pr * cov_chunk) / \
#                                (N_data + N_chunk * pr)
# print('error cov chunk w', cov_data_chunk_w - cov_data_chunk_w_andres)
# print('error cov chunk w meanMethod', cov_data_chunk_w - cov_data_chunk_w_andres_mean)

#################chunk 222222222222222222222222
chunk = np.array([[2, 3, 1, 78], [3, 4, 5, 6], [3, 5, 1, 8]])
N_chunk = 3

cov_chunk = np.cov(chunk, rowvar=False)
mean_chunk = np.mean(chunk, axis=0)

data_chunk = np.vstack((data, chunk))
cov_data_chunk = np.cov(data_chunk, rowvar=False)
mean_data_chunk = np.mean(data_chunk, axis=0)

pr = 1
x_mean = np.resize(mean_chunk - mean_data, (len(sample), 1))
cov_data_chunk_andres2 = ((N_data - 1) / (N_data + N_chunk * pr - 1)) * cov_data + \
                           (pr * (N_chunk - 1) / (N_data + N_chunk * pr - 1)) * cov_chunk + \
                           (N_chunk * N_data * (N_chunk + N_data * pr ) / (
                                       (N_data + N_chunk) * (N_data + N_chunk) * (N_data + N_chunk * pr - 1))) * \
                           np.dot(x_mean, x_mean.T.conj())
print('error cov chunk 2', cov_data_chunk - cov_data_chunk_andres2)

pr = 0.2
prob = [1, 1, 1, 1, 1, 1, 1, pr, pr, pr]

mean_data_chunk_noProb = np.zeros(4)
for i in range(10):
    mean_data_chunk_noProb += data_chunk[i, :]
mean_data_chunk_noProb /= 10

cov_data_chunk_noProb = np.zeros((4, 4))
for i in range(10):
    x_mean = np.resize(data_chunk[i, :] - mean_data_chunk_noProb, (len(data_chunk[i, :]), 1))
    cov_data_chunk_noProb += np.dot(x_mean, x_mean.T.conj()) * prob[i]
cov_data_chunk_noProb /= (np.sum(prob) - 1)


x_mean = np.resize(mean_chunk - mean_data, (len(sample), 1))
cov_data_chunk_w_andres2 = ((N_data - 1) / (N_data + N_chunk * pr - 1)) * cov_data + \
                           (pr * (N_chunk - 1) / (N_data + N_chunk * pr - 1)) * cov_chunk + \
                           (N_chunk * N_data * (N_chunk + N_data * pr ) / (
                                       (N_data + N_chunk) * (N_data + N_chunk) * (N_data + N_chunk * pr - 1))) * \
                           np.dot(x_mean, x_mean.T.conj())



print('error cov chunk w2', cov_data_chunk_noProb - cov_data_chunk_w_andres2)

mean = mean_data
cov = cov_data
N = N_data
w = 0.2

testA = (N * mean + N_chunk * w * mean_chunk) / \
        (N + N_chunk * w)
aux = np.resize(mean_chunk - mean, (len(mean_chunk), 1))
testC = (1 / (N + N_chunk * w - 1)) * \
        (cov * (N - 1) + cov_chunk * w * (N_chunk - 1) +
         np.dot(aux, aux.T.conj()) * (N * N_chunk * w) / (N + N_chunk * w))
