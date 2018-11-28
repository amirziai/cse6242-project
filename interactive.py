import ast
import math

import numpy
import scipy
from scipy.cluster.vq import vq
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, silhouette_samples

import cmeans as Fuzzy
from app import scale_score

# confusion matrix
def computeX2(attrVals, clusters, data, N):
    M, k = attrVals.shape
    for j in range(M):
        Ptk = numpy.count_nonzero(data[:, j])
        Ptkp = N - Ptk
        # Ptk = numpy.where(data[:,j]>0)[0].size
        # Ptkp = numpy.where(data[:,j]==0)[0].size
        for p in range(k):
            clusterObjs = clusters[p]
            otherObjs = numpy.setdiff1d(range(N), clusterObjs)
            if clusterObjs.size == 0:
                P1 = 0
                P2 = 0
            else:
                temp = numpy.count_nonzero(data[clusterObjs, j])
                P1 = temp * 1.0 / clusterObjs.size
                P2 = (clusterObjs.size - temp) * 1.0 / clusterObjs.size

            if otherObjs.size == 0:
                P3 = 0
                P4 = 0
            else:
                temp = numpy.count_nonzero(data[otherObjs, j])
                P4 = temp * 1.0 / otherObjs.size
                P3 = (otherObjs.size - temp) * 1.0 / otherObjs.size

            Pci = clusterObjs.size * 1.0 / N
            Pcip = otherObjs.size * 1.0 / N

            if math.sqrt(Pci * Pcip * Ptk * Ptkp) == 0:
                attrVals[j, p] = 0
            else:
                attrVals[j, p] = (P1 * P3 - P4 * P2) / math.sqrt(Pci * Pcip * Ptk * Ptkp);
    return


f = 50  # # number of key terms for each cluster that will return to the user


def icluster(data, terms, userFeedbackTerm, k, userU=-1):
    N, M = data.shape

    if userU == +1:  # it means reclustering signal has been sent
        # clusterNames = eval(form.getvalue('serverClusterName'))
        userU = numpy.zeros((k, M), float)
        userFeedbackTermId = []
        for i in range(len(userFeedbackTerm)):
            tempArray = []
            if (len(userFeedbackTerm[i]) == 1):
                if (numpy.where(terms == userFeedbackTerm[i][0])[1].size > 0):
                    userU[i, numpy.where(terms == userFeedbackTerm[i][0])[1][0]] = 1
            else:
                step = 0.05  # the lower terms will recive lower value
                for j in range(len(userFeedbackTerm[i])):
                    if (numpy.where(terms == userFeedbackTerm[i][j])[1].size > 0):
                        userU[i, numpy.where(terms == userFeedbackTerm[i][j])[1][0]] = max(1 - j * step, 0.5)

    docs = numpy.arange(1, N + 1).reshape((1, N))

    Vars = numpy.var(data, axis=0).transpose()
    options = (1.1, 25, 0.01, 0)
    keyterms = []
    clusterKeyterms = []
    clusterDocs = []

    realK = 0
    while (
            realK < k):  # in case the number of clusters are less than user specified, it will recluster until it gets the appropriate number.
        idp = []
        selectedCentroids = numpy.empty([k, M], dtype=float)
        fcm = Fuzzy.FuzzyCMeans(data.transpose(), k, options[0], 'cosine', userU, options[1], options[2])
        fcm()
        bestU = fcm.mu  # .transpose()
        for p in range(k):
            sortIDX = numpy.argsort(bestU[p, :])
            sortV = numpy.sort(bestU[p, :])
            tempIndex = numpy.argmax(sortV > (1.0 / k))
            idp.append(sortIDX[tempIndex:])

        for p in range(k):
            idx = []
            idpp = idp[p]

            Varsp = Vars[idpp]
            meanVarsp = numpy.mean(Varsp)
            tempIndex = numpy.where(Varsp >= meanVarsp)[0]
            keyTerms = idpp[tempIndex]

            newDataset = data[:, keyTerms]
            sumDataset = numpy.mean(newDataset, axis=1)

            temp, label = scipy.cluster.vq.kmeans2(sumDataset, 2, iter=50, thresh=1e-03, minit='random', missing='warn')
            idx.append(numpy.where(label == 0)[0])
            idx.append(numpy.where(label == 1)[0])
            if (idx[0].size == 0):
                relDocs = idx[1]
            elif (idx[1].size == 0):
                relDocs = idx[0]
            else:
                if (idx[0].size >= idx[1].size):
                    relDocs = idx[1]
                else:
                    relDocs = idx[0]
            selectedCentroids[p, :] = numpy.mean(data[relDocs, :], axis=0)
        Y = cdist(data, selectedCentroids, 'cosine')

        minY = numpy.min(Y, axis=1)
        maxY = numpy.max(Y, axis=1)
        maxMmin = maxY - minY
        minY = numpy.kron(numpy.ones((k, 1)), minY).transpose()
        maxMmin = numpy.kron(numpy.ones((k, 1)), maxMmin).transpose()
        tempY = numpy.multiply((Y - minY), numpy.power(maxMmin, -1.0))
        tempY = 1 - tempY

        threshold = 0.95
        tempY = (tempY > threshold)
        clusters = []
        for p in range(k):
            clusters.append(numpy.where(tempY[:, p])[0])

        realK = 0
        IDX = numpy.argmin(Y, axis=1)
        newclusters = []
        for p in range(k):
            newclusters.append(numpy.where(IDX == p)[0])
            if (len(newclusters[p]) > 0):
                realK = realK + 1
        del newclusters

    silhouette_avg = silhouette_score(data, IDX, 'cosine')
    sample_silhouette_values = silhouette_samples(data, IDX, 'cosine')
    scores = dict()
    for i, label in enumerate(IDX):
        ith_cluster_silhouette_values = sample_silhouette_values[IDX == label]
        avg = numpy.mean(ith_cluster_silhouette_values)
        scores[str(label)] = scale_score(avg)
    attrVals = numpy.empty([M, k], dtype=float)
    computeX2(attrVals, clusters, data, N)
    for p in range(k):
        temp = numpy.argsort(attrVals[:, p])
        temp = temp[::-1]
        keyterms.append(temp[range(f)])

    for p in range(k):
        tempStr = '['
        comma = ''
        for j in range(len(keyterms[p])):
            tempStr += comma + '\"' + terms[0, keyterms[p][j]] + '\"'
            comma = ','
        tempStr += ']'
        clusterKeyterms.append(tempStr)

    for p in range(k):
        tmp = []

        for j in range(len(clusters[p])):
            tmp.append(docs[0, clusters[p][j]])

        clusterDocs.append(tmp)

    clusterKeyterms = [ast.literal_eval(x) for x in clusterKeyterms]
    
    # clusterDocs = [ast.literal_eval(x) for x in clusterDocs]
    return clusterDocs, clusterKeyterms, keyterms, silhouette_avg, scores
