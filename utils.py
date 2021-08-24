# Copyright (c) Alibaba Group Holding Limited. All Rights Reserved
import numpy as np

def retrieval(X, Y, Kset):
    num = X.shape[0]
    kmax = np.max(Kset)
    recallK = np.zeros(len(Kset))
    #compute Recall@K
    sim = X.dot(X.T)
    minval = np.min(sim) - 1.
    sim -= np.diag(np.diag(sim))
    sim += np.diag(np.ones(num) * minval)
    indices = np.argsort(-sim, axis=1)[:, : kmax]
    YNN = Y[indices]
    for i in range(0, len(Kset)):
        pos = 0.
        for j in range(0, num):
            if Y[j] in YNN[j, :Kset[i]]:
                pos += 1.
        recallK[i] = pos/num
    return recallK
