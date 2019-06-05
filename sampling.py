import numpy as np

def stratified_kfolds(target, k, classes):

    folds_final = np.zeros( k,dtype='object')
    train_index = np.zeros( k,dtype='object')
    test_index = np.zeros( k,dtype='object')

    for i in folds_final:

        train_index[i] = [] # indices dos dados de treinamento relativos ao fold i
        test_index[i] = [] # indices dos dados de teste relativos ao fold i

        folds_final[i] = np.array( [train_index[i],test_index[i]] )

    p = 1/k
    m = [len(np.where(target == c)[0]) for c in classes]

    for i in range(k):
        test_index[i] = []
        for j, c in enumerate(classes):
            test_idx       = np.where(target == c)[0][int(i*p*m[j]):int((i+1)*p*m[j])]
            test_index[i]  = np.union1d(test_index[i] , test_idx).astype('int')
            train_index[i] = np.union1d(train_index[i], np.setdiff1d(np.where(target == c)[0], test_idx)).astype('int')
        folds_final[i] = (train_index[i], test_index[i])

    return folds_final
