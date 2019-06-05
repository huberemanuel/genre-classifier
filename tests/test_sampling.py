from sampling import stratified_kfolds
import pandas as pd
import numpy as np

df_dataset = pd.read_csv( 'data.csv', sep=',', index_col=None)
df_dataset3 = pd.read_csv( 'data.csv', sep=',', index_col=None)

X = df_dataset.iloc[:, 0:-1].values
Y = df_dataset.iloc[:, -1].values
X3 = df_dataset.iloc[:, 0:-1].values
Y3 = df_dataset.iloc[:, -1].values
classes = np.unique(Y3)

randomSeed = 10

idx_perm = np.random.RandomState(randomSeed).permutation(range(len(Y3)))

X4, Y4 = X3[idx_perm, :], Y3[idx_perm]

nFolds = 5
folds = stratified_kfolds(Y4, nFolds, classes)

k = 1

for train_index, test_index in folds:

    print('\n-----------\n%d-fold: \n-----------\n' % (k) )

    if len(train_index)==0 or len(test_index)==0:
        print('\tErro: o vetor com os indices de treinamento ou o vetor com os indices de teste esta vazio')
        break

    X_train, X_test = X4[train_index, :], X4[test_index, :];
    Y_train, Y_test = Y4[train_index], Y4[test_index];

    print('\tQtd. de dados de teste: %d (%1.2f%%)' %(X_test.shape[0], (X_test.shape[0]/X.shape[0])*100) )

    # imprime a porcentagem de dados de treinamento de cada classe
    print("\n\tQtd. de dados de cada classe (treinamento)")
    cTrain, counts_cTrain = np.unique(np.sort(Y_train), return_counts=True)
    for i in range( len(cTrain) ):
        print('\t\tClasse %s: %d (%1.2f%%)' %( cTrain[i],counts_cTrain[i],(counts_cTrain[i]/len(Y_train))*100 ) )

    # imprime a porcetagem de dados de teste de cada classe
    print("\n\tQtd. de dados de cada classe (teste)")
    cTest, counts_cTest = np.unique(np.sort(Y_test), return_counts=True)
    for i in range( len(cTrain) ):
        print('\t\tClasse %s: %d (%1.2f%%)' %( cTest[i],counts_cTest[i],(counts_cTest[i]/len(Y_test))*100 ) )

    k = k + 1


print('\n\n\n'+"-"*80+'\nSe sua implementacao estiver corretas, cada uma das 5 rodadas deve conter:')
print('\t- 304 dados de treinamento da classe 0 (44.19%)')
print('\t- 384 dados de treinamento da classe 1 (55.81%)')

print('\nSe sua implementacao estiver correta, cada fold deve conter:')
print('\t- 76 dados de teste da classe 0 (44.19%)')
print('\t- 96 dados de teste da classe 1 (55.81%)')
