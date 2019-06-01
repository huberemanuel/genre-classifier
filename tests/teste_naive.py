import numpy as np #importa a biblioteca usada para trabalhar com vetores de matrizes
import pandas as pd #importa a biblioteca usada para trabalhar com dataframes (dados em formato de tabela) e análise de dados
from lib_naive import *

# Importa o arquivo e guarda em um dataframe do Pandas
df_dataset = pd.read_csv( 'dados_naive.csv', sep=',', index_col=None)

print('Dados carregados com sucesso!')

# pega os valores das n-1 primeiras colunas e guarda em uma matrix X
X = df_dataset.iloc[:, 0:-1].values 

# pega os valores da última coluna e guarda em um vetor Y
Y = df_dataset.iloc[:, -1].values

classifier = NaiveBayes()

classifier.fit(X,Y)

resultados = np.zeros( X.shape[0] )

for i in range(X.shape[0]):
    resultados[i] = classifier.predict( X[i,:])

acuracia = np.sum(resultados==Y)/len(Y)

print('\n\nAcuracia esperada para essa base = %.2f%%\n' %76.60);
print('Acuracia obtida pelo seu classificador foi = %.2f%%\n' %( acuracia*100 ))


x1_novo = np.array([0,1,1,0,1])

y_pred = classifier.predict(x1_novo)

print(y_pred)
