# -*- coding: utf-8 -*-

import numpy as np  # importa a biblioteca usada para trabalhar com vetores e matrizes
import pandas as pd # importa a biblioteca usada para trabalhar com dataframes (dados em formato de tabela) e análise de dados
from lib_mlp import *

# importa o arquivo e guarda em um dataframe do Pandas
df_dataset = pd.read_csv( 'dados_mlp.csv', sep=',', header=None) 

print('Dados carregados com sucesso!')

# Pega os valores das n-1 primeiras colunas e guarda em uma matrix X
X = df_dataset.iloc[:, 0:-1].values 

# Pega os valores da ultima coluna e guarda em um vetor Y
Y = df_dataset.iloc[:, -1].values

classifier = MLP(400,25,10)
classifier.fit(X,Y)
pred = classifier.predict(X)

print('\nAcuracia no conjunto de treinamento: %f\n'%( np.mean( pred == Y ) * 100) )

print('\nAcuracia esperada: 99.56% (aproximadamente)')
