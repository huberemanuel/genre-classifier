import numpy as np #importa a biblioteca usada para trabalhar com vetores de matrizes
import pandas as pd #importa a biblioteca usada para trabalhar com dataframes (dados em formato de tabela) e análise de dados
from lib_reglog import *

# Importa o arquivo e guarda em um dataframe do Pandas
df_dataset = pd.read_csv( 'dados_reglog.csv', sep=',', index_col=None)

print('Dados carregados com sucesso!')

# pega os valores das n-1 primeiras colunas e guarda em uma matrix X
X = df_dataset.iloc[:, 0:-1].values 

# pega os valores da última coluna e guarda em um vetor Y
Y = df_dataset.iloc[:, -1].values 
m, n = X.shape # m = qtde de objetos e n = qtde de atributos por objeto

X = np.column_stack( (np.ones(m),X) ) # Adiciona uma coluna de 1s em x
logmodel = RegressaoLogistica()
logmodel.fit(X,Y)

p1 = logmodel.predict(X)


acuracia = np.mean(p1 == Y) * 100
print('\nAcuracia obtida na base de treinamento: %1.2f\n' %acuracia);

comprimento_petala = 2.5
largura_petala = 4.0

X1 = np.array( [[1,comprimento_petala,largura_petala]] )

# faz a predição do novo dado
p = logmodel.predict(X1)
print(p)

if p[0] == 1:
    print('Classe = Iris Setosa (y = 1)')
else:
    print('Classe = Iris Versicolour (y = 0)')

