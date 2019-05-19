import numpy as np #importa a biblioteca usada para trabalhar com vetores e matrizes
import pandas as pd #importa a biblioteca usada para trabalhar com dataframes (dados em formato de tabela) e análise de dados
from lib_knn import *

# Importa o arquivo e guarda em um dataframe do Pandas
df_dataset = pd.read_csv( 'dados.csv', sep=',', index_col=None)

print('Dados carregados com sucesso!')

# Pega os valores das n-1 primeiras colunas e guarda em uma matrix X
X = df_dataset.iloc[:, 0:-1].values 

# Pega os valores da ultima coluna e guarda em um vetor Y
Y = df_dataset.iloc[:, -1].values

m, n = X.shape # m = qtde de objetos e n = qtde de atributos por objet
# Inicializa as variaves de saída
X_norm = np.zeros( (m,n) ) #inicializa X_norm (base normalizada)
mu = 0 # inicializa a média
sigma = 1 # inicializa o desvio padrão
mu = X.mean(axis=0)
sigma = X.std(axis=0, ddof=1)
X_norm = (X - mu)/sigma

classifier = KNN(neigh=1)
classifier.fit(X_norm,Y)

x_teste = np.array(([[5.5, 3.2]]))
x_teste_norm = (x_teste-mu) / sigma

y_pred = classifier.predict(x_teste_norm)

if y_pred == 0:
    print('\tClasse 0 ==> Iris Versicolour.\n');
else:
    print('\tClasse 1 ==> Iris Setosa.\n');



