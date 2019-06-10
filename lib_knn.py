import numpy as np

#Initializing the KNN Class
class KNN():

    #Class responsible for K-nearest neighbors
    #Initialize only with the number os neighbors, default =1
    def __init__(self, neigh=3):
        self.k = neigh
        self.X = 0.0
        self.Y = 0.0
        self.X_pred = 0.0
    #Help function to calculate the distance
    def distancia(self,X_pred,X):

        m = self.X.shape[0] # Quantidade de objetos em X
        D = np.zeros(m) # Inicializa a matriz de distâncias D

        for i in range(m):
            dist = (self.X_pred - self.X[i])**2
            dist = dist.sum()
            D[i] = np.sqrt(dist)

        return D
    #Fit method to store the dataset
    def fit(self, X, Y):

        self.X = X
        self.Y = Y
        
    #Predict method which return the K-nearest neighbor for given sample
    def predict(self, X_pred):

        self.X_pred = X_pred

        dist = self.distancia(self.X_pred,self.X)
        kviz = dist.argsort()

        ind_viz = kviz[0:self.k]
        y1 = self.Y[ind_viz]
        vote = np.bincount(y1)
        label = np.argmax(vote)

        return label

        
