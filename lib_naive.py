import numpy as np

#Initializing the Naive Bayes Class
class NaiveBayes():

    #Class responsible for Naive Bayes
    def __init__(self):

        self.o = 0

    def fit(self,X, Y):

        self.labels = np.unique(Y)

        X = np.where(X > 1, 1, 0)

        self.pClasse = np.zeros((len(np.unique(Y)),1))

        for i in self.labels:
            self.pClasse[i] = sum(Y==i)/len(Y)
            
        self.pAtrLabel = np.zeros((X.shape))

        for i in self.labels:
            for j in range(len(X[0])):
                self.pAtrLabel[i][j] = sum(Y[np.where(X[:,j]==1)]==i)/sum(Y==i)

    def predict(self,x):

        probClasse = np.zeros((len(self.labels)))

        for i in range(len(self.labels)):
            probClasse[i] = np.prod(abs((1-x)-self.pAtrLabel[i]))*self.pClasse[i]

        return np.argmax(probClasse)
