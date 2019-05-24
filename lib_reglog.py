import numpy as np

#Initializing the KNN Class
class RegressaoLogistica():

    #Class responsible for Logistic Regression
    def __init__(self, lr=0.01, num_iter=100000):

        self.lr = lr
        self.num_iter = num_iter


    def sigmoid(self,z):

        if isinstance(z, int):
            g = 0

        else:
            g = np.zeros( z.shape )

        g = 1/(1+np.exp(-z))

        return g


    def fit(self,X,y):

        self.uniques = np.unique(y) 

        lambda_reg = 1
        m = len(y)
        J = 0
        self.theta = np.zeros((len(np.unique(y)),X.shape[1]))
        grad = np.zeros( len(self.theta[1]) )
        eps = 1e-15

        for j in np.unique(y):

            positivo = np.where(y==j,1,0)

            for i in range(self.num_iter):

                h = self.sigmoid(np.dot(X, self.theta[j]))
                J = (1/m)*(np.sum((-y*np.log(h)) - (1-y)*np.log((1 - h)+eps)))
                R = (lambda_reg/(2*m))*np.sum(self.theta[j][1:]**2)

                J = J + R
                grad[0] = (X.T.dot(h - y))[0]/m
                grad[1:] = (X.T.dot(h - y))[1:]/m + (lambda_reg/m)*self.theta[j][1:]
                self.theta[j] -= self.lr * grad

    def predict(self,X):

        res = np.zeros((len(X[:,0]),len(self.uniques)))
        total = np.zeros(len(X[:,0]))

        for j in range(len(X[:,0])):

            for i in self.uniques:

                m = X[j].shape[0]
                p = np.zeros(m, dtype=int)
                p = self.sigmoid(np.dot(X[j], self.theta[i]))
                if p >= 0.5:
                    p=1
                else:
                    p=0
                res[j][i] = p

            total[j] = np.amax(res[j],axis=0)

        return total

        
        

        

        

        

    
