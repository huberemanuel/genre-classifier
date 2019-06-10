import numpy as np
import scipy
import scipy.optimize

class MLP():

    def __init__(self, input_layer, hidden_layer, labels):

        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.labels = labels
        self.theta1 = np.random.rand(self.hidden_layer, 1 + self.input_layer) * 2 * 0.12 - 0.12
        self.theta2 = np.random.rand(self.labels, 1 + self.hidden_layer) * 2 * 0.12 - 0.12
        self.initial_rna_params = np.concatenate([np.ravel(self.theta1), np.ravel(self.theta2)])


    def sigmoid(self,z):

        z = 1/(1+np.exp(-z))

        return z

    def sigmoidGradient(self,z):

        g = np.zeros(z.shape)
        g = self.sigmoid(z)*(1-self.sigmoid(z))

        return g

    def funcaoCusto(self,nn_params, input_layer_size, hidden_layer_size, num_labels, X, y):

        # Extrai os parametros de nn_params e alimenta as variaveis Theta1 e Theta2.
        Theta1 = np.reshape( nn_params[0:hidden_layer_size*(input_layer_size + 1)], (hidden_layer_size, input_layer_size+1) )
        Theta2 = np.reshape( nn_params[ hidden_layer_size*(input_layer_size + 1):], (num_labels, hidden_layer_size+1) )

        # Qtde de amostras
        m = X.shape[0]
        eps = 1e-15
                 
        # A variavel a seguir precisa ser retornada corretamente
        J = 0;

        Y = np.eye(num_labels+1)[y]
            
        Y = np.delete(Y,(0), axis=1)
         
        a1 = np.c_[np.ones((m,1)),X]

        a2 = self.sigmoid(np.dot(a1,Theta1.T))
            
        a2 = np.c_[np.ones((m,1)),a2]
            
        h = self.sigmoid(np.dot(a2,Theta2.T))

        J = (1/m)*(np.sum(np.sum((-Y*np.log(h+eps)) - (1-Y)*np.log((1 - h)+eps))))

        return J

    def funcaoCusto_reg(self,nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, vLambda):

        # Extrai os parametros de nn_params e alimenta as variaveis Theta1 e Theta2.
        Theta1 = np.reshape( nn_params[0:hidden_layer_size*(input_layer_size + 1)], (hidden_layer_size, input_layer_size+1) )
        Theta2 = np.reshape( nn_params[ hidden_layer_size*(input_layer_size + 1):], (num_labels, hidden_layer_size+1) )

        # Qtde de amostras
        m = X.shape[0]
             
        # A variavel a seguir precisa ser retornada corretamente
        J = 0;
        Theta_reg1 = Theta1[:,1:]
        Theta_reg2 = Theta2[:,1:]
        
        J = self.funcaoCusto(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y)
        
        reg = (vLambda/(2*m))*((np.sum(np.sum(Theta_reg1**2)))+(np.sum(np.sum(Theta_reg2**2))))
        
        J = J + reg    

        return J


    def funcaoCusto_backp_reg(self,nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, vLambda):

        # Extrai os parametros de nn_params e alimenta as variaveis Theta1 e Theta2.
        Theta1 = np.reshape( nn_params[0:hidden_layer_size*(input_layer_size + 1)], (hidden_layer_size, input_layer_size+1) )
        Theta2 = np.reshape( nn_params[ hidden_layer_size*(input_layer_size + 1):], (num_labels, hidden_layer_size+1) )

        # Qtde de amostras
        m = X.shape[0]
             
        # As variaveis a seguir precisam ser retornadas corretamente
        J = 0;
        Theta1_grad = np.zeros(Theta1.shape)
        Theta2_grad = np.zeros(Theta2.shape)

        Y = np.eye(num_labels+1)[y]
        
        Y = np.delete(Y,(0), axis=1)
        
        J = self.funcaoCusto_reg(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, vLambda)

        a1 = np.c_[np.ones((m,1)),X]

        a2 = self.sigmoid(np.dot(a1,Theta1.T))
        
        a2 = np.c_[np.ones((m,1)),a2]
        
        a3 = self.sigmoid(np.dot(a2,Theta2.T))
        delta1 = np.zeros(Theta1.shape)
        delta2 = np.zeros(Theta2.shape)
        
       
        for i in range(m):
            
            a1n = a1[[i],:]
            a2n = a2[[i],:]
            a3n = a3[[i],:]
            yk = Y[[i],:]       
            
            error3 = a3n - yk
            
            sig1 = np.c_[np.ones((1,1)),self.sigmoidGradient(np.dot(a1n,Theta1.T))]

            error2 = np.dot(Theta2.T,error3.T)*sig1.T
            
            delta1 = delta1 + error2[1:]*a1n
            delta2 = delta2 + error3.T*a2n
            
        Theta1_grad = (1/m)*delta1
        Theta2_grad = (1/m)*delta2
            
        Theta1_grad[:,1:] = ((1/m)*delta1[:,1:])+((vLambda/m)*Theta1[:,1:])
        Theta2_grad[:,1:] = ((1/m)*delta2[:,1:])+((vLambda/m)*Theta2[:,1:])

        # Junta os gradientes
        grad = np.concatenate([np.ravel(Theta1_grad), np.ravel(Theta2_grad)])

        return J, grad


    def fit(self, X, Y):

        print('\nTreinando a rede neural.......')
        print('.......(Aguarde, pois esse processo pode ser um pouco demorado.)\n')

        # Apos ter completado toda a tarefa, mude o parametro MaxIter para
        # um valor maior e verifique como isso afeta o treinamento.
        MaxIter = 500

        # Voce tambem pode testar valores diferentes para lambda.
        vLambda = 1

        # Minimiza a funcao de custo
        result = scipy.optimize.minimize(fun=self.funcaoCusto_backp_reg, x0=self.initial_rna_params, args=(self.input_layer, self.hidden_layer, self.labels, X, Y, vLambda),  
                        method='TNC', jac=True, options={'maxiter': MaxIter})

        # Coleta os pesos retornados pela função de minimização
        nn_params = result.x

        # Obtem Theta1 e Theta2 back a partir de rna_params
        self.theta1 = np.reshape( nn_params[0:self.hidden_layer*(self.input_layer + 1)], (self.hidden_layer, self.input_layer+1) )
        self.theta2 = np.reshape( nn_params[ self.hidden_layer*(self.input_layer + 1):], (self.labels, self.hidden_layer+1) )


    def predict(self,X):


        m = X.shape[0] # número de amostras
        self.labels = self.theta2.shape[0]
                            
        p = np.zeros(m)

        a1 = np.hstack( [np.ones([m,1]),X] )
        h1 = self.sigmoid( np.dot(a1,self.theta1.T) )

        a2 = np.hstack( [np.ones([m,1]),h1] ) 
        h2 = self.sigmoid( np.dot(a2,self.theta2.T) )
                            
        p = np.argmax(h2,axis=1)
        p = p+1

        return p

        
