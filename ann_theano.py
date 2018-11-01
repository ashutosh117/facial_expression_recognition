from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

#function to generate weights
def weights_init(m1,m2):
    w = np.random.randn(m1,m2)*np.sqrt(2.0/m1)
    b = np.zeros(m2)
    return w,b

#hidden layer class
class HiddenLayer(object):
    def __init__(self,m1,m2,f):
        self.m1 = m1
        self.m2 = m2
        self.f = f
        w_init,b_init = weights_init(self.m1,self.m2)
        self.w = theano.shared(w_init)
        self.b = theano.shared(b_init)
        self.params = [self.w,self.b]
        
    def forward(self,X):
        return self.f(X.dot(self.w) + self.b)
    
    
#class
class ANN(object):
    def __init__(self,hls):
        self.hls = hls
        
    def fit(self,X,Y,X_valid,Y_valid,batch_sz,epochs = 10,lr = 1e-3,reg = 0.0,print_period = 1,show_fig=False):
        self.n,self.m = X.shape
        self.p = Y.shape[1]
        
        #create a list of all hidden layers
        self.layers = []
        m1 = self.m
        for m2 in self.hls:
            h = HiddenLayer(m1,m2,T.nnet.relu)
            self.layers.append(h)
            m1 = m2
            
        #add final layer to the layers list
        m2 = self.p
        h = HiddenLayer(m1,m2,T.nnet.softmax)
        self.layers.append(h)
        
        #create a list of all parameters
        self.all_params = []
        for layer in self.layers:
            self.all_params += layer.params
        
        #placeholder matrices
        thX = T.matrix()
        thY = T.matrix()
        
        #forward propagation
        Y_hat = self.forward(thX)
        
        #compute predictions
        predictions = T.argmax(Y_hat,axis=1)
        
        #compute cost
        reg_cost = T.mean([(p*p).sum() for p in self.all_params])
        cost = -T.mean(thY*T.log(Y_hat)) + reg*reg_cost
        
        #compute gradients
        grads = T.grad(cost,self.all_params)
        
        #updates parameters
        updates = [
            (p,p - lr*dp) for p,dp in zip(self.all_params,grads)
        ]
        
        #train function
        train = theano.function(inputs = [thX,thY],updates = updates,outputs = cost)
        
        #predict function
        self.predict_op = theano.function(inputs = [thX],outputs = predictions)
        n_iter = int(self.n/batch_sz)
        train_loss = []
        valid_err = []
        train_err = []
        for epoch in range(epochs):
            for iter in range(n_iter):
                #create training batch
                X_batch = X[(iter*batch_sz):((n_iter*batch_sz)+batch_sz),:]
                Y_batch = Y[(iter*batch_sz):((n_iter*batch_sz)+batch_sz),:]
                
                #train the model
                l = train(X_batch,Y_batch)
                train_loss.append(l)
                train_preds = self.predict_op(X_batch)
                t_err = np.mean(train_preds != np.argmax(Y_batch,axis=1))
                train_err.append(t_err)
            #print loss info
            if epoch % print_period == 0:
                preds = self.predict_op(X_valid)
                #print preds
                #print np.argmax(Y_valid,axis=1)
                err = np.mean(preds != np.argmax(Y_valid,axis=1))
                valid_err.append(err)
                print 'Epoch : %d, Training loss : %.4f,Validation Error : %.4f' %(epoch,l,err)
                print 'Train Error : %.4f' %train_err[-1]
        if show_fig:
            plt.plot(train_loss,label = 'Training loss')
            plt.legend()
            
        
    def forward(self,X):
        Z = X
        for h in self.layers:
            Z = h.forward(Z)
        return Z
    
    def get_predict(self,X):
        return self.predict_op(X)