#class based ANN using tensorflow
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle

#function to generate initial weights
def weights_init(m1,m2):
    w = np.random.randn(m1,m2)*np.sqrt(2.0/m1)
    b = np.zeros(m2)
    return w,b

#Hidden layer class
class HiddenLayer(object):
    def __init__(self,m1,m2,f):
        self.m1 = m1
        self.m2 = m2
        self.f = f
        w_init,b_init = weights_init(self.m1,self.m2)
        self.w = tf.Variable(w_init.astype(np.float32))
        self.b = tf.Variable(b_init.astype(np.float32))
        self.params = [self.w,self.b]
        
    def forward(self,X):
        if self.f != None:
            return self.f(tf.matmul(X,self.w) + self.b)
        else:
            return tf.matmul(X,self.w) + self.b
        
#ANN class
class ANN(object):
    #hls : hidden layer sizes
    def __init__(self,hls):
        self.hls = hls
        
    def fit(self,X,Y,X_valid,Y_valid,batch_sz,epochs = 10,lr = 0.01,reg = 0.1,decay = 0.99,mu = 0.99,print_period = 1,show_fig = False):
        self.n,self.m = X.shape
        self.p = Y.shape[1]
        
        #initialize all hidden layers
        self.layers = []
        m1 = self.m
        for m2 in self.hls:
            h = HiddenLayer(m1,m2,tf.nn.relu)
            self.layers.append(h)
            m1 = m2
        
        #final(output) layer 
        m2 = self.p
        h = HiddenLayer(m1,m2,None)
        self.layers.append(h)
        
        #collect all parameters
        self.all_params = []
        for h in self.layers:
            self.all_params += h.params
            
        #create placeholders for data
        tfX = tf.placeholder(tf.float32,shape = (None,self.m),name = 'X')
        tfY = tf.placeholder(tf.float32,shape = (None,self.p),name = 'Y')
        
        #logits from final layer
        Z_final = self.forward(tfX)
        
        #compute cost
        reg_cost = reg*sum([tf.nn.l2_loss(p) for p in self.all_params])
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits = Z_final,labels = tfY)
        ) + reg_cost
        
        optimizer = tf.train.RMSPropOptimizer(lr,decay = decay,momentum = mu).minimize(cost)
        
        predictions = self.predict(tfX)
        valid_predictions = self.predict(X_valid)
        
        n_iters = int(self.n/batch_sz)
        train_loss = []
        valid_err = []
        train_err = []
        #create session
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for epoch in range(epochs):
                for n_iter in range(n_iters):
                    X_batch = X[(n_iter*batch_sz):((n_iter*batch_sz)+batch_sz)]
                    Y_batch = Y[(n_iter*batch_sz):((n_iter*batch_sz)+batch_sz)]
                    
                    _,l,preds = session.run([optimizer,cost,predictions],feed_dict = {tfX : X_batch,tfY : Y_batch})
                    train_loss.append(l)
                    train_err.append(self.error_rate(np.argmax(Y_batch,1),preds))
                if epoch % print_period == 0:
                    v_preds = valid_predictions.eval()
                    #print v_preds
                    #print np.argmax(Y_valid,1)
                    v_err = self.error_rate(np.argmax(Y_valid,axis=1),v_preds)
                    valid_err.append(v_err)
                    print 'Epoch : %d,training loss : %.4f,validation error : %.4f' %(epoch,l,v_err)
                    print 'train err : %.4f' %train_err[-1]
        if show_fig:
            plt.figure(figsize = (12,8))
            plt.subplot(1,2,1)
            plt.plot(train_loss,label = 'Training loss')
            plt.legend()
            
            #plt.subplot(1,2,2)
            #plt.plot(valid_err,label = 'valid_err')
            #plt.legend()
        
    def forward(self,X):
        Z = X
        for h in self.layers:
            Z = h.forward(Z)
        return Z
    
    def predict(self,X):
        Y_hat = self.forward(X)
        return tf.argmax(Y_hat,1)
    
    def error_rate(self,labels,preds):
        return np.mean(labels != preds)