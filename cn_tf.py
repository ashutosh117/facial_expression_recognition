#class based CNN with tensorflow

from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf

from util import init_filter,init_weights

#class to generate ConvPool layer
class ConvPoolLayer:
    def __init__(self,mi,mo,fw=5,fh=5,poolsz=(2,2)):
        filter_sz = (fw,fh,mi,mo)
        self.poolsz = poolsz
        w_init = init_filter(filter_sz,self.poolsz)
        self.w = tf.Variable(w_init)
        self.b = tf.Variable(np.zeros(mo,dtype = np.float32))
        self.params = [self.w,self.b]
        
    def forward(self,X):
        conv_out = tf.nn.conv2d(X,self.w,strides=[1,1,1,1],padding='SAME')
        conv_out = tf.nn.bias_add(conv_out,self.b)
        p1,p2 = self.poolsz
        pool_out = tf.nn.max_pool(
            conv_out,
            ksize = [1,p1,p2,1],
            strides = [1,p1,p2,1],
            padding = 'SAME'
        )
        return tf.tanh(pool_out)
    
#class to generate fully connected layers
class HiddenLayer:
    def __init__(self,m1,m2):
        self.m1 = m1
        self.m2 = m2
        w_init,b_init = init_weights(self.m1,self.m2)
        self.w = tf.Variable(w_init)
        self.b = tf.Variable(b_init)
        self.params = [self.w,self.b]
        
    def forward(self,X):
        return tf.nn.relu(tf.matmul(X,self.w) + self.b)
    

#class to generate CNN model
class CNN:
    def __init__(self,convpool_layer_sizes,hidden_layer_sizes):
        self.convpool_layer_sizes = convpool_layer_sizes
        self.hidden_layer_sizes = hidden_layer_sizes
        
    def fit(self,X,Y,batch_sz,epochs,lr = 0.01,reg = 0.1,mu=0.99,decay = 0.999,print_period = 1,show_fig = True):
        
        self.n,self.width,self.height,self.c = X.shape
        self.p = Y.shape[1]
        
        #list of convpool layers
        self.convpool_layers = []
        mi = self.c
        outw = self.width
        outh = self.height
        for mo,fw,fh in self.convpool_layer_sizes:
            layer = ConvPoolLayer(mi,mo,fw,fh)
            self.convpool_layers.append(layer)
            outw = (outw)//2
            outh = (outh)//2
            mi = mo
        
        #list of hidden layers
        self.hidden_layers = []
        m1 = self.convpool_layer_sizes[-1][0]*outh*outw
        for m2 in self.hidden_layer_sizes:
            layer = HiddenLayer(m1,m2)
            self.hidden_layers.append(layer)
            m1 = m2
        
        #logistic regression layer
        m2 = self.p
        w_init,b_init = init_weights(m1,m2)
        self.W = tf.Variable(w_init)
        self.b = tf.Variable(b_init)
        
        #list of all parameters 
        self.all_params = [self.W,self.b]
        for layer in self.convpool_layers:
            self.all_params.extend(layer.params)
        for layer in self.hidden_layers:
            self.all_params += layer.params
        
        #placeholders
        tfX = tf.placeholder(dtype = tf.float32,shape = (None,self.width,self.height,self.c),name = 'X')
        tfY = tf.placeholder(tf.float32,shape = (None,self.p),name = 'Y')
        
        #forward propagation
        Y_hat = self.forward(tfX)
        
        #compute cost
        reg_cost = reg*sum([tf.nn.l2_loss(p) for p in self.all_params])
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Y_hat,labels = tfY)) + reg_cost
        
        #define optimizer
        optimizer = tf.train.RMSPropOptimizer(lr,decay=decay,momentum=mu).minimize(cost)
        
        #compute predictions
        predictions = self.predict(tfX)
        
        #X,Y = shuffle(X,Y)
        X_train,X_valid = X[:-1000,:,:,:],X[-1000:,:,:,:]
        Y_train,Y_valid = Y[:-1000,:],Y[-1000:,:]
        
        train_loss = []
        valid_errs = []
        
        #create a session
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            #training loop
            n_iters = (self.n-1000)//batch_sz
            for epoch in range(epochs):
                for n_iter in range(n_iters):
                    X_batch = X_train[(batch_sz*n_iter):((batch_sz*n_iter)+batch_sz)]
                    Y_batch = Y_train[(batch_sz*n_iter):((batch_sz*n_iter)+batch_sz)]

                    session.run(optimizer,feed_dict = {tfX:X_batch,tfY:Y_batch})
                    
                    if n_iter%print_period == 0:
                        l = session.run(cost,feed_dict = {tfX:X_batch,tfY:Y_batch})
                        train_loss.append(l)
                        valid_preds = session.run(predictions,feed_dict={tfX:X_valid})
                        valid_err = self.compute_error(np.argmax(Y_valid,axis=1),valid_preds)
                        valid_errs.append(valid_err)
                        print('\n')
                        print('Epoch : %d,iter : %d,Training loss : %.4f,Valid err : %.4f' %(epoch,n_iter,l,valid_err))
                    
            if show_fig:
                plt.plot(train_loss,label = 'Training loss')
                plt.legend()
                
        
        
        
    def forward(self,X):
        conv_out = X
        for layer in self.convpool_layers:
            conv_out = layer.forward(conv_out)
        out_shape = conv_out.get_shape().as_list()
        hidden_out = tf.reshape(conv_out,[-1,np.prod(out_shape[1:])])
        for layer in self.hidden_layers:
            hidden_out = layer.forward(hidden_out)
        #final logistic regression layer
        Y_hat = tf.matmul(hidden_out,self.W) + self.b
        return Y_hat
        
        
    def predict(self,X):
        Y_hat = self.forward(X)
        return tf.argmax(Y_hat,1)
        
    def compute_error(self,labels,preds):
        return (labels != preds).mean()
        