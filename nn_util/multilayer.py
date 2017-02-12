import numpy as np
import activation as act

class Multilayer():
    '''
    About some bookkeeping variables of the network
        Notation:
            n: number of input data samples, the row number of input matrix
            m: number of dimension of corresponding data, different for different layers
            k: number of dimension of next layer
        self.o[i]       n x m   [1,depth]        output of the layer i (after non-linear mapping)
        self.weight[i]  m x k   [1,depth - 1]    weight between layer i and i + 1
        self.dw[i]                               gradient of w
        self.bias[i]    1 x k                    bias
        self.db[i]                               gradient of bias
        self.delta[i]   n x m   [1,depth]        gradient of output
    '''

    def __init__(self,layer,eta = 0.1, epoch = 10, moment = 0.9, T = 100,reg = 0.01,batch_num = 1000):
        self.layer = layer
        self.depth = len(layer)
        self.weight = {}
        self.b = {}
        self.o = {}
        self.delta = {}
        self.dw = {}
        self.db = {}
        i = 1
        while i < self.depth:
            self.weight[i] = np.random.randn(layer[i-1][0],layer[i][0]) / np.sqrt(layer[i-1][0])
            self.dw[i] = np.zeros_like(self.weight[i])
            self.b[i] = np.zeros((1,layer[i][0]))
            self.db[i] = np.zeros_like(self.b[i])
            i += 1
        self.eta = eta
        self.moment = moment
        self.T = T
        self.reg = reg
        self.epoch = epoch
        self.eta_n = eta
        self.batch_num = batch_num


    def predict(self, X):
        '''
        Calculate the prediction score of input samples
        '''
        i = 1
        self.o[i] = X
        while i < self.depth - 1:
            self.o[i+1] = getattr(act,self.layer[i][1])(np.dot(self.o[i], self.weight[i]) + self.b[i])
            i += 1

        self.o[i+1] = np.dot(self.o[i],self.weight[i])
        # self.o[i+1] = np.exp(np.dot(self.o[i],self.weight[i]) + self.b[i])
        # self.o[i+1] /= np.sum(self.o[i+1],axis = 1).reshape((-1,1))

    def backprop(self, Y):
        '''
        Get gradient of each set of parameters in the network
        In each iteration, current layer has k neurons, and the previous one has m.
        '''

        self.delta[self.depth] = (self.o[self.depth] - Y) / Y.shape[0]
        self.eta_n *= 0.999
        i = self.depth - 1
        while i > 0:
            self.dw[i] = self.moment * self.dw[i] - self.eta_n * (np.dot(self.o[i].T, self.delta[i+1]) +  self.reg * self.weight[i])
            self.db[i] = self.moment * self.db[i] - self.eta_n * np.sum(self.delta[i+1], axis = 0).reshape(1,-1)
            if i != 1:
                self.delta[i] = np.dot(self.delta[i+1], self.weight[i].T) * getattr(act,'d'+self.layer[i-1][1])(self.o[i])
            i -= 1
        return

    def update(self):
        i = self.depth - 1
        while i > 0:
            self.weight[i] += self.dw[i]
            self.b[i] += self.db[i]
            i -= 1
        return

    def training(self,X, Y):
        epoch = 0
        losstrain = []
        lossv = {}
        lossv['train'] = []
        lossv['val'] = []
        lossv['test'] = []
        accuv = {}
        accuv['train'] = []
        accuv['val'] = []
        accuv['test'] = []

        self.predict(X['test'])
        accuv['test'].append(self.accu(Y['test']))
        lossv['test'].append(self.loss(Y['test']))
        self.predict(X['val'])
        accuv['val'].append(self.accu(Y['val']))
        lossv['val'].append(self.loss(Y['val']))
        self.predict(X['train'])
        accuv['train'].append(self.accu(Y['train']))
        lossv['train'].append(self.loss(Y['train']))
        print 'Original loss is ' + str(lossv['train'][-1])

        while epoch < self.epoch:
            randperm = np.random.permutation(X['train'].shape[0])
            batch_size = int(X['train'].shape[0] / self.batch_num)
            for i in range(self.batch_num):
                self.predict(X['train'][randperm[i * batch_size : (i + 1) * batch_size]])
                losstrain.append(self.loss(Y['train'][randperm[i * batch_size : (i + 1) * batch_size]]))
                self.eta_n = self.eta / ( 1+ epoch * 1.0 / self.T)
                self.backprop(Y['train'][randperm[i * batch_size : (i + 1) * batch_size]])
                d = self.depth - 1
                while d > 0:
                    self.weight[d] += self.dw[d]
                    self.b[d] += self.db[d]
                    d -= 1
            epoch += 1
            self.predict(X['test'])
            accuv['test'].append(self.accu(Y['test']))
            lossv['test'].append(self.loss(Y['test']))
            self.predict(X['val'])
            accuv['val'].append(self.accu(Y['val']))
            lossv['val'].append(self.loss(Y['val']))
            self.predict(X['train'])
            accuv['train'].append(self.accu(Y['train']))
            lossv['train'].append(self.loss(Y['train']))
            print 'Loss of epoch ' + str(epoch) + ' is ' + str(lossv['train'][-1])

        return lossv, accuv,losstrain

    def loss(self, Y):
        '''
        Return loss function value of current prediction, in this problem 'softmax' is used
        '''
        loss = -np.sum(np.log(self.o[self.depth]) * Y) / Y.shape[0]
        i = 1
        while i < self.depth:
            loss += self.reg *  0.5 * np.sum(self.weight[i] ** 2)
            i += 1
        return loss

    def accu(self, Y):
        '''
        Return the classification accuracy of current predictions
        '''
        return 1.0 * np.sum(Y[np.arange(Y.shape[0]),np.argmax(self.o[self.depth],axis = 1)] == 1)/Y.shape[0]
