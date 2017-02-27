import numpy as np
import gym
from matplotlib import pyplot as plt
import sys
sys.path.append('..')
import tensorflow as tf
from util.exp_replay import Exp_Replay as Memory

ETA = 0.001
GAMMA = 0.95
MOMENTUM = 0
EPS_START = 0.05
EPS_END = 0.95
MEM_SIZE = 10000
BATCH = 10
MSIZE = 20
EP = 20
ITER = 50


class NaiveQN():
    def __init__(self,layer):
        self.layers = {}
        self.x_in = tf.placeholder(tf.float32, shape=[None, layer[0]], name = 'input')
        self.layers['layer1'] =  tf.nn.relu(tf.contrib.layers.fully_connected(
                                                inputs = self.x_in,
                                                num_outputs = layer[1]))
        for i in range(2,len(layer)- 1):
            self.layers['layer'+str(i)] = tf.nn.relu(tf.contrib.layers.fully_connected(
                                        inputs = self.layers['layer'+str(i -1)],
                                        num_outputs = layer[i]))

        self.out = tf.contrib.layers.fully_connected(
                                    inputs = self.layers['layer'+str(len(layer) -2)],
                                    num_outputs = layer[-1])

        self.label = tf.placeholder(tf.float32, shape = [None, layer[-1]])

        self.mseloss = tf.losses.mean_squared_error(labels = self.label,
                                                    predictions = self.out)


env = gym.make('CartPole-v0')
o = env.reset()
layer = [env.observation_space.shape[0],64,32, env.action_space.n]
memory = Memory(MEM_SIZE,[layer[0]])

model = NaiveQN(layer)
sess = tf.Session()
optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(model.mseloss)
sess.run(tf.global_variables_initializer())


rendering = False

episode = 0
dummy_i = 0
turns = 0
avrQ = []
avrq = 0

with sess.as_default():

    while episode < 20000:
        turns += 1
        dummy_i += 1

        if rendering:
            env.render()
        o = o.reshape(1,-1)
        out_s = model.out.eval(feed_dict = {model.x_in : o})
        action = np.argmax(out_s)
        if np.random.rand() > EPS_START + turns * EPS_END / 2000:
            action = env.action_space.sample()

        avrq += np.ravel(out_s)[action]

        o1,reward,done,_ = env.step(action) # take a random action
        o1 = o1.reshape(1,-1)
        memory.insert(o,action,reward,o1, False)
        o = o1

        if done or dummy_i == 200:
            memory.t[memory.pointer] = True
            memory.r[memory.pointer] = -200
            o = env.reset()
            episode += 1
            if dummy_i > 1000:
                rendering = True
            avrQ.append(avrq * 1.0 / dummy_i)
            print 'Episode :' + str(episode), avrQ[-1],dummy_i
            exist = 0
            dummy_i = 0

        if memory.length < 5 * BATCH*MSIZE:
            continue

        ind_ = np.random.permutation(memory.length)[:BATCH * MSIZE]

        for i in range(BATCH):
            ind = ind_[i * MSIZE: (i+1)* MSIZE ]

            out_s1 = model.out.eval(feed_dict = {model.x_in: memory.s1[ind]})

            # For non-terminal states, Q(s,a) = r(s,a) + gamma * max(Q(s',a')) (Empirical Mean)
            y = memory.r[ind] + GAMMA * np.max(out_s1,axis = 1) * (np.logical_not(memory.t[ind]))
            out_s = model.out.eval(feed_dict = {model.x_in: memory.s[ind]})
            out_s[np.arange(MSIZE), memory.a[ind]] = y

            optimizer.run(feed_dict = {model.x_in:memory.s[ind], model.label:out_s})
