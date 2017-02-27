import tensorflow as tf
import numpy as np
import gym
import sys
sys.path.append('..')
from util.exp_replay import Exp_Replay
from util.DQN import DQN
from scipy.misc import imresize

GAMMA = 0.99
MEM_SIZE = 5000
BATCH = 10
MSIZE = 20
INPUT_DIM = [120,80]
EPS_START = 0.05
EPS_END = 0.95

env = gym.make('DoubleDunk-v0')
model = DQN(INPUT_DIM, env.action_space.n)
memory = Exp_Replay(MEM_SIZE,INPUT_DIM)
o = env.reset()

sess = tf.Session()
optimizer = tf.train.RMSPropOptimizer(1e-4).minimize(model.mseloss)
sess.run(tf.global_variables_initializer())



episode = 0
dummy_i = 0
turns = 0
avrQ = []
avrq = 0
with sess.as_default():

    while episode < 20:
        turns += 1
        dummy_i += 1
        s = imresize(np.dot(o[...,:3], [0.299, 0.587, 0.114]),INPUT_DIM,interp = 'lanczos')
        s = s.reshape((1,) + s.shape+ (1,))

        out_s = model.fc_2_o.eval(feed_dict = {model.x_in : s})
        action = np.argmax(out_s)
        if np.random.rand() > EPS_START + turns * EPS_END / 500:
            action = env.action_space.sample()

        avrq += np.ravel(out_s)[action]

        o1,reward,done,_ = env.step(action) # take a random action

        s1 = imresize(np.dot(o1[...,:3], [0.299, 0.587, 0.114]),INPUT_DIM,interp = 'lanczos')
        s1 = s1.reshape((1,) + s1.shape + (1,))

        memory.insert(s,action,reward,s1, False)

        o = o1

        if done:
            memory.t[memory.pointer] = True
            o = env.reset()
            episode += 1
            avrQ.append(avrq * 1.0 / dummy_i)
            print 'Episode :' + str(episode), avrQ[-1]
            dummy_i = 0

        if memory.length < 5 * BATCH*MSIZE:
            continue

        ind_ = np.random.permutation(memory.length)[:BATCH * MSIZE]

        for i in range(BATCH):
            ind = ind_[i * MSIZE: (i+1)* MSIZE ]

            out_s1 = model.fc_2_o.eval(feed_dict = {model.x_in: memory.s1[ind]})

            # For non-terminal states, Q(s,a) = r(s,a) + gamma * max(Q(s',a')) (Empirical Mean)
            y = memory.r[ind] + GAMMA * np.max(out_s1,axis = 1) * (np.logical_not(memory.t[ind]))
            out_s = model.fc_2_o.eval(feed_dict = {model.x_in: memory.s[ind]})
            out_s[np.arange(MSIZE), memory.a[ind]] = y

            optimizer.run(feed_dict = {model.x_in:memory.s[ind], model.y:out_s})
