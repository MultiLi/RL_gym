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
memory = Exp_Replay(MEM_SIZE,INPUT_DIM, env.action_space.n)
o = env.reset()

sess = tf.Session()
optimizer = tf.train.RMSPropOptimizer(1e-4).minimize(model.mseloss)
sess.run(tf.global_variables_initializer())


i = 0
episode = 0
with sess.as_default():

    while episode < 10:

        s = imresize(np.dot(o[...,:3], [0.299, 0.587, 0.114]),INPUT_DIM,interp = 'lanczos')
        s = s.reshape((1,) + s.shape+(1,))
        action = np.argmax(model.fc_2_o.eval(feed_dict = {model.x_in : s}))
        if np.random.rand() < EPS_START + i * EPS_END / 500:
            action = env.action_space.sample()

        o1,reward,done,_ = env.step(action) # take a random action
        s1 = imresize(np.dot(o1[...,:3], [0.299, 0.587, 0.114]),INPUT_DIM,interp = 'lanczos')
        s1 = s1.reshape(s1.shape+(1,))
        # print s1.shape
        memory.insert(s,action,reward,s1)

        o = o1
        if done:
            o = env.reset()
            episode += 1

        if memory.length < 5 * BATCH*MSIZE:
            continue

        ind_ = np.random.permutation(memory.length)[:BATCH * MSIZE]

        for i in range(BATCH):
            ind = ind_[i * MSIZE: (i+1)* MSIZE ]
            ind = np.array(filter(lambda x: memory.r[x] == 0, ind))
            tMSIZE = ind.size

            out_s1 = model.fc_2_o.eval(feed_dict = {model.x_in: memory.s1[ind]})
            y = memory.r[ind] + GAMMA * np.max(out_s1,axis = 1) * (memory.r[ind] == 0)

            out_s = model.fc_2_o.eval(feed_dict = {model.x_in: memory.s[ind]})
            # print memory.a[ind]
            out_s[np.arange(tMSIZE), memory.a[ind]] = y

            optimizer.run(feed_dict = {model.x_in:memory.s[ind], model.y:out_s})







#     env.render()
#
# o = env.reset()
