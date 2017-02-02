# Another way to introduce neural network with TD learning into easy problem.

import numpy as np
import gym
from matplotlib import pyplot as plt

env = gym.make('FrozenLake-v0')
REWARD_DECAY = 0.99
ep_Start = 0.05
ep_End = 0.95
LRATE = 0.1
Qnet = np.zeros((16,4))

Rs = []
Avr = []
for iter in range(10000):
    s = env.reset()
    s_vec = np.zeros((1,16))
    s_vec[0,s] = 1
    i = 0
    rr = 0
    while i < 100:
        Q_out = s_vec.dot(Qnet)
        a = env.action_space.sample() if np.random.rand() > ep_Start + ep_End * iter / 1000 else np.argmax(Qout)
        s1,r,d,_ = env.step(a)
        s_vec = np.zeros((1,16))
        s_vec[0,s1] = 1
        Qnet[s,a] +=  LRATE / (1 + iter / 10)* (r + REWARD_DECAY * np.max(s_vec.dot(Qnet)) - Qout[0,a])
        s = s1
        i += 1
        rr+= r
        if d:
            break
    Rs.append(rr)
    if iter % 1000 == 0:
        Avr.append(np.mean(np.array(Rs[-1000:])))
# print Qtable
plt.plot(np.array(Rs))
plt.figure()
plt.plot(np.array(Avr))
plt.show()
