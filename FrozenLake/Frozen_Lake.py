import numpy as np
import gym
from matplotlib import pyplot as plt

env = gym.make('FrozenLake-v0')
REWARD_DECAY = 0.95
ep_Start = 0.05
ep_End = 0.95
LRATE = 0.8
Qtable = np.zeros((16,4))
Rs = []
Avr = []
for iter in range(10000):
    s = env.reset()
    i = 0
    rr = 0
    while i < 100:
        a = env.action_space.sample() if np.random.rand() > ep_Start + ep_End * iter / 1000 else np.argmax(Qtable[s])
        s1,r,d,_ = env.step(a)
        Qtable[s,a] = Qtable[s,a] + LRATE / (1 + iter / 100)* (r + REWARD_DECAY * np.max(Qtable[s1]) - Qtable[s,a])
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
