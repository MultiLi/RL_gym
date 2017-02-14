import numpy as np
import gym
from matplotlib import pyplot as plt
import sys
sys.path.append('..')
from util.multilayer import Multilayer
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


nn = Multilayer([[4,'in'],[64,'tanh'],[32,'tanh'],[2,'linear']],
                eta = ETA, moment = MOMENTUM, reg = 0)

exp_replay = Memory(MEM_SIZE)

pl = []
env = gym.make('CartPole-v0')
rendering = False
iter = 0
anum = 0
while iter<ITER:
    # Game data collection

    if anum / EP > 200 or rendering:
        rendering = True
        env.render()
    anum = 0
    num = 0
    obs = env.reset()
    ep = 0
    while ep < EP:

        nn.predict(obs.reshape(1,-1))
        q = nn.o[nn.depth][0]
        action = np.argmax(q)
        if np.random.rand() > EPS_START + (EPS_END - EPS_START) * (iter * EP + ep)/ ITER / EP:
            action = env.action_space.sample()
        obs1,r,done,info = env.step(action)
        num += 1
        exp_replay.insert(obs,action,r, obs1)
        obs = obs1

        if done:
            obs = env.reset()
            exp_replay.r[exp_replay.pointer] = -1
            pl.append(num)
            num = 0
            ep += 1
            anum += num

        if exp_replay.length <  10 * BATCH * MSIZE :
            continue

        '''
        mini- batch version
        '''
        ind_ = np.random.permutation(exp_replay.length)[:BATCH * MSIZE]
        for i in range(BATCH):
            ind = ind_[i * MSIZE: (i+1)* MSIZE ]

            nn.predict(exp_replay.s1[ind])
            y = exp_replay.r[ind] + GAMMA * np.max(nn.o[nn.depth],axis = 1) * (exp_replay.r[ind]> 0)

            nn.predict(exp_replay.s[ind])
            t = np.copy(nn.o[nn.depth])
            t[np.arange(MSIZE), exp_replay.a[ind]] = y

            nn.backprop(t)

        nn.update()

    iter += 1
    print iter


plt.figure()
plt.plot(np.array(pl))
plt.title('Duration')
plt.show()
