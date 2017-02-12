import numpy as np
import gym
from matplotlib import pyplot as plt
import sys
sys.path.append('..')
from nn_util.multilayer import Multilayer

ETA = 0.01
GAMMA = 0.98
MOMENTUM = 0.95
EPS_START = 0.05
EPS_END = 0.95
MEM_SIZE = 5000
BATCH = 10
MSIZE = 30


class Memory():
    def __init__(self, num):
        self.pointer = -1
        self.size = num
        self.length = 0
        self.s = np.zeros((num,4))
        self.a = np.zeros(num,dtype = int)
        self.r = np.zeros(num)
        self.s1 = np.zeros((num,4))

    def insert(self,s,a,r,ns):
        self.pointer += 1
        if self.pointer == self.size:
            self.pointer = 0
        self.s[self.pointer] = s
        self.a[self.pointer] = a
        self.r[self.pointer] = r
        self.s1[self.pointer] = ns
        if self.length < self.size:
            self.length += 1


nn = Multilayer([[4,'in'],[64,'tanh'],[32,'tanh'],[2,'linear']],
                eta = ETA, moment = MOMENTUM, reg = 0)

exp_replay = Memory(MEM_SIZE)

pl = []
env = gym.make('CartPole-v0')
rendering = False
iter = 0
while iter<50:
    # Game data collection
    num = 0
    obs = env.reset()
    ep = 0
    while ep < 30:
        if num > 1000 or rendering:
            rendering = True
            env.render()
        nn.predict(obs.reshape(1,-1))
        q = nn.o[nn.depth][0]
        action = np.argmax(q) if np.random.rand() < EPS_START + (EPS_END - EPS_START) * (iter * 30 + ep)/ 1000  else env.action_space.sample()
        obs1,r,done,info = env.step(action)
        num += 1
        exp_replay.insert(obs,action,r, obs1)
        obs = obs1

        if done:
            obs = env.reset()
            exp_replay.r[exp_replay.pointer] = -10
            pl.append(num)
            num = 0
            ep += 1

        if exp_replay.length <  BATCH * MSIZE :
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
