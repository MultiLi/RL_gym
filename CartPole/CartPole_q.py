import numpy as np
import gym
from matplotlib import pyplot as plt

ETA = 0.001
GAMMA = 0.9
MOMENTUM = 0
EPS_START = 0.05
EPS_END = 0.95
MEM_SIZE = 5000
MSIZE = 30

def sigmoid(x):
    return 1.0 / ( 1+ np.exp(-x))

def forward(x,w1,w2,b1,b2):
    h = sigmoid(np.dot(x, w1) + b1) # Hidden Layer
    y = np.dot(h, w2) + b2          # Output Layer
    return y,h

def backward(t, y, h, x, w2,b2):
    diff = (y - t) /y.shape[0]
    dw2 = np.dot(h.T, diff)
    db2 = np.sum(diff, axis = 0)#.reshape((1,-1))
    dh = np.dot(diff, w2.T) * h * (1-h)
    dw1 = np.dot(x.T, dh)
    db1 = np.sum(dh, axis = 0)#.reshape((1,-1))
    return dw1, dw2, db1, db2

w1 = np.random.randn(4,6) / np.sqrt(4)
w2 = np.random.randn(6,2) / np.sqrt(6)
b1,b2 = np.zeros([1,6]),np.zeros([1,2])
dw1,dw2,db1,db2 = np.zeros_like(w1),np.zeros_like(w2),np.zeros_like(b1),np.zeros_like(b2)


initilized = False
pl = []
env = gym.make('CartPole-v0')
rendering = False
iter = 0
while iter<100:
    # Game data collection
    num = 0
    obs = env.reset()

    s_, a_, r_, s1_ = [],[],[],[]
    ep = 0
    while ep < 5:
        if num > 500 or rendering:
            rendering = True
            env.render()
        s_.append(obs)
        q,_ = forward(np.array(obs).reshape((1,-1)),w1,w2,b1,b2)
        action = np.argmax(q[0]) if np.random.rand() < EPS_START + (EPS_END - EPS_START) * iter/ 1000  else env.action_space.sample()
        obs1,r,done,info = env.step(action)
        a_.append(action)
        if done:
            obs = env.reset()
            r_.append(-200)
            s1_.append([0,0,0,0])
            pl.append(num)
            num = 0
            ep += 1
            continue
        r_.append(r)
        s1_.append(obs1)
        obs = obs1
        num += 1

    # print r_

    if not initilized:
        mem_xs, mem_ac, mem_re, mem_xs1 = np.vstack(s_), np.array(a_), np.array(r_), np.vstack(s1_)
        initilized = True
    else:
        start_ind = np.max([mem_xs.shape[0] + len(s_) - MEM_SIZE, 0])
        # print 1
        mem_xs = np.vstack((mem_xs[start_ind:], s_))
        # print mem_xs
        mem_xs1 = np.vstack((mem_xs1[start_ind:], s1_))
        mem_re = np.hstack((mem_re[start_ind:], r_))
        mem_ac = np.hstack((mem_ac[start_ind:], a_))

    if mem_xs.shape[0] < MSIZE * 20:
        continue
    # print mem_ac

    bind = np.random.permutation(mem_xs.shape[0])

    for i in range(mem_xs.shape[0] / MSIZE):
        ind = bind[i * MSIZE: (i + 1) * MSIZE]
        # ind = np.random.permutation(mem_xs.shape[0])[:MSIZE]
        # print mem_xs[ind]
        yy, h  = forward(mem_xs[ind],w1,w2,b1,b2)
        t = np.copy(yy)
        tmask = ( np.sum(mem_xs1[ind],axis = 1) != 0 )

        y1, _ = forward(mem_xs1[ind],w1,w2,b1,b2)

        # print mem_re[ind]

        t[np.arange(MSIZE),mem_ac[ind]] += mem_re[ind] + GAMMA * np.max(y1 ,axis = 1) * tmask
        # print t- yy
        # print np.sqrt(np.sum(yy**2))
        print np.mean((t-yy)**2)
        xw1,xw2,xb1,xb2 = backward(t,yy,h,mem_xs[ind],w2,b2)
        dw1 = MOMENTUM * dw1 + ETA / (1 + ((iter * 20 + i ) / 10)) * xw1
        dw2 = MOMENTUM * dw2 + ETA / (1 + ((iter * 20 + i ) / 10)) * xw2
        db1 = MOMENTUM * db1 + ETA / (1 + ((iter * 20 + i ) / 10)) * xb1
        db2 = MOMENTUM * db2 + ETA / (1 + ((iter * 20 + i ) / 10)) * xb2
        w1 -=  dw1
        w2 -=  dw2
        b1 -=  db1
        b2 -=  db2
        # yyy,_ = forward(xs[ind],w1,w2,b1,b2)
        # print np.mean((t-yyy)**2)

    # break
    iter += 1
    # print iter

plt.plot(np.array(pl))
plt.show()
