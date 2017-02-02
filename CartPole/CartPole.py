import gym
import numpy as np
from matplotlib import pyplot as plt

REWARD_DECAY = 0.98
LRATE = 0.05
BATCH_SIZE = 20


def sigmoid(x):
    return 1.0 / ( 1+ np.exp(-x))

def forward(x,w1,w2,b1,b2):
    h = np.dot(x, w1) + b1           # Hidden Layer
    h[h < 0] = 0
    y = sigmoid(np.dot(h, w2) + b2)            # Output Layer
    return y,h

def backward(t, y, h, x, w2,b2, r):
    diff = (1 - y - t) * r /y.shape[0]
    dw2 = np.dot(h.T, diff)                     # (n x 4)' * (n x 2) -> ( 4 x 2)
    db2 = np.sum(diff, axis = 0)
    dh = np.dot(diff, w2.T)        # (n x 2) * ( 2 x 4) -> ( n x 4)
    dw1 = np.dot(x.T, dh)
    db1 = np.sum(dh, axis = 0)
    return dw1, dw2, db1, db2

def accumulated_reward(r):
    accu = 0
    accu_reward = np.zeros((len(r),1))
    i = len(r) - 1
    while i >= 0:
        accu = REWARD_DECAY * accu + r[i]
        accu_reward[i] = accu
        i -= 1
    accu_reward -= np.mean(accu_reward)
    accu_reward /= np.std(accu_reward)
    return accu_reward


w1 = np.random.randn(4,8) / np.sqrt(4)
w2 = np.random.randn(8,1) / np.sqrt(8)
b1 = np.zeros([1,8])
b2 = 0.0

dw1,dw2,db1,db2 = np.zeros_like(w1),np.zeros_like(w2),np.zeros_like(b1),0

env = gym.make('CartPole-v0')
xin,reward,out,hidden, label = [],[],[],[],[]
iter = 1
rendering = False
rsum = 0
o = env.reset()

avrreward = []
while iter < 10000:
    xin.append(o.reshape(1,-1))
    # if rsum / BATCH_SIZE > 500 or rendering:
    #     rendering = True
    #     env.render()
    y,h = forward(xin[-1],w1,w2,b1,b2)
    hidden.append(h)
    action = 1 if np.random.rand() < y else 0
    out.append(y)
    label.append( 1- action )
    o, rr, done, info = env.step(action)
    rsum += rr
    reward.append(rr)

    if done:
        H = np.vstack(hidden)
        T = np.vstack(label)
        Y = np.vstack(out)
        X = np.vstack(xin)
        R = accumulated_reward(reward)
        xin,reward, obs,out,hidden, label = [],[],[],[],[],[]

        xw1,xw2,xb1,xb2 = backward(T,Y,H,X,w2,b2,R)
        dw1 += LRATE / (1 + iter / 1000) * xw1
        dw2 += LRATE / (1 + iter / 1000) * xw2
        db1 += LRATE / (1 + iter / 1000) * xb1
        db2 += LRATE / (1 + iter / 1000) * xb2
        iter += 1
        if iter % BATCH_SIZE == 0:
            w1 +=  dw1
            w2 +=  dw2
            b1 +=  db1
            b2 +=  db2
            dw1,dw2,b1,b2 = np.zeros_like(w1),np.zeros_like(w2),np.zeros_like(b1),0
            avrreward.append(rsum / BATCH_SIZE)
            rsum = 0
            # if iter == 5000:
            #     print rsum / BATCH_SIZE
            # print iter

        o = env.reset()
        # print iter

plt.plot(np.array(avrreward))
plt.show()
