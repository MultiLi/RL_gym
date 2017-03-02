import gym
from gym import wrappers
import numpy as np

GAMMA = 0.95
LRATE = 0.01
MOMENTUM = 0.95
BATCH_SIZE = 5


def sigmoid(x):
    return 1.0 / ( 1+ np.exp(-x))

def forward(x,w1,w2,b1,b2):
    h = np.dot(x, w1) + b1                  # Hidden Layer
    h[h < 0] = 0                            # Relu Activation
    y = sigmoid(np.dot(h, w2) + b2)         # Output Layer
    return y,h

def backward(t, y, h, x, w2,b2, r):
    diff = (1 - y - t) * r /y.shape[0]      # REINFORCE gradient
    dw2 = np.dot(h.T, diff)
    db2 = np.sum(diff, axis = 0)
    dh = np.dot(diff, w2.T)
    dw1 = np.dot(x.T, dh)
    db1 = np.sum(dh, axis = 0)
    return dw1, dw2, db1, db2

def accumulated_reward(r):
    accu = 0
    accu_reward = np.zeros((len(r),1))
    i = len(r) - 1
    while i >= 0:
        accu = GAMMA * accu + r[i]
        accu_reward[i] = accu
        i -= 1
    accu_reward -= np.mean(accu_reward)
    accu_reward /= np.std(accu_reward)
    return accu_reward

# Parameters Initialization
w1 = np.random.randn(4,8) / np.sqrt(4)
w2 = np.random.randn(8,1) / np.sqrt(8)
b1 = np.zeros([1,8])
b2 = 0.0
dw1,dw2,db1,db2 = np.zeros_like(w1),np.zeros_like(w2),np.zeros_like(b1),0


xin,reward,out,hidden, label, avrreward = [],[],[],[],[],[]

rendering = False

env = gym.make('CartPole-v0')
o = env.reset()
rsum = 0

iter = 1
while iter < 1000:
    xin.append(o.reshape(1,-1))
    if rendering:
        env.render()
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

        # Parameter update
        for i in range(10):
            xw1,xw2,xb1,xb2 = backward(T,Y,H,X,w2,b2,R)
            dw1 = MOMENTUM * dw1 + LRATE / (1 + iter / 1000) * xw1
            dw2 = MOMENTUM * dw2 + LRATE / (1 + iter / 1000) * xw2
            db1 = MOMENTUM * db1 + LRATE / (1 + iter / 1000) * xb1
            db2 = MOMENTUM * db2 + LRATE / (1 + iter / 1000) * xb2
            w1 +=  dw1
            w2 +=  dw2
            b1 +=  db1
            b2 +=  db2
            forward(X,w1,w2,b1,b2)

        iter += 1

        if iter % BATCH_SIZE == 0:
            avrreward.append(rsum / BATCH_SIZE)
            if avrreward[-1] > 5000:
                rendering = True
            rsum = 0
            print iter
        o = env.reset()

plt.figure()
plt.plot(np.array(avrreward))
plt.show()
