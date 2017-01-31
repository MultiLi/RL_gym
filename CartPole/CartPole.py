import gym
import numpy as np
from matplotlib import pyplot as plt

REWARD_DECAY = 0.98
BATCH_NUM = 50

def sigmoid(x):
    return 1.0 / ( 1+ np.exp(-x))

def forward(w1,w2,observation):
    hidden_out = np.dot(observation, w1)            # Hidden Layer
    hidden_sig = sigmoid(hidden_out)                # Nonlinear Map
    final_out = np.dot(hidden_sig, w2)             # Output Layer
    return final_out, hidden_sig

def backward(y, out, mask, h_sig, state, w2):
    diff = out - y
    diff[np.arange(y.shape[0]),1 - mask] = 0
    dw2 = np.dot(h_sig.T, diff)          # (n x 4)' * (n x 2) -> ( 4 x 2)
    dh = np.dot(diff, w2.T)                 # (n x 2) * ( 2 x 4) -> ( n x 4)
    dh *= h_sig * (1- h_sig)
    dw1 = np.dot(state.T, dh)
    return dw1, dw2

w1 = np.random.randn(5,4) / np.sqrt(9)
w2 = np.random.randn(4,2) / np.sqrt(6)

env = gym.make('CartPole-v0')
env.reset()
value, state = [],[]
end = 100
count = 0
actions = []
while True:
    observation = env.reset()
    t = 0
    print count
    while True:

        env.render()
        aug_obs = np.append(1,observation)      # Augment state s with a constant bias
        state.append(aug_obs)                   # Record state s
        out,_ = forward(w1,w2,aug_obs)          # Get Q(s,a) for each action a fitted by an nn

        if value:                               # Q(s,a) = r(s') + gamma max (Q(s',a'))
            value[-1][actions[-1]] = REWARD_DECAY * np.max(out) + reward

        value.append(out)

        action = np.argmax(out) if np.random.rand() > 0.05 + 0.9 * np.exp(-1.0 *count/ 50) else env.action_space.sample()
        actions.append(action)
        observation, reward, done, info = env.step(action)
        t += 1

        if done:
            print("Episode finished after {} timesteps".format(t))
            value[-1][actions[-1]] = 0

            # Store the obtained training data
            X = np.vstack(state)
            Y = np.vstack(value)
            A = np.array(actions)
            # print

            state,value, actions = [], [], []

            # Train the Q-network
            epoch = 0
            loss = []
            while epoch < 100:
                O, H = forward(w1,w2,X)
                # Q(s,a) = R(s')+ gamma * max(Q(s',a'))
                Y[Y<0] = np.random.rand(Y.shape[0],Y.shape[1])[Y < 0]
                Y[np.arange(Y.shape[0] - 1), A[:-1]] = 1 + REWARD_DECAY * np.max(Y[1:], axis = 1)
                #print Y
                loss.append( np.sum((Y[A] - O[A])**2)/ 2 / Y.shape[0])
                dw1, dw2 = backward(Y, O, A, H, X, w2)
                w1 -= 0.1 / (1 + epoch / 50) * dw1
                w2 -= 0.1 / (1 + epoch / 50) * dw2
                epoch += 1
            break

    count += 1


# if loss:
    # print loss
# epoch = 0
# loss = []
# while epoch < 100:
#     mini_epoch = 0
#     perm = np.random.permutation(n)
#     loss.append(0)
#         scope = perm[mini_epoch * BATCH_NUM :np.min([(mini_epoch+1) * BATCH_NUM, n])]
#         x = X[scope]
#         y = Y[scope]
#
#         action = np.argmax(o, axis = 1) if np.random.rand() > 0.05 + 0.9 * np.exp(-1.0 *epoch/ 50) else np.random.randint(2, size=y.shape[0])
#         loss[-1] += 0.5 * np.sum((y - o)[np.arange(y.shape[0]),action] ** 2)/y.shape[0]
#         dw1, dw2 = backward(y, o, action, h, x, w2)
#         w1 -= 0.1 / (1 + epoch / 50) * dw1
#         w2 -= 0.1 / (1 + epoch / 50) * dw2
#
#         pind = np.zeros(n).astype(dtype = np.bool_)
#         pind[scope[y[np.arange(scope.shape[0]),action] != 0]] = True
#         ind = np.roll(pind , 1)
#         y[y[np.arange(y.shape[0]),action] != 0, actions[pind]] = 1 + REWARD_DECAY *np.max(Y[ind],axis = 1)
#
#         mini_epoch += 1
#     loss[-1] /= (n / BATCH_NUM)
#     epoch += 1
#
# test ,_ = forward(w1, w2, X)
# print test[:30]
# while True:
#     observation = env.reset()
#     t = 0
#     while True:
#         env.render()
#         aug_obs = np.append(1,observation)                      # Augemented Environment Vector
#         #state.append(aug_obs)
#         out,_  = forward(w1,w2,aug_obs)
#
#
#         value.append(out)
#         observation, reward, done, info = env.step(np.argmax(out))
#
#         t += 1
#         if done:
#             print("Episode finished after {} timesteps".format(t))
#             break
