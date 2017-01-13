import gym
import numpy as np

def policy_forward(w1,w2,observation):
    hidden_out = np.dot(observation, w1)            # Hidden Layer
    hidden_relu = hidden_out * (hidden_out > 0)     # Nonlinear Map
    final_out = np.dot(hidden_relu, w2)             # Output Layer
    actions = 1.0 / ( 1.0 + np.exp(-final_out))     # Sigmoid
    return actions,hidden_relu

def policy_backward(scores, h, state,w2):
    dw2 = np.dot(h.T, scores)
    dh = np.outer(scores, w2)
    dh[h<0] = 0
    dw1 = np.dot(state.T, dh)
    return dw1, dw2

w1 = np.random.randn(5,4) / np.sqrt(9)
w2 = np.random.randn(4) / 2

env = gym.make('CartPole-v0')
env.reset()
scores, hidden, state = [],[],[]
end = 20
count = 0
for _ in range(1000):
    observation = env.reset()
    t = 0
    while t < end:
        env.render()
        aug_obs = np.append(1,observation)                      # Augemented Environment Vector
        actions,h = policy_forward(w1,w2,aug_obs)
        state.append(aug_obs)
        action = 1 if np.random.rand() < actions else 0
        score = action - actions
        hidden.append(h)
        scores.append(score)
        observation, reward, done, info = env.step(action)
        t += 1
        if done:
            print("Episode finished after {} timesteps".format(t))
            reward = 0
            break
    if t == end:
        count += 1
        reward = 1.0
    if count == 10:
        end += 10
        count = 0

    s = np.array(scores)
    h = np.vstack(hidden)
    obvs = np.vstack(state)
    discounted_reward = reward * np.logspace(0,np.log2(0.99)*(t-1),num=t,base=2)[::-1]
    discounted_reward -= np.mean(discounted_reward)
    discounted_reward /= np.std(discounted_reward)
    s *= discounted_reward

    dw1, dw2 = policy_backward(s,h,obvs,w2)

    w2 += dw2 * 0.001
    w1 += dw1 * 0.001

    scores, hidden, state = [],[],[]
