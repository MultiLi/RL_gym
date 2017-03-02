import numpy as np
import gym
from gym import wrappers
import sys
sys.path.append('..')
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import RMSprop
from util.exp_replay import Exp_Replay as Memory

ETA = 0.001
GAMMA = 0.99
MOMENTUM = 0
EPS_START = 0.05
EPS_END = 0.95
MEM_SIZE = 10000
BATCH = 10
MSIZE = 20
TIMESTEP = 5
EP = 20
ITER = 50



def naiveQN(layer,activation = 'tanh'):
    model = Sequential()
    model.add(SimpleRNN(layer[1], input_shape = (None, layer[0]),return_sequences=True))
    for i in range(2,len(layer) - 1):
        model.add(Dense(layer[i], input_dim = layer[i-1]))
        model.add(Activation(activation))
    model.add(Dense(layer[-1], input_dim = layer[-2]))
    optimizer = RMSprop(lr = ETA, decay = 0.01)
    model.compile(optimizer= optimizer, loss='mean_squared_error')

    return model

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1')
o = env.reset()
layer = [env.observation_space.shape[0],64,32, env.action_space.n]
memory = Memory(MEM_SIZE,[layer[0]])

model = naiveQN(layer,activation = 'relu')

life = []

rendering = False

episode = 0
dummy_i = 0
turns = 0
avrQ = []
avrq = 0

while episode < 20000:
    turns += 1
    dummy_i += 1

    model.reset_states()
    # if rendering:
        # env.render()

    out_s = model.predict(o.reshape(1,1,-1))
    action = np.argmax(out_s)
    if np.random.rand() > EPS_START + turns * EPS_END / 2000:
        action = env.action_space.sample()

    avrq += np.ravel(out_s)[action]

    o1,reward,done,_ = env.step(action) # take a random action

    memory.insert(o,action,0,o1, False)
    o = o1

    if done or dummy_i == 200:
        memory.t[memory.pointer] = True
        memory.r[memory.pointer] = -1
        o = env.reset()
        episode += 1
        life.append(dummy_i)
        avrQ.append(avrq * 1.0 / dummy_i)
        print 'Episode :' + str(episode), avrQ[-1],dummy_i
        exist = 0
        dummy_i = 0
        if np.mean(np.array(life[-5:])) > 185:
            rendering = True
            continue

    if memory.length < 3 * BATCH*MSIZE:
        continue

    ind_ = np.random.permutation(memory.length)[:BATCH * MSIZE]

    for i in range(BATCH):
        model.reset_states()
        ind = (np.random.permutation(memory.length - TIMESTEP)+ TIMESTEP)[:MSIZE]
        ind = ind.reshape(-1,1) + np.arange(TIMESTEP).reshape(1,-1)
        ind[ ind >= memory.length] -= memory.length
        out_s1 = model.predict(memory.s1[ind])


        # For non-terminal states, Q(s,a) = r(s,a) + gamma * max(Q(s',a')) (Empirical Mean)
        y = memory.r[ind] + GAMMA * np.max(out_s1,axis = 2) * (np.logical_not(memory.t[ind]))
        out_s = model.predict(memory.s[ind])
        # print y.shape
        out_s[np.matlib.repmat(np.arange(MSIZE).reshape(-1,1),1,TIMESTEP),\
              np.matlib.repmat(np.arange(TIMESTEP).reshape(1,-1),MSIZE,1),\
                                memory.a[ind]]   = y

        model.fit(memory.s[ind], out_s, batch_size = MSIZE, verbose = 0, nb_epoch = 1 )
