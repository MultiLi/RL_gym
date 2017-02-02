import tensorflow as tf
import numpy as np
import gym
from matplotlib import pyplot as plt

env = gym.make('FrozenLake-v0')
REWARD_DECAY = 0.95
LRATE = 0.03
ep_Start = 0.05
ep_End = 0.95

input_x = tf.placeholder(tf.float32,[None,16],name = 'input_x')
label_y = tf.placeholder(tf.float32,[None,4],name = 'input_x')
weight = tf.Variable(tf.zeros([16,4]),name = 'weight')
Q = tf.matmul(input_x, weight)
loss = tf.reduce_sum(tf.square(label_y - Q))
dw = tf.Variable(tf.zeros([16,4]),name = 'dw')
adam = tf.train.AdamOptimizer([dw,weight])

init = tf.global_variables_initializer()

Rs = []
Avr = []

with tf.Session() as sess:

    iter = 0
    while iter < 5000:
        sess.run(init)
        s = env.reset()
        x = np.zeros([1,16],dtype = np.float32)
        x[0,s] = 1
        i = 0
        rr = 0
        while i < 100:
            out = sess.run(Q,feed_dict = {input_x:x})
            action = env.action_space.sample() if np.random.rand() > ep_Start + ep_End * iter / 1000 else np.argmax(out)
            s1,r,d,_ = env.step(action)
            x1 = np.zeros([1,16],dtype = np.float32)
            x1[0,s1] = 1
            out = REWARD_DECAY * sess.run(Q,feed_dict = {input_x:x1})
            out[0,action] += r
            grad = sess.run(dw, feed_dict = {input_x:x, label_y:out})
            sess.run(weight,feed_dict={dw:grad})
            rr += r
            s = s1
            x = x1
            i += 1
            if d:
                break

        Rs.append(r)
        if iter % 1000 == 0:
            Avr.append(np.mean(np.array(Rs[-1000:])))
        iter += 1

plt.plot(np.array(Rs))
plt.figure()
plt.plot(np.array(Avr))
plt.show()
