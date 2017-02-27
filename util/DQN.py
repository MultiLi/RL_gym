import tensorflow as tf

class DQN():

    def __init__(self,input_shape, out_dim):
        self.input_shape = input_shape
        if len(input_shape)< 3:
            self.input_shape.append(1)

        self.x_in = tf.placeholder(tf.float32, shape=[None] + self.input_shape, name = 'input')

        # Convolutional Layer 1
        self.filter_1 = tf.Variable(tf.truncated_normal([8, 8, self.input_shape[2], 16], stddev=0.1))
        self.conv_1 = tf.nn.conv2d(input = self.x_in, filter = self.filter_1 , strides = [1, 4, 4, 1],padding = 'SAME')
        self.relu_1 = tf.nn.relu(self.conv_1)

        # Convolutional Layer 2
        self.filter_2 = tf.Variable(tf.truncated_normal([4, 4, 16, 32], stddev=0.1))
        self.conv_2 = tf.nn.conv2d(input = self.relu_1, filter = self.filter_2 , strides = [1, 2, 2, 1],padding = 'SAME')
        self.relu_2 = tf.nn.relu(self.conv_2)

        # Flatten
        self.flatten = tf.contrib.layers.flatten(inputs = self.relu_2)

        # FC Layer 1
        self.fc_1_o = tf.contrib.layers.fully_connected(inputs = self.flatten, num_outputs = 256)
        self.relu_fc_1 = tf.nn.relu(self.fc_1_o)

        # FC Layer 2
        self.fc_2_o = tf.contrib.layers.fully_connected(inputs = self.relu_fc_1, num_outputs = out_dim)

        # label
        self.y = tf.placeholder(tf.float32, shape=[None,out_dim], name = 'output')

        #loss
        self.mseloss = tf.contrib.losses.mean_squared_error(labels = self.y, predictions = self.fc_2_o )
