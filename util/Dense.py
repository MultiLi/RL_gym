import tensorflow as tf

class Dense():
    def __init__(self, input_x, out_dim):
        shape= input_x.get_shape().as_list()
        self.weight = tf.Variable(tf.truncated_normal([shape[-1], out_dim], stddev=1.0 / shape[-1]))
        self.bias = tf.zeros([out_dim])
        # self.out = tf.nn.bias_add(tf.matmul(input_x, self.weight), self.bias)
        self.out = tf.matmul(input_x, self.weight) + self.bias
