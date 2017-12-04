import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
Input:x -> (reshape) -> Input:x_image
 -> (convolution+ReLU):W_conv1,b_conv1 -> Hidden:h_conv1 -> (max-pooling) -> Hidden:h_pool1
 -> (convolution+ReLU):W_conv2,b_conv2 -> Hidden:h_conv2 -> (max-pooling) -> Hidden:h_pool2
 -> (reshape) -> h_pool2_reshape
 -> (fully connect+ReLU):W_fc1,b_fc1 -> Hidden:h_fc1 -> (dropout):keep_prob -> Hidden:h_fc1_drop
 -> (fully connect):W_fc2, b_fc2 -> Hidden:y_hat -> (softmax) -> Output:y_hat_softmax
'''

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32, [None, 10])

'''
    Build network
'''


def weight_variable(shape):
    # shape := [height, width, channel, num_filter]
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # shape := [num_filter]
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # strides := [batch, height, width, channel]
    return tf.nn.conv2d(x, W,
                        strides=[1, 1, 1, 1],
                        padding='SAME')


def max_pool_2x2(x):
    # ksize := [batch, height, width, channel]
    # strides := [batch, height, width, channel]
    return tf.nn.max_pool(x,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')


# Reshape
x_image = tf.reshape(x, [-1, 28, 28, 1])

# First convolution : 5 x 5 patch, 32 filters
FIRST_CONV_NUM_FILTER = 32
W_conv1 = weight_variable([5, 5, 1, FIRST_CONV_NUM_FILTER])
b_conv1 = bias_variable([FIRST_CONV_NUM_FILTER])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# First pooling : 2 x 2 max
h_pool1 = max_pool_2x2(h_conv1)

# Second convolution : 5 x 5 patch, 64 filters
SECOND_CONV_NUM_FILTER = 64
W_conv2 = weight_variable([5, 5, FIRST_CONV_NUM_FILTER, SECOND_CONV_NUM_FILTER])
b_conv2 = bias_variable([SECOND_CONV_NUM_FILTER])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# Second pooling : 2 x 2 max
h_pool2 = max_pool_2x2(h_conv2)

# Reshape
h_pool2_reshape = tf.reshape(h_pool2, [-1, 7 * 7 * SECOND_CONV_NUM_FILTER])

# First fully connect : 1024 units
FIRST_FULL_CONN_NUM_UNIT = 1024
W_fc1 = weight_variable([7 * 7 * SECOND_CONV_NUM_FILTER, FIRST_FULL_CONN_NUM_UNIT])
b_fc1 = bias_variable([FIRST_FULL_CONN_NUM_UNIT])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_reshape, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Second fully connect : 10 units
SECOND_FULL_CONN_NUM_UNIT = 10
W_fc2 = weight_variable([FIRST_FULL_CONN_NUM_UNIT, SECOND_FULL_CONN_NUM_UNIT])
b_fc2 = bias_variable([SECOND_FULL_CONN_NUM_UNIT])

y_hat = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

y_hat_softmax = tf.nn.softmax(y_hat)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y_true))

'''
    Builder trainer
'''
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_hat_softmax, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


'''
    Do train
'''
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for i in range(20000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch_xs, y_true: batch_ys, keep_prob: 1.0})

        if i % 10 == 0:
            print('Training', i, session.run(accuracy, feed_dict={x: batch_xs, y_true: batch_ys, keep_prob: 1.0}))

    print('Test', session.run(accuracy, feed_dict={x: mnist.test.images, y_true: mnist.test.labels, keep_prob: 1.0}))
