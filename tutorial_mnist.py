import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
Input:x -> (fully connect):W,b -> Hidden:y_hat -> (softmax) -> Output:y_hat_softmax
'''

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y_hat = tf.matmul(x, W) + b

y_hat_softmax = tf.nn.softmax(y_hat)

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_hat_softmax), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y_true))

# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_hat_softmax, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch_xs, y_true: batch_ys})

        if i % 10 == 0:
            print('Training', i, session.run(accuracy, feed_dict={x: batch_xs, y_true: batch_ys}))

    print('Test', session.run(accuracy, feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))
