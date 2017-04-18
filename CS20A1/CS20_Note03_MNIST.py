import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Read Data
MNIST = input_data.read_data_sets("data/mnist", one_hot=True)

# Step 2 define parameters

learning_rate = 0.01
batch_size = 128
n_epochs = 25
n_images = 10
# Create placeholders

X = tf.placeholder(tf.float32, [batch_size, 784])
Y = tf.placeholder(tf.float32, [batch_size, n_images])

# Create weights an biases

w = tf.Variable(tf.random_normal(shape=[784, n_images], stddev=0.01), name="weights")
u = tf.Variable(tf.random_normal(shape=[784, n_images], stddev=0.01), name="weights2")
b = tf.Variable(tf.zeros([1, n_images]), name="bias")


# Predict Y from X and w,b
logits = tf.matmul(X,u)+ b

# define loss function

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy)

# define training op
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    n_batches = int(MNIST.train.num_examples/batch_size)
    for i in range(n_epochs):
        for _ in range(n_batches):
            X_batch, Y_batch = MNIST.train.next_batch(batch_size)
            sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})


# test the model
    n_batches = int(MNIST.test.num_examples/batch_size)
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, Y_batch = MNIST.test.next_batch(batch_size)
        _, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict={X: X_batch, Y:Y_batch})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # simila
        total_correct_preds += sess.run(accuracy)
        print("Accuracy {0}".format(total_correct_preds/MNIST.test.num_examples))