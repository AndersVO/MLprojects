"""
This file will contain the code that are run through on note 2.
More info to come: ??
"""


import tensorflow as tf

x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x,y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # start the file writer
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    for _ in range(10):
            sess.run(z) # create the op add only when you need to compute it
    writer.close()
# Remember to close the writer
