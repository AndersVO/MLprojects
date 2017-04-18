'''
this file contains code from note 03 

Dataset Decription:
Name = fire and theft in chicago
X = fires per 1000 housing units
Y = thefts per 1000 population
sorted by zipcode

Total zipcodes: 42


we assume a linear relationship:
Y = wX+b
'''

# lets start important toolboxes

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = "data/fire_theft.xls"

# Step 1: Read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1,sheet.nrows)])
n_samples = sheet.nrows -1

# Step 2: create placehodlers for input X ( number of fires) and labels Y (numbers of thefts)
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# step 3: create weight and bias - initialize it to zero
w = tf.Variable(0.0, name="weights_1")
u = tf.Variable(0.0, name="weights_2")
b = tf.Variable(0.0, name="bias")

# step 4: construct model to predict Y ( numbers of thefts from the number of fires X.
#Y_predicted = tf.Variable(tf.random_normal([1]), name='bias')
#for pow_i in range(1, 3):
#    Y_predicted = tf.add(tf.multiply(tf.pow(X, pow_i), w), Y_predicted)
Y_predicted = X * X * w + X * u + b

# Step 5: Define a loss function in this case lets use the square error function

loss = tf.square(Y-Y_predicted, name="loss")

# Step 6: define and optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001).minimize(loss)

with tf.Session() as sess:
    # Step 7 initialize variables
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)
    # step 8: train model
    for i in range(10000): # run 100 epochs
        total_loss = 0
        for x,y in data:
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y:y})
        total_loss += l
        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    writer.close()
    # step 9 output values w and b
    w_value, b_value = sess.run([w, b])

# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w_value + b_value, 'r', label='Predicted data')
plt.legend()
plt.show()
