from __future__ import print_function

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

learning_rate = .01
n_epochs = 1000
show_step_every = 50

x_train = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])

y_train = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])

n_samples = x_train.shape[0]

X = tf.placeholder('float')
Y = tf.placeholder('float')

W = tf.Variable(np.random.randn(), name='weights')
b = tf.Variable(np.random.randn(), name='biases')

pred = tf.add(tf.multiply(X, W), b)

cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(n_epochs):
		sess.run(optimizer, feed_dict={X:x_train, Y:y_train})

		if (epoch+1) % show_step_every == 0:
			c = sess.run(cost, feed_dict = {X:x_train, Y:y_train})
			print('Epoch: ', '%04d' % (epoch+1), ' cost= ', '{:.9f}'.format(c))

	print('optimization finished!')

	training_cost = sess.run(cost, feed_dict={X:x_train, Y:y_train})

	print('final training cost = {:.5f}'.format(training_cost))

	plt.plot(x_train, y_train,'ro', label='Original_data')
	plt.plot(x_train, sess.run(W) * x_train + sess.run(b), label = 'fitted line')
	plt.legend()
	plt.show()

	x_test = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
	y_test = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

	test_cost = sess.run(tf.reduce_sum(tf.pow(pred- Y, 2))/(2*x_test.shape[0]), feed_dict={X:x_test, Y:y_test})
	print('Test cost {.4f}'.format(test_cost))
	print('train-test-difference {.4f}'.format(abs(training_cost -test_cost)))

	plt.plot(x_test, y_test, 'bo', label='Testing data')
	plt.plot(x_train, sess.run(W) * x_train+sess.run(b), label='fitted line')
	plt.legend()
	plt.show()

