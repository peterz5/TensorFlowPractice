from __future__ import print_function

import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

learning_rate=.01
n_epochs = 50
batch_size = 100
step = 1

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

w = tf.Variable(tf.zeros([784, 10]), name='weights')
b = tf.Variable(tf.zeros([10]), name='biases')

pred = tf.nn.softmax(tf.add(tf.matmul(x, w), b))

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(n_epochs):
		avg_cost = 0
		total_batch = int(mnist.train.num_examples/batch_size)
		for i in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			_, c = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y})

		avg_cost += c/total_batch

		if (epoch+1) % step ==0:
			print('Epoch: ', str(epoch), ', cost:', str(avg_cost))
	print('Optimization finished')

	correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print('accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

