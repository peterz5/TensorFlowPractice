import math
import tensorflow as tf 

n_classes = 10

image_size = 28
image_pixels = image_size * image_size

def inference(images, hidden1_units, hidden2_units):
	with tf.name_scope('hidden1'):
		weights = tf.Variable(tf.truncated_normal([image_pixels, hidden1_units],stddev=1/math.sqrt(float(image_pixels))), name='weights')
		biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
		hidden1 = tf.nn.relu(tf.add(tf.matmul(images, weights), biases))

	with tf.name_scope('hidden2'):
		weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units], stddev=1/math.sqrt(float(hidden1_units))),name='weights')
		biases = tf.Variable(tf.zeros([hidden2_units]),name='biases')
		hidden2 = tf.nn.relu(tf.add(tf.matmul(hidden1, weights), biases))

	with tf.name_scope('linear_softmax'):
		weights = tf.Variable(tf.truncated_normal([hidden2_units, n_classes], stddev=1/math.sqrt(float(hidden2_units))), name='weights')
		biases= tf.Variable(tf.zeros([n_classes]), name='biases')
		logits = tf.add(tf.matmul(hidden2, weights), biases)
	return logits

def loss(logits, labels):
	labels = tf.to_int64(labels)
	return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

def training(loss, learning_rate):
	tf.summary.scalar('loss', loss)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	global_step = tf.Variable(0, name='global_step', trainable=False)
	train_op = optimizer.minimize(loss, global_step=global_step)
	return train_op

def evaluation(logits, labels):
	correct = tf.nn.in_top_k(logits, labels, 1)
	return tf.reduce_sum(tf.cast(correct, tf.int32))


