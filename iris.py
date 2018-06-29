import tensorflow as tf 
import pandas as pd 
import numpy as np 

IRIS_TRAINING = 'iris_train.csv'
IRIS_TEST = 'iris_test.csv'

n_h1 = 5
n_h2 = 10
n_h3 = 5
n_classes = 3

features = tf.placeholder('float',[None, 120])
labels = tf.placeholder('float', [120])

def preprocess(filename):
	df =  pd.read_csv(IRIS_TRAINING)

	x = np.asarray(df.drop('label', axis=1)).astype(float)
	y = np.asarray(df['label']).astype(float)

	x = tf.convert_to_tensor(x, dtype='float')
	y = tf.convert_to_tensor(y, dtype='float')

	x = tf.transpose(x)
	return x,y,x.shape[1]

def neural_net(data):

	h1 = {'weights': tf.Variable(tf.random_normal([120, n_h1])), 'biases': tf.Variable(tf.random_normal([n_h1]))}
	h2 = {'weights': tf.Variable(tf.random_normal([n_h1, n_h2])), 'biases': tf.Variable(tf.random_normal([n_h2]))}
	h3 = {'weights': tf.Variable(tf.random_normal([n_h2, n_h3])), 'biases': tf.Variable(tf.random_normal([n_h3]))}
	output = {'weights': tf.Variable(tf.random_normal([n_h3, n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data, h1['weights']), h1['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, h2['weights']), h2['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, h3['weights']), h3['biases'])
	l3 = tf.nn.relu(l3)

	output_layer = tf.add(tf.matmul(l3, output['weights']), output['biases'])

	print(output_layer.shape)
	return output_layer

def train(x, y, n_samples):
	n_epochs = 10

	predictions = neural_net(x)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
	optimizer = tf.train.AdamsOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(n_epochs):
			epoch_loss = 0
			for _ in range(n_samples):
				_, c = sess.run([optimizer, cost], feed_dict = {x:x, y:y})
				epoch_loss += c
			print('epoch ', epoch, ' completed out of ', n_epochs, ' , ', epoch_loss)

		correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

features, labels, n_samples = preprocess(IRIS_TRAINING)
train(features, labels, n_samples)



