import tensorflow as tf 

IRIS_TRAINING = 'iris_training.csv'
IRIS_TEST = 'iris_test.csv'

FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

def input_fn(filename, n, batch, training):
	def _parse_csv(rows_string_tensor):
		num_features = len(FEATURE_KEYS)
		num_columns = num_features + 1
		columns = tf.decode_csv(rows_string_tensor, record_defaults = [[]] * num_columns)
		features = dict(zip(FEATURE_KEYS, columns[:num_features]))
		labels = tf.cast(columns[num_features], tf.int32)

		return features, labels

	def _input_fn():
		dataset = tf.data.TextLineDataset([filename])
		dataset = dataset.skip(1)
		dataset = dataset.map(_parse_csv)

		if training:
			dataset = dataset.shuffle(n)
			dataset = dataset.repeat()
		dataset = dataset.batch(batch)

		iterator = dataset.make_one_shot_iterator()
		features, labels = iterator.get_next()

		return features, labels

	return _input_fn

def main(unused_argv):
	tf.logging.set_verbosity(tf.logging.INFO)

	num_training_data = 120
	num_test_data = 30

	feature_columns = [tf.feature_column.numeric_column(key, shape=1) for key in FEATURE_KEYS]
	classifier = tf.estimator.DNNClassifier(feature_columns = feature_columns, hidden_units = [10, 20, 10], n_classes=3)

	train_input_fn = input_fn(IRIS_TRAINING, num_training_data, batch=32, training = True)
	classifier.train(input_fn=train_input_fn, steps=400)

	test_input_fn = input_fn(IRIS_TEST, num_test_data, batch = 32, training = False)
	scores = classifier.evaluate(input_fn=test_input_fn)

	print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))

if __name__ == '__main__':
	tf.app.run()




