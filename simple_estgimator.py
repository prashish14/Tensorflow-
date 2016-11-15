
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import cross_validation
from sklearn import datasets
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn


def my_model(features, target):
  
 
  target = tf.one_hot(target, 3, 1, 0)


  normalizer_fn = layers.dropout
  normalizer_params = {'keep_prob': 0.9}
  features = layers.stack(features, layers.fully_connected, [10, 20, 10],
                          normalizer_fn=normalizer_fn,
                          normalizer_params=normalizer_params)

  # Compute logits (1 per class) and compute loss.
  logits = layers.fully_connected(features, 3, activation_fn=None)
  loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

  # Create a tensor for training op.
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
      learning_rate=0.1)

  return ({
      'class': tf.argmax(logits, 1),
      'prob': tf.nn.softmax(logits)}, loss, train_op)


def main(unused_argv):
  iris = datasets.load_iris()
  x_train, x_test, y_train, y_test = cross_validation.train_test_split(
      iris.data, iris.target, test_size=0.2, random_state=42)

  classifier = learn.Estimator(model_fn=my_model)
  classifier.fit(x_train, y_train, steps=1000)

  y_predicted = [
      p['class'] for p in classifier.predict(x_test, as_iterable=True)]
  score = metrics.accuracy_score(y_test, y_predicted)
  print('Accuracy: {0:f}'.format(score))


if __name__ == '__main__':
  tf.app.run()