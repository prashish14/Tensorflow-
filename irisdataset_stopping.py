

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil

from sklearn import datasets
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import tensorflow as tf

from tensorflow.contrib import learn


def clean_folder(folder):
  """Cleans the given folder if it exists."""
  try:
    shutil.rmtree(folder)
  except OSError:
    pass


def main(unused_argv):
  iris = datasets.load_iris()
  x_train, x_test, y_train, y_test = train_test_split(
      iris.data, iris.target, test_size=0.2, random_state=42)

  x_train, x_val, y_train, y_val = train_test_split(
      x_train, y_train, test_size=0.2, random_state=42)
  val_monitor = learn.monitors.ValidationMonitor(
      x_val, y_val, early_stopping_rounds=200)

  model_dir = '/tmp/iris_model'
  clean_folder(model_dir)

  # classifier with early stopping on training data
  classifier1 = learn.DNNClassifier(
      feature_columns=learn.infer_real_valued_columns_from_input(x_train),
      hidden_units=[10, 20, 10], n_classes=3, model_dir=model_dir)
  classifier1.fit(x=x_train, y=y_train, steps=2000)
  predictions1 = list(classifier1.predict(x_test, as_iterable=True))
  score1 = metrics.accuracy_score(y_test, predictions1)

  model_dir = '/tmp/iris_model_val'
  clean_folder(model_dir)

  # classifier with early stopping on validation data, save frequently for
  # monitor to pick up new checkpoints.
  classifier2 = learn.DNNClassifier(
      feature_columns=learn.infer_real_valued_columns_from_input(x_train),
      hidden_units=[10, 20, 10], n_classes=3, model_dir=model_dir,
      config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))
  classifier2.fit(x=x_train, y=y_train, steps=2000, monitors=[val_monitor])
  predictions2 = list(classifier2.predict(x_test, as_iterable=True))
  score2 = metrics.accuracy_score(y_test, predictions2)

  # In many applications, the score is improved by using early stopping
  print('score1: ', score1)
  print('score2: ', score2)
  print('score2 > score1: ', score2 > score1)


if __name__ == '__main__':
  tf.app.run()