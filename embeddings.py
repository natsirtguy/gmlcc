'''Use machine learning for movie reviews.'''

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import log_loss, roc_curve, confusion_matrix
import tensorflow as tf
from tensorflow.data import Dataset as Ds
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  # noqa: E402


plt.ion()

# Set up pandas, tensorflow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 11
pd.options.display.float_format = '{:.2f}'.format

# Get data.
train_url = ('https://download.mlcc.google.com/'
             'mledu-datasets/sparse-data-embedding/train.tfrecord')
train_path = tf.keras.utils.get_file('train.tfrecord', train_url)


def _parse_fn(record):
    '''Parse a single item from a TFRecordDataset.'''
    features = {
        "terms": tf.VarLenFeature(dtype=tf.string),
        "labels": tf.FixedLenFeature(shape=[1], dtype=tf.float32)
    }

    parsed = tf.parse_single_example(record, features)
    terms = parsed['terms'].values

    return {'terms': terms}, parsed['labels']


def train_fn(ds, shuffle=10000, batch_size=1, repeat=None):
    '''Feed data for train.'''
    if shuffle:
        return lambda: (ds.shuffle(shuffle)
                        .padded_batch(batch_size, ds.output_shapes)
                        .repeat(repeat)
                        .make_one_shot_iterator().get_next())
    return lambda: (ds.padded_batch(batch_size, ds.output_shapes)
                    .repeat(repeat)
                    .make_one_shot_iterator().get_next())


def get_predictions(model, ds):
    '''Retrieve predictions from model.'''
    preds = model.predict(train_fn(ds, shuffle=False))
    preds = list(preds)
    probabilities = np.vstack(pred["probabilities"] for pred in preds)
    class_ids = np.hstack(pred["class_ids"] for pred in preds)
    return probabilities, class_ids


def bucketize(feature, fc, n_bins):
    '''Bin pandas series in dataframe df.

    Args:
      feature: pandas.Series
      fc: tensorflow.feature_column.numeric_column
      n_bins: int

    Returns:
      tensorflow.feature_column.bucketized_column
    '''

    qs = list(feature.quantile(np.linspace(0, 1, n_bins+1)))
    return tf.feature_column.bucketized_column(fc, qs)


informative_terms = ("bad", "great", "best", "worst", "fun",
                     "beautiful", "excellent", "poor", "boring",
                     "awful", "terrible", "definitely", "perfect",
                     "liked", "worse", "waste", "entertaining",
                     "loved", "unfortunately", "amazing",
                     "enjoyed", "favorite", "horrible",
                     "brilliant", "highly", "simple", "annoying",
                     "today", "hilarious", "enjoyable", "dull",
                     "fantastic", "poorly", "fails",
                     "disappointing", "disappointment", "not",
                     "him", "her", "good", "time", "?", ".", "!",
                     "movie", "film", "action", "comedy", "drama",
                     "family")


def train(ds, hidden_units=None, features=None, lr=1e-4,
          steps=100, optimizer=tf.train.GradientDescentOptimizer,
          l1_strength=None, batch_size=1, model=None, dropout=None,
          embedding=None, show_loss=False):
    '''Create and train a linear or neural network model.

    Args:
      ds: tf.data.DataSet, dataset to train on
      hidden_units: list of ints, number of neurons per layer
      features: list of selected features from examples
      lr: float, learning rate
      steps: int, number of steps to train
      optimizer: tf.train.Optimizer, type of optimizer to use
      l1_strength: float, strength of L1 regularization
      batch_size: int, number of examples per batch
      model: tensorflow LinearClassifier, previously trained model
      dropout: float between 0 and 1, probability to dropout a given node
      embedding: int, number of dimensions for embedding_column
      show_loss: bool, if True show loss over 10 periods

    Returns:
      A trained tensorflow.estimator.LinearClassifier or DNNClassifier.
    '''

    # Create feature columns and dictionary mapping feature names to them.
    # Construct informative terms categorical column.
    terms_fc = tf.feature_column.categorical_column_with_vocabulary_list(
        "terms", informative_terms)

    if hidden_units:
        if embedding:
            fcs = [tf.feature_column.embedding_column(terms_fc, embedding)]
        else:
            fcs = [tf.feature_column.indicator_column(terms_fc)]
    else:
        fcs = [terms_fc]

    if l1_strength:
        opt = optimizer(
            learning_rate=lr, l1_regularization_strength=l1_strength)
    else:
        opt = optimizer(learning_rate=lr)
    opt = tf.contrib.estimator.clip_gradients_by_norm(opt, 5.0)

    if not model:
        m_kwargs = {'feature_columns': fcs,
                    'optimizer': opt,
                    # 'config': tf.estimator.RunConfig(keep_checkpoint_max=1)
                    }
        if hidden_units:
            m_type = tf.estimator.DNNClassifier
            m_kwargs['hidden_units'] = hidden_units
            m_kwargs['dropout'] = dropout
        else:
            m_type = tf.estimator.LinearClassifier

        model = m_type(**m_kwargs)

    if show_loss:
        for _ in range(10):
            try:
                model.train(
                    train_fn(ds, shuffle=10000, batch_size=batch_size),
                    steps=steps//10)
                print("Loss:",
                      model.evaluate(train_fn(ds, shuffle=False),
                                     steps=1000)["loss"])
            except KeyboardInterrupt:
                print("\nTraining stopped by user.")
                break
    else:
        model.train(
            train_fn(ds, shuffle=1000, batch_size=batch_size),
            steps=steps)

    return model


def evaluate(model, ds, steps=1000, features=None):
    '''Check the mse on the validation set. '''
    results = model.evaluate(train_fn(ds, shuffle=False), steps=steps)

    for stat_name, stat_value in results.items():
        print(f"{stat_name:>20} | {stat_value}")

    return results


def count_nonzero_weights(model):
    '''Find number of nonzero weights.'''
    variables = model.get_variable_names()
    return sum(np.count_nonzero(model.get_variable_value(variable))
               for variable in variables if
               all((bad not in variable) for bad in
                   ['global_step', 'centered_bias_weight',
                    'bias_weight', 'Ftrl']))


# Train a linear classifier.
trained_linear = train(train_ds,
                       optimizer=tf.train.AdagradOptimizer,
                       # model=trained,
                       # hidden_units=[20, 20],
                       # l1_strength=0.5,
                       lr=1e-1, steps=1000, batch_size=25)
# Remove tf events.
list(map(os.remove,
         glob.glob(os.path.join(
             trained_linear.model_dir, "events.out.tfevents*"))))
print("Evaluated on training set:")
res = evaluate(trained_linear, train_ds)


# Train a neural net classifier.
trained_nn = None
for steps in (10, 90, 900):
    trained_nn = train(train_ds,
                       optimizer=tf.train.AdamOptimizer,
                       model=trained_nn,
                       hidden_units=[20, 20],
                       # dropout=.3,
                       # l1_strength=0.5,
                       show_loss=False,
                       embedding=2,
                       lr=1e-1, steps=steps, batch_size=25)
    # Remove tf events.
    list(map(os.remove,
             glob.glob(os.path.join(
                 trained_nn.model_dir, "events.out.tfevents*"))))
    print("Evaluated on training set:")
    res = evaluate(trained_nn, train_ds)

    # Investigate embedding layer.
    embed_dims = trained_nn.get_variable_value(
        'dnn/input_from_feature_columns/input_layer/'
        'terms_embedding/embedding_weights')
    plt.figure()
    plt.title(f"Embedding after {res['global_step']} steps")
    x_lims = np.array([embed_dims.T[0].min(), embed_dims.T[0].max()])
    y_lims = np.array([embed_dims.T[1].min(), embed_dims.T[1].max()])
    plt.xlim(1.5*x_lims)
    plt.ylim(1.5*y_lims)
    for x, y, term in zip(*embed_dims.T, informative_terms):
        plt.text(x, y, term, fontsize=8)


will_test = False
if will_test:
    # Get and parse the test data.
    test_url = ('https://download.mlcc.google.com/'
                'mledu-datasets/sparse-data-embedding/test.tfrecord')
    test_path = tf.keras.utils.get_file('test.tfrecord', test_url)
    test_ds = tf.data.TFRecordDataset(test_path)
    test_ds = test_ds.map(_parse_fn)

    print("Evaluated on test set:")
    tres = evaluate(trained_linear, test_ds)
