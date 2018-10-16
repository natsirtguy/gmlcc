'''Use machine learning to classify handwritten digits.'''

import os
import numpy as np
import pandas as pd
import matplotlib
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import log_loss, roc_curve
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
mnist = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/mnist_train_small.csv",
    sep=",", header=None)
mnist = mnist.head(10000)


def preprocess(df):
    '''Preprocess features.'''
    processed = (df.loc[:, 1:]/255).copy()
    return processed


def preprocess_labels(df):
    '''Preprocess label.'''
    labels_df = df.loc[:, 0:0].copy()
    return labels_df


# Permute to avoid selecting data from one part of California.
perm = np.random.permutation(mnist.index)
training_examples = preprocess(mnist).iloc[perm[:7500], :]
training_labels = preprocess_labels(mnist).iloc[perm[:7500], :]
validation_examples = preprocess(mnist).iloc[perm[7500:], :]
validation_labels = preprocess_labels(mnist).iloc[perm[7500:], :]


def show_digit(example):
    '''Show a digit from an mnist example.'''
    plt.matshow(np.array(example[1:]).reshape(28, 28))
    plt.title(f"Label: {example[0]}")


# Show some choice of training, validation digits.
show_digit(np.hstack((training_labels.iloc[37],
                      training_examples.iloc[37])))
show_digit(np.hstack((validation_labels.iloc[37],
                      validation_examples.iloc[37])))


def train_fn(ds, shuffle=True, batch_size=1, repeat=None):
    '''Feed data for train.'''
    return lambda: (ds.shuffle(shuffle).batch(batch_size).repeat(repeat)
                    .make_one_shot_iterator().get_next())


def get_predictions(model, ds):
    '''Retrieve predictions from model.'''
    preds = model.predict(
        lambda: ds.batch(1).make_one_shot_iterator().get_next())
    preds = list(preds)
    probabilities = np.vstack(pred["probabilities"] for pred in preds)
    classes = np.vstack(pred["classes"] for pred in preds)
    return probabilities, classes


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


def train(examples, labels, hidden_units=None, features=None, lr=1e-4,
          steps=100, optimizer=tf.train.GradientDescentOptimizer,
          l1_strength=None, batch_size=1, model=None):
    '''Create and train a linear model.

    Args:
      examples: pandas.DataFrame with examples
      labels: pandas.DataFrame with labels
      features: list of selected features from examples
      bucket_sizes: dict with size of buckets; if a value is None,
        don't bucketize that feature
      crosses: list of lists of features to be crossed
      lr: float, learning rate
      l1_strength: float, strength of L1 regularization
      steps: int, number of steps to train
      batch_size: int, number of examples per batch
      model: tensorflow LinearClassifier, previously trained model

    Returns:
      A trained tensorflow.estimator.LinearClassifier.
    '''

    # Create feature columns and dictionary mapping feature names to them.
    fcs = set([tf.feature_column.numeric_column('pixels', shape=784)])

    ds = Ds.from_tensor_slices(
        ({"pixels": examples.values}, np.array(labels)))

    if l1_strength:
        opt = optimizer(
            learning_rate=lr, l1_regularization_strength=l1_strength)
    else:
        opt = optimizer(learning_rate=lr)
    opt = tf.contrib.estimator.clip_gradients_by_norm(opt, 5.0)

    if not model:
        model = tf.estimator.LinearClassifier(
            fcs,
            optimizer=opt,
            n_classes=len(np.unique(labels)),
            config=tf.estimator.RunConfig(keep_checkpoint_max=1))

    for _ in range(10):
        model.train(
            train_fn(ds, batch_size=batch_size),
            steps=steps//10)
        predictions, classes = get_predictions(model, ds)
        print("Log loss:", log_loss(labels, predictions))

    return model


def validate(model, examples, labels, features=None):
    '''Check the mse on the validation set. '''
    if not features:
        features = examples.values

    ds = Ds.from_tensor_slices(
        ({"pixels": features}, np.array(labels)))
    predictions, classes = get_predictions(model, ds)
    print("Validation log loss:", log_loss(labels, predictions))

    return predictions


def evaluate(model, examples, labels, features=None):
    '''Check the mse on the validation set. '''
    if not features:
        features = examples.columns

    ds = Ds.from_tensor_slices(
        ({"pixels": features}, np.array(labels)))

    results = model.evaluate(
        lambda: ds.batch(1).make_one_shot_iterator().get_next())

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


# Train.
trained = train(training_examples,
                training_labels,
                optimizer=tf.train.AdagradOptimizer,
                # model=trained,
                # hidden_units=[20, 10],
                # features=chosen,
                # crosses=[["latitude", "longitude"]],
                # l1_strength=0.5,
                lr=3e-2, steps=100, batch_size=10)

# Find number of nonzero weights.
print("Number of nonzero weights:", count_nonzero_weights(trained))


# Validate.
y = validate(trained, validation_examples, validation_labels)

# Evaluate the model.
res = evaluate(trained, validation_examples, validation_labels)

# Plot the ROC curve.
# if "classifier" in str(type(trained)).casefold():
#     fpr, tpr, thresholds = roc_curve(validation_labels, y)
#     plt.figure()
#     plt.plot(fpr, tpr, [0, 1], [0, 1])

will_test = False
if will_test:
    # Get the test data.
    chdt = pd.read_csv(
        "https://download.mlcc.google.com"
        "/mledu-datasets/california_housing_test.csv",
        sep=",")
    test_examples = preprocess(chdt)
    test_labels = preprocess_labels(chdt)

    # Check the test.
    ty = validate(trained, test_examples, test_labels)
    tres = evaluate(trained, test_examples, test_labels)
    # if "classifier" in str(type(trained)).casefold():
    #     tfpr, ttpr, thresholds = roc_curve(test_labels, ty)
    #     plt.figure()
    #     plt.plot(tfpr, ttpr, [0, 1], [0, 1])
