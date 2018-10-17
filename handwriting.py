'''Use machine learning to classify handwritten digits.'''

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


def show_digit(features, labels, index):
    '''Show a digit from an mnist example.'''
    feature = features.iloc[index]
    label = labels.iloc[index][0]
    plt.matshow(np.array(feature).reshape(28, 28))
    plt.title(f"Label: {label}")


# Show some choice of training, validation digits.
show_digit(training_examples, training_labels, 38)
show_digit(validation_examples, validation_labels, 38)


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
    class_ids = np.vstack(pred["class_ids"] for pred in preds)
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


def train(examples, labels, hidden_units=None, features=None, lr=1e-4,
          steps=100, optimizer=tf.train.GradientDescentOptimizer,
          l1_strength=None, batch_size=1, model=None, dropout=None):
    '''Create and train a linear or neural network model.

    Args:
      examples: pandas.DataFrame with examples
      labels: pandas.DataFrame with labels
      hidden_units: list of ints, number of neurons per layer
      features: list of selected features from examples
      lr: float, learning rate
      steps: int, number of steps to train
      optimizer: tf.train.Optimizer, type of optimizer to use
      l1_strength: float, strength of L1 regularization
      batch_size: int, number of examples per batch
      model: tensorflow LinearClassifier, previously trained model
      dropout: float between 0 and 1, probability to dropout a given node

    Returns:
      A trained tensorflow.estimator.LinearClassifier or DNNClassifier.
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
        m_kwargs = {'feature_columns': fcs,
                    'optimizer': opt,
                    'n_classes': len(np.unique(labels)),
                    'config': tf.estimator.RunConfig(keep_checkpoint_max=1)}
        if hidden_units:
            m_type = tf.estimator.DNNClassifier
            m_kwargs['hidden_units'] = hidden_units
            m_kwargs['dropout'] = dropout
        else:
            m_type = tf.estimator.LinearClassifier

        model = m_type(**m_kwargs)

    for _ in range(10):
        try:
            model.train(
                train_fn(ds, batch_size=batch_size),
                steps=steps//10)
            predictions, class_ids = get_predictions(model, ds)
            print("Log loss:", log_loss(labels, predictions))
        except KeyboardInterrupt:
            print("\nTraining stopped by user.")
            print("Final log loss:", log_loss(labels, predictions))
            break

    return model


def validate(model, examples, labels, features=None):
    '''Check the mse on the validation set. '''
    if not features:
        features = examples.values

    ds = Ds.from_tensor_slices(
        ({"pixels": features}, np.array(labels)))
    predictions, class_ids = get_predictions(model, ds)
    print("Validation log loss:", log_loss(labels, predictions))

    return predictions, class_ids


def evaluate(model, examples, labels, features=None):
    '''Check the mse on the validation set. '''
    if not features:
        features = examples.values

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


# Train a linear classifier.
trained_linear = train(training_examples,
                       training_labels,
                       optimizer=tf.train.AdagradOptimizer,
                       # model=trained,
                       # hidden_units=[20, 10],
                       # l1_strength=0.5,
                       lr=1e-1, steps=100, batch_size=10)
# Remove tf events.
list(map(os.remove,
         glob.glob(os.path.join(
             trained_linear.model_dir, "events.out.tfevents*"))))


# Train a neural net classifier.
trained_nn = train(training_examples,
                   training_labels,
                   optimizer=tf.train.AdamOptimizer,
                   # model=trained_nn,
                   hidden_units=[100, 50],
                   dropout=.3,
                   # l1_strength=0.5,
                   lr=3e-4, steps=400, batch_size=50)
# Remove tf events.
list(map(os.remove,
         glob.glob(os.path.join(
             trained_nn.model_dir, "events.out.tfevents*"))))


# Find number of nonzero weights.
print("Number of nonzero weights:", count_nonzero_weights(trained_nn))


# Validate.
y, y_class_ids = validate(trained_nn, validation_examples,
                          validation_labels)

# Show target, predicted for some of the validation examples.
n = 50
plt.figure()
plt.plot(range(n),
         [int(cid) for cid in y_class_ids[-n:]], '.', label="Predicted")
plt.plot(range(n), validation_labels[-n:], label="Actual")
plt.legend()


# Evaluate the model.
res = evaluate(trained_nn, validation_examples, validation_labels)

# Create the confusion matrix and scale for number of examples.
cm = confusion_matrix(validation_labels, y_class_ids)
cmc = cm.copy()
cmc = cmc/cmc.sum(axis=1).reshape(1, 10)
plt.matshow(cmc)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix")

# Show only off-diagonal.
np.fill_diagonal(cmc, 0)
plt.matshow(cmc)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Off-diagonal confusion matrix")

# Construct ROC curves for each class.
for i in range(len(y[0])):
    prob = y[:, i]
    fpr, tpr, thesholds = roc_curve(validation_labels == i, prob)
    plt.figure()
    plt.plot(fpr, tpr, [0, 1], [0, 1])
    plt.title(f"ROC curve for class {i}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")


# Get the weights.
nn_weights = [trained_nn.get_variable_value(f"dnn/hiddenlayer_{i}/kernel")
              for i in range(2)]
nodes = nn_weights[0].shape[1]

# Visualize weights in the first layer.
n = 40
for i in range(n, n+10):
    plt.matshow(nn_weights[0].T[i].reshape(28, 28))


will_test = False
if will_test:
    # Get the test data.
    mnistt = pd.read_csv(
        "https://download.mlcc.google.com/mledu-datasets/mnist_test.csv",
        sep=",", header=None)
    test_examples = preprocess(mnistt)
    test_labels = preprocess_labels(mnistt)

    # Check the test.
    ty, ty_class_ids = validate(trained_nn, test_examples, test_labels)
    tres = evaluate(trained_nn, test_examples, test_labels)
