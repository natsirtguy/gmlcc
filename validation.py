'''Practice using validation for training.'''

import os
import numpy as np
import pandas as pd
import matplotlib
from sklearn.metrics import mean_squared_error as mse
import tensorflow as tf  # pylint: disable=wrong-import-position
from tensorflow.data import Dataset as Ds
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position


plt.ion()

# Set up pandas, tensorflow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 11
pd.options.display.float_format = '{:.2f}'.format

# Get data.
chd = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")


def preprocess(chd):
    '''Preprocess features, selecting some and making a new one.'''
    processed = chd[list(set(chd.columns) - {'median_house_value'})].copy()
    processed['rooms_per_person'] = chd['total_rooms']/chd['population']
    return processed


def preprocess_labels(chd):
    '''Preprocess label by dividing by 1000.'''
    return (chd[['median_house_value']]/1000).copy()


# Permute to avoid selecting data from one part of California.
perm = np.random.permutation(chd.index)
training_examples = preprocess(chd).iloc[perm[:12000], :]
training_labels = preprocess_labels(chd).iloc[perm[:12000], :]
validation_examples = preprocess(chd).iloc[perm[12000:], :]
validation_labels = preprocess_labels(chd).iloc[perm[12000:], :]

# Examine the data.
print("Training examples:")
print(training_examples.iloc[:, :3].describe())
print(training_examples.iloc[:, 3:7].describe())
print(training_examples.iloc[:, 7:].describe())
print()
print("Training targets:")
print(training_labels.describe())
print()
print("Validation examples:")
print(validation_examples.iloc[:, :3].describe())
print(validation_examples.iloc[:, 3:7].describe())
print(validation_examples.iloc[:, 7:].describe())
print("Validation targets:")
print(validation_labels.describe())
print()

correlation_data = training_examples.copy()
correlation_data['targets'] = training_labels['median_house_value']
corr = correlation_data.corr()
print(corr.iloc[:, :3])
print(corr.iloc[:, 3:6])
print(corr.iloc[:, 6:9])
print(corr.iloc[:, 9:])


for e, l in [(training_examples, training_labels),
             (validation_examples, validation_labels)]:
    plt.figure()
    plt.scatter(e['latitude'], e['longitude'], cmap='coolwarm',
                c=l['median_house_value']/l['median_house_value'].max())

# Create feature columns.
chosen = ['rooms_per_person', 'median_income']
fcs = [tf.feature_column.numeric_column(feature) for feature in chosen]


def train_fn(ds, shuffle=True, batch_size=1, repeat=None):
    '''Feed data for train.'''
    return lambda: (ds.shuffle(shuffle).batch(batch_size).repeat(repeat)
                    .make_one_shot_iterator().get_next())


# Create training function.
def train(examples, labels, lr=1e-4, steps=100, batch_size=1, model=None):
    '''Create and train a linear regression model.'''
    # Create datasets.
    ds = Ds.from_tensor_slices(
        ({feature: examples[feature] for feature in chosen}, labels))

    opt = tf.contrib.estimator.clip_gradients_by_norm(
        tf.train.GradientDescentOptimizer(learning_rate=lr),
        5.0)

    if not model:
        model = tf.estimator.LinearRegressor(fcs, optimizer=opt)

    for _ in range(10):
        model.train(train_fn(ds), steps=steps//10)
        preds = model.predict(
            lambda: ds.batch(1).make_one_shot_iterator().get_next())
        predictions = np.hstack(pred['predictions'] for pred in preds)
        print("Mean squared error: ", mse(predictions, labels))

    return model


def validate(model, examples, labels):
    '''Check the mse on the validation set. '''
    ds = Ds.from_tensor_slices(
        ({feature: examples[feature] for feature in chosen}, labels))
    preds = model.predict(lambda: ds.batch(
        1).make_one_shot_iterator().get_next())
    predictions = np.hstack(pred['predictions'] for pred in preds)
    plt.subplot(1, 2, 1)
    plt.scatter(examples['latitude'], examples['longitude'], cmap='coolwarm',
                c=labels['median_house_value'])
    plt.subplot(1, 2, 2)
    plt.scatter(examples['latitude'], examples['longitude'], cmap='coolwarm',
                c=predictions)

    print("Validation mse: ", mse(predictions, labels))


# Train with good hypers.
trained = train(training_examples, training_labels,
                lr=3e-5, batch_size=2, steps=200)

# Get the test data.
chdt = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv",
    sep=",")
test_examples = preprocess(chdt)
test_labels = preprocess_labels(chdt)

# Check the test.
# validate(trained, test_examples, test_labels)
