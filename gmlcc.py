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


def preprocess(hdf):
    '''Preprocess features, selecting some and making a new one.'''
    processed = hdf[list(set(hdf.columns) - {'median_house_value'})].copy()
    processed['rooms_per_person'] = hdf['total_rooms']/hdf['population']
    return processed


def preprocess_labels(hdf):
    '''Preprocess label by dividing by 1000.'''
    return (hdf[['median_house_value']]/1000).copy()


def examine(df):
    '''Examine data in dataframe.'''
    for column_index in range(0, len(chd.columns), 2):
        print(df.iloc[:, column_index:column_index+2].describe())


# Permute to avoid selecting data from one part of California.
perm = np.random.permutation(chd.index)
training_examples = preprocess(chd).iloc[perm[:12000], :]
training_labels = preprocess_labels(chd).iloc[perm[:12000], :]
validation_examples = preprocess(chd).iloc[perm[12000:], :]
validation_labels = preprocess_labels(chd).iloc[perm[12000:], :]

will_examine = False
if will_examine:
    # Examine the data.
    print("Training examples:")
    examine(training_examples)
    print()
    print("Training labels:")
    examine(training_labels)
    print()
    print("Validation examples:")
    examine(validation_examples)
    print()
    print("Validation labels:")
    examine(validation_labels)
    print()
    print("Correlations:")
    correlation_data = training_examples.copy()
    correlation_data['targets'] = training_labels['median_house_value']
    corrd = correlation_data.corr()
    examine(coord)
    print("Correlation to label:")
    print(corrd.iloc[-11:-1, -1])


for e, l in [(training_examples, training_labels),
             (validation_examples, validation_labels)]:
    plt.figure()
    plt.scatter(e['longitude'], e['latitude'], cmap='coolwarm',
                c=l['median_house_value']/l['median_house_value'].max())


def train_fn(ds, shuffle=True, batch_size=1, repeat=None):
    '''Feed data for train.'''
    return lambda: (ds.shuffle(shuffle).batch(batch_size).repeat(repeat)
                    .make_one_shot_iterator().get_next())


# Create training function.
def train(examples, labels,
          features=None, lr=1e-4, steps=100, batch_size=1, model=None):
    '''Create and train a linear regression model.'''
    # Create datasets.
    if not features:
        features = examples.columns
    fcs = [tf.feature_column.numeric_column(feature) for feature in features]

    ds = Ds.from_tensor_slices(
        ({feature: examples[feature] for feature in features}, labels))

    opt = tf.contrib.estimator.clip_gradients_by_norm(
        tf.train.GradientDescentOptimizer(learning_rate=lr),
        5.0)

    if not model:
        model = tf.estimator.LinearRegressor(fcs, optimizer=opt)

    for _ in range(10):
        model.train(
            train_fn(ds, batch_size=batch_size),
            steps=steps//10)
        preds = model.predict(
            lambda: ds.batch(1).make_one_shot_iterator().get_next())
        predictions = np.hstack(pred['predictions'] for pred in preds)
        print("Mean squared error: ", mse(predictions, labels))

    return model


def validate(model, examples, labels, features=None):
    '''Check the mse on the validation set. '''
    if not features:
        features = examples.columns

    ds = Ds.from_tensor_slices(
        ({feature: examples[feature] for feature in features}, labels))
    preds = model.predict(lambda: ds.batch(
        1).make_one_shot_iterator().get_next())
    predictions = np.hstack(pred['predictions'] for pred in preds)
    plt.close()
    plt.subplot(1, 2, 1)
    plt.scatter(examples['longitude'], examples['latitude'], cmap='coolwarm',
                c=labels['median_house_value'])
    plt.subplot(1, 2, 2)
    plt.scatter(examples['longitude'], examples['latitude'], cmap='coolwarm',
                c=predictions)

    print("Validation mse: ", mse(predictions, labels))


will_test = False
if will_test:
    # Get the test data.
    chdt = pd.read_csv(
        "https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv",
        sep=",")
    test_examples = preprocess(chdt)
    test_labels = preprocess_labels(chdt)

    # Check the test.
    validate(trained3, test_examples, test_labels, chosen)
