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

# Create boundaries of latitude bins in the data.
chdlats = np.linspace(chd['latitude'].min(), chd['latitude'].max(), 11)


def preprocess(hdf):
    '''Preprocess features, selecting some and making a new one.'''
    processed = hdf[list(set(hdf.columns) - {'median_house_value'})].copy()
    processed['rooms_per_person'] = hdf['total_rooms']/hdf['population']

    # Create one-hot encoding for latitude.
    for i, lat in enumerate(chdlats[:-1]):
        feat = f'lat_{lat:.2f}_to_{chdlats[i+1]:.2f}'
        processed[feat] = np.logical_and(hdf['latitude'] >= lat,
                                         hdf['latitude'] <= chdlats[i+1])
    return processed


def preprocess_labels(hdf):
    '''Preprocess label by dividing by 1000.'''
    return (hdf[['median_house_value']]/1000).copy()


# Permute to avoid selecting data from one part of California.
perm = np.random.permutation(chd.index)
training_examples = preprocess(chd).iloc[perm[:12000], :]
training_labels = preprocess_labels(chd).iloc[perm[:12000], :]
validation_examples = preprocess(chd).iloc[perm[12000:], :]
validation_labels = preprocess_labels(chd).iloc[perm[12000:], :]

examine = False
if examine:
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
    corrd = correlation_data.corr()
    print(corrd.iloc[:, :3])
    print(corrd.iloc[:, 3:6])
    print(corrd.iloc[:, 6:9])
    print(corrd.iloc[:, 9:])
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


# Train with good hypers.
chosen = ['rooms_per_person', 'median_income']
trained = train(training_examples, training_labels, chosen,
                lr=1e-1, batch_size=2, steps=200)
validate(trained, validation_examples, validation_labels, chosen)

chosen = ['latitude', 'median_income']
trained2 = train(training_examples, training_labels, chosen,
                 lr=1e-3, batch_size=5, steps=500)
validate(trained2, validation_examples, validation_labels, chosen)

one_hot_lats = [feat for feat in training_examples if 'lat_' in feat]
chosen = one_hot_lats + ['median_income', 'rooms_per_person']
trained3 = train(training_examples, training_labels, chosen,
                 lr=3e-2, batch_size=2, steps=800)
validate(trained3, validation_examples, validation_labels, chosen)


# Get the test data.
chdt = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv",
    sep=",")
test_examples = preprocess(chdt)
test_labels = preprocess_labels(chdt)

# Check the test.
validate(trained3, test_examples, test_labels, chosen)
