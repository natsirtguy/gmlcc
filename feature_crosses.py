'''Practice using binning, crossed features for training.'''

import os
import numpy as np
import pandas as pd
import matplotlib
from sklearn.metrics import mean_squared_error as mse
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
chd = pd.read_csv(
    "https://download.mlcc.google.com"
    "/mledu-datasets/california_housing_train.csv", sep=",")


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
    for i in range(0, len(df.columns), 2):
        print(df.iloc[:, i:i+2].describe())


# Permute to avoid selecting data from one part of California.
perm = np.random.permutation(chd.index)
training_examples = preprocess(chd).iloc[perm[:12000], :]
training_labels = preprocess_labels(chd).iloc[perm[:12000], :]
validation_examples = preprocess(chd).iloc[perm[12000:], :]
validation_labels = preprocess_labels(chd).iloc[perm[12000:], :]

will_examine = True
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
    examine(corrd)
    print("Correlation to label:")
    print(corrd.iloc[-11:-1, -1])


for e, l in [(training_examples, training_labels),
             (validation_examples, validation_labels)]:
    plt.figure()
    plt.scatter(e['longitude'], e['latitude'], cmap='coolwarm',
                c=l['median_house_value']/l['median_house_value'].max())


def train_fn(ds, shuffle=10000, batch_size=1, repeat=None):
    '''Feed data for train.'''
    return lambda: (ds.shuffle(shuffle).batch(batch_size).repeat(repeat)
                    .make_one_shot_iterator().get_next())


def get_predictions(model, ds):
    '''Retrieve predictions from model.'''
    preds = model.predict(
        lambda: ds.batch(1).make_one_shot_iterator().get_next())
    return np.hstack(pred['predictions'] for pred in preds)


def o_h_encode(feature, df):
    '''One-hot encode a categorical Series feature in dataframe df.'''
    f_name = feature.name if feature.name else str(id(feature))
    values = set(feature)
    for value in values:
        df[f_name + '_' + str(value)] = (feature == value)
    return df


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


def train(examples, labels, features=None, bucket_sizes=None,
          crosses=None, lr=1e-4, steps=100, batch_size=1, model=None):
    '''Create and train a linear regression model.

    Args:
      examples: pandas.DataFrame with examples
      labels: pandas.DataFrame with labels
      features: list of selected features from examples
      bucket_sizes: dict with size of buckets
      crosses: list of lists of features to be crossed
      lr: float, learning rate
      steps: int, number of steps to train
      batch_size: int, number of examples per batch
      model: tensorflow.estimator.LinearRegressor, previously trained model

    Returns:
      A trained tensorflow.estimator.LinearRegressor.
    '''

    # Create feature columns and dictionary mapping feature names to them.
    if not features:
        features = examples.columns
    fcdict = {feature: tf.feature_column.numeric_column(feature)
              for feature in features}
    fcs = fcdict.values()

    # Use buckets if bucket_sizes is specified.
    if bucket_sizes:
        if len(bucket_sizes) != len(features):
            raise ValueError(
                'The number of buckets must match the number of features.')

        fcdict = {feature:
                  bucketize(examples[feature], fc, bucket_sizes[feature])
                  if bucket_sizes[feature] else fc
                  for feature, fc in fcdict.items()}

        fcs = fcdict.values()

    # Use crossed columns if crosses is specified.
    if crosses:
        for cross in crosses:
            cross_name = '_x_'.join(cross)
            cross_fc = [fcdict[feature] for feature in cross]
            fcdict[cross_name] = tf.feature_column.crossed_column(
                cross_fc, 1000)

        fcs = fcdict.values()

    ds = Ds.from_tensor_slices(
        ({feature: examples[feature] for feature in features}, labels))

    opt = tf.contrib.estimator.clip_gradients_by_norm(
        tf.train.FtrlOptimizer(learning_rate=lr),
        5.0)

    if not model:
        model = tf.estimator.LinearRegressor(fcs, optimizer=opt)

    for _ in range(10):
        model.train(
            train_fn(ds, batch_size=batch_size),
            steps=steps//10)
        predictions = get_predictions(model, ds)
        print("Mean squared error: ", mse(predictions, labels))

    return model


def validate(model, examples, labels, features=None):
    '''Check the mse on the validation set. '''
    if not features:
        features = examples.columns

    ds = Ds.from_tensor_slices(
        ({feature: examples[feature] for feature in features}, labels))
    predictions = get_predictions(model, ds)
    plt.close()
    plt.subplot(1, 2, 1)
    plt.scatter(examples['longitude'], examples['latitude'], cmap='coolwarm',
                c=labels['median_house_value'])
    plt.subplot(1, 2, 2)
    plt.scatter(examples['longitude'], examples['latitude'], cmap='coolwarm',
                c=predictions)

    print("Validation mse: ", mse(predictions, labels))


binned = {"longitude": 10,
          "latitude": 10,
          "housing_median_age": 7,
          "households": 7,
          "median_income": None,
          "rooms_per_person": None}

trained = train(training_examples, training_labels,
                features=binned.keys(), crosses=[["latitude", "longitude"]],
                bucket_sizes=binned, lr=1,
                steps=500, batch_size=100)

validate(trained, validation_examples,
         validation_labels, features=binned.keys())


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
    validate(trained, test_examples, test_labels, features=binned.keys())
