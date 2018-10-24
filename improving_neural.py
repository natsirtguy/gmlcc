'''Improve our neural net by manipulating featurs, using different optimizers.'''

import os
import itertools
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
chd = pd.read_csv(
    "https://download.mlcc.google.com"
    "/mledu-datasets/california_housing_train.csv", sep=",")


def linear_scale(series):
    '''Scale a pd.Series linearly so that the the values are in [-1,1].'''
    s = series
    return 2.0*(s - s.min())/(s.max() - s.min()) - 1.0


def normalize_linear_scale(df):
    '''Apply linear_scale to all features in dataframe df.'''
    df_copy = df.copy()
    for feature in df_copy:
        df_copy[feature] = linear_scale(df_copy[feature])
    return df_copy


def preprocess(hdf):
    '''Preprocess features, selecting some and making a new one.'''
    processed = hdf[list(set(hdf.columns) - {'median_house_value'})].copy()
    processed['rooms_per_person'] = hdf['total_rooms']/hdf['population']
    return normalize_linear_scale(np.log(np.abs(processed)))


def preprocess_labels(hdf):
    '''Preprocess label by dividing by 1000.'''
    labels_df = hdf.copy()[['median_house_value']]
    labels_df['median_house_value'] /= 1000
    return labels_df


def examine(df):
    '''Examine data in dataframe.'''
    for i in range(0, len(df.columns), 2):
        print(df.iloc[:, i:i+2].describe())


# Permute to avoid selecting data from one part of California, and normalize.
perm = np.random.permutation(chd.index)
training_examples = preprocess(chd).iloc[perm[:12000], :]
training_labels = preprocess_labels(chd).iloc[perm[:12000], :]
validation_examples = preprocess(chd).iloc[perm[12000:], :]
validation_labels = preprocess_labels(chd).iloc[perm[12000:], :]


def scatter_pairs(df):
    '''Show scatter plots of pairs of features against each other.'''
    for f1, f2 in itertools.combinations(df.columns, 2):
        plt.figure()
        plt.plot(df[f1], df[f2], '.')
        plt.xlabel(f1)
        plt.ylabel(f2)


show_scatters = False
if show_scatters:
    scatter_pairs(chd)

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
    correlation_data = training_examples.copy()
    correlation_data['targets'] = training_labels['median_house_value']
    corrd = correlation_data.corr()
    print("Correlation to label:")
    print(corrd.iloc[-11:-1, -1])


for e, l in [(training_examples, training_labels),
             (validation_examples, validation_labels)]:
    plt.figure()
    plt.scatter(e['longitude'], e['latitude'], cmap='coolwarm',
                c=l["median_house_value"])


def train_fn(ds, shuffle=10000, batch_size=1, repeat=None):
    '''Feed data for train.'''
    return lambda: (ds.shuffle(shuffle).batch(batch_size).repeat(repeat)
                    .make_one_shot_iterator().get_next())


def get_predictions(model, ds):
    '''Retrieve predictions from model.'''
    preds = model.predict(
        lambda: ds.batch(1).make_one_shot_iterator().get_next())
    if "classifier" in str(type(model)).casefold():
        pred_name = 'logistic'
    else:
        pred_name = 'predictions'
    return np.hstack(pred[pred_name] for pred in preds)


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


def train(examples, labels, hidden_units=None, features=None, bucket_sizes=None,
          crosses=None, classifier=False, lr=1e-4, steps=100,
          optimizer=tf.train.GradientDescentOptimizer,
          l1_strength=None, batch_size=1, model=None):
    '''Create and train a DNN model.

    Args:
      examples: pandas.DataFrame with examples
      labels: pandas.DataFrame with labels
      features: list of selected features from examples
      classifier: Boolean, whether to train a classifier
      bucket_sizes: dict with size of buckets; if a value is None,
        don't bucketize that feature
      crosses: list of lists of features to be crossed
      lr: float, learning rate
      l1_strength: float, strength of L1 regularization
      steps: int, number of steps to train
      batch_size: int, number of examples per batch
      model: tensorflow DNNRegressor or DNNClassifer,
        previously trained model

    Returns:
      A trained tensorflow.estimator.DNNRegressor.
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
        ({feature: examples[feature] for feature in features},
         np.array(labels)))

    if l1_strength:
        opt = optimizer(learning_rate=lr,
                        l1_regularization_strength=l1_strength)
    else:
        opt = optimizer(learning_rate=lr)

    opt = tf.contrib.estimator.clip_gradients_by_norm(opt, 5.0)

    if not model:
        model = tf.estimator.DNNRegressor(hidden_units, fcs, optimizer=opt)

    for _ in range(10):
        try:
            model.train(
                train_fn(ds, batch_size=batch_size),
                steps=steps//10)
            predictions = get_predictions(model, ds)
            if classifier:
                print("Log loss:", log_loss(labels, predictions))
            else:
                print("Mean squared error:", mse(predictions, labels))
        except KeyboardInterrupt:
            print("\nTraining stopped by user.")
            if classifier:
                print("Final log loss:", log_loss(labels, predictions))
            else:
                print("Final mean squared error:", mse(predictions, labels))
            break

    return model


def validate(model, examples, labels, features=None):
    '''Check the mse on the validation set.'''
    if not features:
        features = examples.columns

    ds = Ds.from_tensor_slices(
        ({feature: examples[feature] for feature in features}, labels))
    predictions = get_predictions(model, ds)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(examples['longitude'], examples['latitude'], cmap='coolwarm',
                c=labels.iloc[:, 0])
    plt.subplot(1, 2, 2)
    plt.scatter(examples['longitude'], examples['latitude'], cmap='coolwarm',
                c=predictions)
    if "classifier" in str(type(model)).casefold():
        print("Validation log loss:", log_loss(labels, predictions))
    else:
        print("Validation mse:", mse(predictions, labels))
    return predictions


def evaluate(model, examples, labels, features=None):
    '''Check the mse on the validation set.'''
    if not features:
        features = examples.columns

    ds = Ds.from_tensor_slices(
        ({feature: examples[feature] for feature in features}, labels))

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
binned = {"longitude": 50,
          "latitude": 50,
          # "housing_median_age": 10,
          # "households": 10,
          # "median_income": None,
          # "rooms_per_person": None
          }
trained = train(training_examples,
                training_labels,
                # model=trained,
                hidden_units=[20, 10],
                # features=binned.keys(),
                # bucket_sizes=binned,
                optimizer=tf.train.AdagradOptimizer,
                # crosses=[["latitude", "longitude"]],
                # l1_strength=0.5,
                lr=5e-1, steps=1000, batch_size=50)

# Find number of nonzero weights.
print("Number of nonzero weights:", count_nonzero_weights(trained))


# Validate.
y = validate(trained, validation_examples, validation_labels,
             # features=binned.keys()
             )

# Evaluate the model.
res = evaluate(trained, validation_examples, validation_labels)

# Map out the behavior of the model in latitude, longitude.
map_lat_long = True
if map_lat_long:
    n = 3000
    grid_lat = np.random.rand(n)
    grid_long = np.random.rand(n)
    medians = training_examples.median()
    grid_features = pd.DataFrame(
        {"latitude": grid_lat, "longitude": grid_long})
    for feature in set(training_examples.columns) - {"latitude", "longitude"}:
        grid_features[feature] = pd.Series(np.repeat(medians[feature], n))
    grid_labels = pd.DataFrame({"median_house_value": np.repeat(100, n)})
    gy = validate(trained, grid_features, grid_labels)


# Plot the ROC curve.
if "classifier" in str(type(trained)).casefold():
    fpr, tpr, thresholds = roc_curve(validation_labels, y)
    plt.figure()
    plt.plot(fpr, tpr, [0, 1], [0, 1])

will_test = True
if will_test:
    # Get the test data.
    chdt = pd.read_csv(
        "https://download.mlcc.google.com"
        "/mledu-datasets/california_housing_test.csv",
        sep=",")
    test_examples = preprocess(chdt)
    test_labels = preprocess_labels(chdt)

    # Check the test.
    ty = validate(trained, test_examples, test_labels,
                  # features=binned.keys()
                  )
    tres = evaluate(trained, test_examples, test_labels,
                    # features=binned.keys()
                    )
    if "classifier" in str(type(trained)).casefold():
        tfpr, ttpr, thresholds = roc_curve(test_labels, ty)
        plt.figure()
        plt.plot(tfpr, ttpr, [0, 1], [0, 1])
