'''Google machine learning crash course.'''

import os
import pandas as pd
import numpy as np
import matplotlib
import tensorflow as tf
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn import metrics
from matplotlib import cm
from tensorflow.python.data import Dataset

plt.ion()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

chd = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv",
    sep=",")

chd = chd.reindex(np.random.permutation(chd.index))
chd['median_house_value'] /= 1000

my_feature = chd[['total_rooms']]
feature_columns = [tf.feature_column.numeric_column('total_rooms')]
targets = chd['median_house_value']

# Create optimizer, use gradient clipping
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-7)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    '''Train a linear regression model with one feature.

    Args:
      features: pandas DataFrame of features
      targets: pandas Series of targets
      batch_size: int, size of batches to be passed to model
      shuffle: Boolean, whether to shuffle the data
      num_epochs: int, number of epochs to repeat the data.
        None = repeat indefinitely.
    Returns:
      Tuple of (features, labels) for next data batch.
    '''

    # Convert pandas features into dict of numpy arrays.
    features = {col: np.array(features[col]) for col in features}

    # Construct dataset, configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle data if specified.
    if shuffle:
        ds.shuffle(buffer_size=10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


linear_regressor.train(
    input_fn=lambda: my_input_fn(my_feature, targets),
    steps=100
)


def prediction_input_fn():
    return my_input_fn(
        my_feature, targets, num_epochs=1, shuffle=False
    )


predictions = linear_regressor.predict(input_fn=prediction_input_fn)
predictions = np.array([prediction['predictions'][0]
                        for prediction in predictions])

mse = metrics.mean_squared_error(predictions, targets)
rmse = np.sqrt(mse)


min_house_value = chd["median_house_value"].min()
max_house_value = chd["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print("Min. Median House Value: %0.3f" % min_house_value)
print("Max. Median House Value: %0.3f" % max_house_value)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % rmse)

cdata = pd.DataFrame({'predictions': pd.Series(predictions),
                      'targets': targets})
cdata.describe()

# Retrieve the final weight and bias generated during training.
weight = linear_regressor.get_variable_value(
    'linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

sample = chd.sample(n=300)
plt.plot(xs, ys, sample['total_rooms'], sample['median_house_value'], '.')


def train_model(learning_rate, steps, batch_size, input_feature='total_rooms'):
    '''Trains a linear regression model of one feature.

    Args:
      learning_rate: float, learning rate.
      steps: nonzero int, total number of training steps.
      batch_size: nonzero int, batch size.
      input_feature: string specifying a column to use as an input feature.
    '''

    periods = 10
    steps_per_period = steps / periods

    my_feature = input_feature
    my_feature_data = chd[[my_feature]]
    my_label = 'median_house_value'
    targets = chd[my_label]

    # Create feature columns.
    feature_columns = [tf.feature_column.numeric_column(my_feature)]

    # Create training and prediction functions.
    def training_input_fn():
        return my_input_fn(my_feature_data, targets, batch_size=batch_size)

    def prediction_input_fn():
        return my_input_fn(my_feature_data, targets,
                           num_epochs=1, shuffle=False)

    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(
        my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

    # Set up to plot the state of our model's line each period.
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned Line by Period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    sample = chd.sample(n=300)
    plt.scatter(sample[my_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    print('Training model...')
    print('RMSE (on training data):')
    rmses = []
    for period in range(periods):
        # Train the model.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        # Compute predictions.
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([prediction['predictions'][0]
                                for prediction in predictions])

        # Compute loss.
        rmse = np.sqrt(metrics.mean_squared_error(predictions, targets))
        print(f'period {period:02d}: {rmse:0.2f}')
        rmses.append(rmse)

        # Track weight and bias.
        weight = linear_regressor.get_variable_value(
            f'linear/linear_model/{input_feature}/weights')[0]
        bias = linear_regressor.get_variable_value(
            'linear/linear_model/bias_weights')

        # Plot line.
        xs = np.array([min(sample[my_feature]), max(sample[my_feature])])
        ys = bias + weight*xs
        plt.plot(xs, ys, color=colors[period])

    print('Model training finished.')

    # Plot loss metrics over periods.
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.tight_layout()
    plt.plot(rmses)

    # Print calibration data.
    cdata = pd.DataFrame({'predictions': pd.Series(predictions),
                          'targets': targets})
    print(cdata.describe())

    print(f'Final RMSE (on training data): {rmse:0.2f}')
