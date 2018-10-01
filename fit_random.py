'''Fit a set of random data points with a linear regressor.'''

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import tensorflow as tf  # pylint: disable=wrong-import-position


plt.ion()

# Create the random data.
np.random.seed(seed=1)
DATA = pd.DataFrame({'xs': pd.Series(np.random.rand(int(1e4))),
                     'ys': pd.Series(np.random.normal(.5, .1, size=int(1e4)))})


def plot_line(model, axis, start, end):
    '''Plot the line for a linear model on axis ax.'''
    weight = model.get_variable_value('linear/linear_model/xs/weights')[0]
    bias = model.get_variable_value('linear/linear_model/bias_weights')
    startend = np.array([start, end])
    axis.plot(startend, bias + weight*startend)


def trainer(learning_rate, steps, batch_size):
    '''Automate the learning process.'''

    # Plot sample of features and targets.
    samp = DATA.sample(n=500)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(samp['xs'], samp['ys'], '.')
    ax2.plot(samp['xs'], samp['ys'], '.')

    # Set up optimizer and linear regressor.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    model = tf.estimator.LinearRegressor(
        feature_columns=[tf.feature_column.numeric_column('xs')],
        optimizer=optimizer
    )

    # Create the data object for tensorflow. Shuffle, batch, and
    # repeat indefinitely.
    tdata = (tf.data.Dataset.from_tensor_slices(({'xs': DATA['xs']},
                                                 DATA['ys']))
             .shuffle(buffer_size=int(1e5)).batch(batch_size).repeat(None))

    n_periods = 5
    for _ in range(n_periods):
        # Train the model.
        model.train(lambda: tdata.make_one_shot_iterator().get_next(),
                    steps=steps//n_periods)

        # Find the current weight, bias and plot.
        plot_line(model, ax1, min(samp['xs']), max(samp['xs']))

    # Plot the final line separately.
    plot_line(model, ax2, min(samp['xs']), max(samp['xs']))
    weight = model.get_variable_value('linear/linear_model/xs/weights')[0][0]
    bias = model.get_variable_value('linear/linear_model/bias_weights')[0]
    fig.suptitle(
        f'lr: {learning_rate}, batch size: {batch_size}, steps: {steps},\n'
        f'final line: y = {weight:.2f}*x + {bias:.2f}')
