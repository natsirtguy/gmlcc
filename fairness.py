'''Learn about fairness in machine learning.'''

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sbn
import tensorflow as tf
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  # noqa: E402


plt.ion()

# Set up pandas, tensorflow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 11
pd.options.display.float_format = '{:.2f}'.format

# Prepare to get data.
columns = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]

# Get data.
train_df = pd.read_csv(
    "https://archive.ics.uci.edu/"
    "ml/machine-learning-databases/adult/adult.data",
    names=columns, sep=r'\s*,\s*', engine='python', na_values='?')
train_df = train_df.dropna(how='any', axis=0)


def examine(df):
    '''Examine data in dataframe.'''
    for i in range(0, len(df.columns)):
        print(df.iloc[:, i:i+1].describe())


def plot_hists(df):
    '''Plot histograms for categorical and numerical data.'''
    num_cols = df.columns[df.dtypes == "int64"]
    cat_cols = list(set(df.columns) - set(num_cols))
    for col in cat_cols:
        plt.figure()
        sbn.countplot(df[col])
    for col in num_cols:
        df[[col]].hist()


plot_hists(train_df)


# Notes: possible hard cap on age at 90, more men than women by a lot,
# mostly husbands, capital_gain and loss only for 8.4% and 4.7%,
# possible hard cap on hours_per_week at 99, education and
# education_num are redundant, possible hard cap on captial_gain at 99999,
# units of capital_gain and loss unclear (thousand? dollars?)

def preprocess(df):
    '''Prepocess the dataframe and return features and labels.'''
    features = df.copy()[list(set(columns) - {"income_bracket"})]
    labels = df.copy()[["income_bracket"]]
    return features, labels


# Separate into training and validation data.
perm = np.random.permutation(train_df.index)
all_features, all_labels = preprocess(train_df)
train_features = all_features.loc[perm[:22000]]
train_labels = all_labels.loc[perm[:22000]]
validate_features = all_features.loc[perm[:22000]]
validate_labels = all_labels.loc[perm[:22000]]


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


def train(examples, labels, hidden_units=None, bucket_sizes=None,
          features=None, lr=1e-4,
          steps=100, optimizer=tf.train.GradientDescentOptimizer,
          crosses=None,
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

    # Construct tensorflow dataset.
    ds = tf.data.Dataset.from_tensor_slices(
        ({feature: examples[feature] for feature in train_features},
         np.array(labels)))

    # Construct informative terms categorical column.
    # terms_fc = tf.feature_column.categorical_column_with_vocabulary_list(
    #     "terms", informative_terms)

    # if hidden_units:
    #     if embedding:
    #         fcs = [tf.feature_column.embedding_column(terms_fc, embedding)]
    #     else:
    #         fcs = [tf.feature_column.indicator_column(terms_fc)]
    # else:
    #     fcs = [terms_fc]

    if l1_strength:
        opt = optimizer(
            learning_rate=lr, l1_regularization_strength=l1_strength)
    else:
        opt = optimizer(learning_rate=lr)
    opt = tf.contrib.estimator.clip_gradients_by_norm(opt, 5.0)

    if not model:
        m_kwargs = {'feature_columns': fcs,
                    'optimizer': opt,
                    'config': tf.estimator.RunConfig(keep_checkpoint_max=1)
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


def evaluate(model, features, labels, steps=1000, features=None):
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
trained_linear = train(train_features, train_labels,
                       optimizer=tf.train.AdagradOptimizer,
                       # model=trained,
                       # hidden_units=[20, 20],
                       # l1_strength=0.5,
                       lr=1e-1, steps=1000, batch_size=25)
# Remove tf events.
list(map(os.remove,
         glob.glob(os.path.join(
             trained_linear.model_dir, "events.out.tfevents*"))))
print("Evaluated on validation set:")
res = evaluate(trained_linear, validate_labels)


# Train a neural net classifier, create word clouds.
trained_nn = None
term_choice = None
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
    word_cloud = True
    if word_cloud:
        if term_choice is None:
            # Pick 100 random terms on the first loop.
            term_choice = np.random.permutation(len(all_terms))[:100]
            random_terms = all_terms[term_choice]

        # Extract the weights for these terms.
        embed_weights = trained_nn.get_variable_value(
            'dnn/input_from_feature_columns/input_layer/'
            'terms_embedding/embedding_weights')
        random_weights = embed_weights[term_choice, :]

        # Plot the terms.
        plt.figure()
        plt.title(f"Embedding after {res['global_step']} steps")
        x_lims = np.array([random_weights.T[0].min(),
                           random_weights.T[0].max()])
        y_lims = np.array([random_weights.T[1].min(),
                           random_weights.T[1].max()])
        plt.xlim(1.5*x_lims)
        plt.ylim(1.5*y_lims)
        for x, y, term in zip(*random_weights.T, informative_terms):
            plt.text(x, y, term, fontsize=8)


will_test = False
if will_test:
    # Get and parse the test data.
    test_df = pd.read_csv("https://archive.ics.uci.edu/"
                          "ml/machine-learning-databases/adult/adult.test",
                          names=columns, sep=r'\s*,\s*', engine='python',
                          skiprows=[0], na_values='?')
    test_df = test_df.dropna(how='any', axis=0)

    print("Evaluated on test set:")
    tres = evaluate(trained_nn, test_ds)
