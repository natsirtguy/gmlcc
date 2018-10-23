'''Learn about fairness in machine learning.'''

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import tensorflow as tf
matplotlib.use('TkAgg')
import seaborn as sbn           # noqa: E402
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
train_examples = pd.read_csv(
    "https://archive.ics.uci.edu/"
    "ml/machine-learning-databases/adult/adult.data",
    names=columns, sep=r'\s*,\s*', engine='python', na_values='?')
train_examples = train_examples.dropna(how='any', axis=0)


def examine(examples):
    '''Examine data in dataframe.'''
    for i in range(0, len(examples.columns)):
        print(examples.iloc[:, i:i+1].describe())


def num_cat(df):
    '''Return the numerical and categorical columns.'''
    num_cols = df.columns[np.logical_or(df.dtypes == "int64",
                                        df.dtypes == "float64")]
    cat_cols = list(set(df.columns) - set(num_cols))
    return num_cols, cat_cols


def plot_hists(examples):
    '''Plot histograms for categorical and numerical data.'''
    num_cols, cat_cols = num_cat(examples)
    for col in cat_cols:
        plt.figure()
        sbn.countplot(examples[col])
    for col in num_cols:
        examples[[col]].hist()


plot_hists(train_examples)


# Notes: possible hard cap on age at 90, more men than women by a lot,
# mostly husbands, capital_gain and loss only for 8.4% and 4.7%,
# possible hard cap on hours_per_week at 99, education and
# education_num are redundant, possible hard cap on captial_gain at 99999,
# units of capital_gain and loss unclear (thousand? dollars?)

def preprocess(examples):
    '''Prepocess the dataframe and return features and labels.'''
    features = examples.copy()[list(
        set(columns) - {"income_bracket", "education_num"})]

    num_cols, cat_cols = num_cat(features)

    # Scale numerical columns, use z-score.
    for feature in num_cols:
        s = features[feature]
        features[feature] = (s - s.mean())/(s.std())

    labels = pd.DataFrame()
    labels = examples.copy()[["income_bracket"]]
    labels = (labels == ">50K")

    return features, labels


# Separate into training and validation data.
perm = np.random.permutation(train_examples.index)
all_features, all_labels = preprocess(train_examples)
train_features = all_features.loc[perm[:22000]]
train_labels = all_labels.loc[perm[:22000]]
validate_features = all_features.loc[perm[:22000]]
validate_labels = all_labels.loc[perm[:22000]]


def get_predictions(model, examples):
    '''Retrieve predictions from model.'''
    preds = model.predict(input_fn(examples, None, shuffle=False))
    preds = list(preds)
    probabilities = np.vstack(pred["probabilities"] for pred in preds)
    class_ids = np.hstack(pred["class_ids"] for pred in preds)
    return probabilities, class_ids


def bucketize(feature, fc, n_bins):
    '''Bin pandas series in dataframe examples.

    Args:
      feature: pandas.Series
      fc: tensorflow.feature_column.numeric_column
      n_bins: int

    Returns:
      tensorflow.feature_column.bucketized_column
    '''

    qs = list(feature.quantile(np.linspace(0, 1, n_bins+1)))
    return tf.feature_column.bucketized_column(fc, qs)


def input_fn(features, labels, batch_size=1, shuffle=10000,
             repeat=None):

    fdict = {feature: features[feature] for feature in features}

    if labels is not None:
        ds = tf.data.Dataset.from_tensor_slices((fdict, np.array(labels)))
    else:
        ds = tf.data.Dataset.from_tensor_slices(fdict)

    if shuffle:
        return lambda: (ds.shuffle(shuffle).batch(batch_size).repeat(repeat)
                        .make_one_shot_iterator().get_next())
    return lambda: (ds.batch(batch_size).repeat(repeat)
                    .make_one_shot_iterator().get_next())


def train(examples, labels, steps=1000, batch_size=1,
          learning_rate=0.1, hidden_units=[10],
          show_loss=False, model=None,
          optimizer=tf.train.GradientDescentOptimizer):
    '''Write the training function from scratch for practice.'''

    # Find numeric and categorical columns.
    num_cols, cat_cols = num_cat(examples)

    # Create feature columns.
    fcs = [tf.feature_column.numeric_column(feature)
           for feature in num_cols]
    fcs.extend(
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                col, examples[col].unique()))
        for col in cat_cols)

    opt = optimizer(learning_rate)

    if not model:
        if hidden_units:
            model = tf.estimator.DNNClassifier(
                hidden_units, fcs, optimizer=opt)
        else:
            model = tf.estimator.LinearClassifier(fcs, optimizer=opt)

    # Use exponentially increasing period lengths.
    if show_loss:
        period_steps = (np.arange(10) + 1)*steps//10
        for period_step in period_steps:
            try:
                model.train(input_fn(examples, labels,
                                     batch_size=batch_size, shuffle=10000),
                            steps=period_step)
                evals = model.evaluate(input_fn(examples, labels,
                                                shuffle=False), steps=1000)
                print(
                    f"Steps: {evals['global_step']:4}, Loss: {evals['loss']}")
            except KeyboardInterrupt:
                print("Training halted by user.")
                break
    else:
        model.train(input_fn(examples, labels,
                             batch_size=batch_size, shuffle=10000),
                    steps=steps)
        evals = model.evaluate(input_fn(examples, labels,
                                        shuffle=False), steps=1000)
        print(f"Steps: {evals['global_step']:4}, Loss: {evals['loss']}")

    return model


def evaluate(model, features, labels, steps=1000):
    '''Check the mse on the validation set.'''

    results = model.evaluate(input_fn(features, labels, shuffle=False),
                             steps=steps)

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
t_features = train_features
trained_linear = train(t_features, train_labels,
                       optimizer=tf.train.AdagradOptimizer,
                       # model=trained,
                       # hidden_units=[20, 20],
                       # l1_strength=0.5,
                       show_loss=True,
                       learning_rate=1e-1, steps=1000, batch_size=25)
# Remove tf events.
list(map(os.remove,
         glob.glob(os.path.join(
             trained_linear.model_dir, "events.out.tfevents*"))))
print("Evaluated on validation set:")
res = evaluate(trained_linear, validate_features, validate_labels, steps=1000)


# Train a neural net classifier, create word clouds.
trained_nn = None
term_choice = None
for steps in (10, 90, 900):
    trained_nn = train(train_features, train_examples
                       optimizer=tf.train.AdamOptimizer,
                       # model=trained_nn,
                       hidden_units=[20, 20],
                       # dropout=.3,
                       # l1_strength=0.5,
                       show_loss=False,
                       # embedding=2,
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
    test_examples = pd.read_csv("https://archive.ics.uci.edu/"
                                "ml/machine-learning-databases/adult/adult.test",
                                names=columns, sep=r'\s*,\s*', engine='python',
                                skiprows=[0], na_values='?')
    test_examples = test_examples.dropna(how='any', axis=0)

    print("Evaluated on test set:")
    tres = evaluate(trained_nn, test_ds)
