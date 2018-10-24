'''Learn about fairness in machine learning.'''

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
from sklearn.metrics import confusion_matrix
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


def examine(examples: pd.DataFrame):
    '''Examine data in dataframe.'''
    for i in range(0, len(examples.columns)):
        print(examples.iloc[:, i:i+1].describe())


def num_cat(df: pd.DataFrame):
    '''Return the numerical and categorical columns.'''
    numeric_cols = list(
        df.columns[np.logical_or.reduce((df.dtypes == "int64",
                                         df.dtypes == "float64"))])

    categorical_cols = list(set(df.columns) - set(numeric_cols))
    return numeric_cols, categorical_cols


def plot_hists(examples: pd.DataFrame):
    '''Plot histograms for categorical and numerical data.'''
    num_cols, cat_cols = num_cat(examples)
    for col in cat_cols:
        plt.figure()
        sbn.countplot(examples[col])
    for col in num_cols:
        examples[[col]].hist()


# plot_hists(train_examples)


# Notes: possible hard cap on age at 90, more men than women by a lot,
# mostly husbands, capital_gain and loss only for 8.4% and 4.7%,
# possible hard cap on hours_per_week at 99, education and
# education_num are redundant, possible hard cap on captial_gain at 99999,
# units of capital_gain and loss unclear (thousand? dollars?)

def preprocess(examples: pd.DataFrame):
    '''Prepocess the dataframe and return features and labels.'''
    features = examples.copy()[list(
        set(columns) - {"income_bracket", "education_num"})]

    numeric_cols, categorical_cols = num_cat(features)

    # Scale numerical columns, use z-score.
    numeric_cols = set(numeric_cols) - {'age'}
    for feature in numeric_cols:
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


def bucketize(feature: pd.DataFrame, fc:
              tf.feature_column.numeric_column, n_bins: int):
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


def make_dataset(features: pd.DataFrame,
                 labels: pd.DataFrame=None):
    '''Create the tf dataset.'''
    fdict = {feature: features[feature] for feature in features}

    if labels is not None:
        ds = tf.data.Dataset.from_tensor_slices((fdict, np.array(labels)))
    else:
        ds = tf.data.Dataset.from_tensor_slices(fdict)

    return ds


def train_fn(ds: tf.data.Dataset, batch_size=1, shuffle=10000,
             repeat: int=None):
    '''Create input function for training, prediction, evaluation.'''

    if shuffle:
        ds = ds.shuffle(shuffle)
    ds = ds.batch(batch_size)
    if repeat != 1:
        ds = ds.repeat(repeat)

    return lambda: ds.make_one_shot_iterator().get_next()


def train_model(examples: pd.DataFrame, labels: pd.DataFrame,
                steps=1000, batch_size=1,
                learning_rate=0.1, hidden_units=None, show_loss=False,
                model=None, eval_steps=100, dropout=None,
                buckets=None, embeddings=None,
                l1_regularization_strength=None,
                optimizer=tf.train.GradientDescentOptimizer):
    '''Write the training function from scratch for practice.'''

    # Find numeric and categorical columns.
    numeric_cols, categorical_cols = num_cat(examples)

    # Create numeric feature columns.
    fcdict = {feature: tf.feature_column.numeric_column(feature)
              for feature in numeric_cols}

    # Bucket the numeric columns if specified in buckets.
    if buckets:
        if set(buckets.keys()) > set(numeric_cols):
            print("Bucket keys must be numeric column names.")
            raise ValueError
        else:
            bucket_features = (
                {feature: bucketize(examples[feature],
                                    fcdict.pop(feature),
                                    buckets[feature])
                 for feature in buckets})
        fcdict.update(bucket_features)

    # Add categorical features.
    fcdict.update(
        {feature:
         tf.feature_column.categorical_column_with_vocabulary_list(
             feature, examples[feature].unique())
         for feature in categorical_cols})

    # Add columns with embeddings.
    if embeddings:
        embedded_features = (
            {feature:
             tf.feature_column.embedding_column(
                 fcdict.pop(feature),
                 embeddings[feature])
             for feature in embeddings})
        fcdict.update(embedded_features)
        categorical_cols = set(categorical_cols) - set(embeddings.keys())

    # Make indicator columns for features that are not embedded.
    indicators = {feature:
                  tf.feature_column.indicator_column(
                      fcdict.pop(feature))
                  for feature in categorical_cols}
    fcdict.update(indicators)

    # Get the feature columns from the dictionary.
    fcs = list(fcdict.values())

    # Create the dataset.
    ds = make_dataset(examples, labels)

    # Make optimizer.
    if l1_regularization_strength:
        opt = optimizer(learning_rate=learning_rate,
                        l1_regularization_strength=l1_regularization_strength)
    opt = tf.contrib.estimator.clip_gradients_by_norm(
        optimizer(learning_rate=learning_rate), 5.0)

    # Initialize model.
    if not model:
        mkwargs = {'feature_columns': fcs,
                   'optimizer': opt}
        if hidden_units:
            model_type = tf.estimator.DNNClassifier
            mkwargs['hidden_units'] = hidden_units
            if dropout:
                mkwargs['dropout'] = dropout
        else:
            model_type = tf.estimator.LinearClassifier

        model = model_type(**mkwargs)

    # Use exponentially increasing period lengths.
    if show_loss:
        for period in range(10):
            try:
                model.train(train_fn(ds, batch_size=batch_size,
                                     shuffle=10000), steps=steps//10)
                # probs, cids = get_predictions(model, ds)
                res = model.evaluate(train_fn(ds, batch_size=1,
                                              shuffle=False), steps=100)
                print(f"Steps: {res['global_step']}, "
                      f"Loss: {res['loss']}")
            except KeyboardInterrupt:
                print("\nTraining halted by user.")
                break
    else:
        model.train(
            train_fn(ds, batch_size=batch_size, shuffle=10000),
            steps=steps)
        evals = model.evaluate(
            train_fn(ds, shuffle=False), steps=eval_steps)
        print(f"Steps: {evals['global_step']:4}, Loss: {evals['loss']}")

    return model


def evaluate(model: tf.estimator.Estimator,
             features: pd.DataFrame,
             labels: pd.DataFrame,
             steps: int=None):
    '''Check the mse on the validation set.'''

    ds = make_dataset(features, labels)

    results = model.evaluate(train_fn(ds, shuffle=False, repeat=1),
                             steps=steps)

    for stat_name, stat_value in results.items():
        print(f"{stat_name:>20} | {stat_value}")

    return results


def get_predictions(model: tf.estimator.Estimator,
                    ds: tf.data.Dataset):
    '''Retrieve predictions from model.'''
    preds = model.predict(train_fn(ds, shuffle=False, repeat=1))
    preds = list(preds)
    probabilities = np.vstack(pred["probabilities"] for pred in preds)
    class_ids = np.hstack(pred["class_ids"] for pred in preds)
    return probabilities, class_ids


def plot_confusion(cm: np.array):
    '''Plot a confusion matrix.'''
    plt.matshow(cm.T)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    xs = np.array([0, 1, 0, 1]) - .15
    ys = np.array([0, 0, 1, 1]) + .03
    ts = [f"TN: {cm[0, 0]:>5}",
          f"FN: {cm[1, 0]:>5}",
          f"FP: {cm[0, 1]:>5}",
          f"TP: {cm[1, 1]:>5}"]
    for x, y, t in zip(xs, ys, ts):
        plt.text(x, y, t, color="r")


def group_confusions(labels: pd.DataFrame,
                     features: pd.DataFrame,
                     class_ids: np.array):
    cm = confusion_matrix(labels, class_ids)
    plot_confusion(cm)
    plt.title('Confusion matrix for all examples')
    for category in ('race', 'gender'):
        for group in features[category].unique():
            mask = (features[category] == group)
            masked_labels = labels[mask]
            masked_class_ids = class_ids[mask]
            cm = confusion_matrix(masked_labels, masked_class_ids)
            plot_confusion(cm)
            plt.title(f'Confusion matrix for {category}: {group}')


# Train a neural net classifier.
tfs = ['marital_status', 'age', 'hours_per_week', 'education',
       'occupation', 'gender', 'race',
       'workclass', 'relationship']
buckets = {'age': 16}
trained_nn = train_model(train_features[tfs], train_labels,
                         optimizer=tf.train.AdamOptimizer,
                         hidden_units=[256, 128, 56],
                         dropout=.2,
                         buckets=buckets,
                         # l1_regularization_strength=0.0001,
                         show_loss=True,
                         # embedding=2,
                         learning_rate=2e-4, steps=1000, batch_size=8)
# Remove tf events.
list(map(os.remove,
         glob.glob(os.path.join(
             trained_nn.model_dir, "events.out.tfevents*"))))

# Validate.
print("Evaluated on validation set:")
res = evaluate(trained_nn, validate_features[tfs], validate_labels)

# Get probabilities on training data.
probabilities, class_ids = get_predictions(
    trained_nn, make_dataset(train_features))

# Show confusion matrices for various subgroups.
group_confusions(train_labels, train_features, class_ids)

will_test = False
if will_test:
    # Get and parse the test data.
    test_examples = pd.read_csv(
        "https://archive.ics.uci.edu/"
        "ml/machine-learning-databases/adult/adult.test",
        names=columns, sep=r'\s*,\s*', engine='python',
        skiprows=[0], na_values='?')
    test_examples = test_examples.dropna(how='any', axis=0)
    test_features, test_labels = preprocess(test_examples)

    print("Evaluated on test set:")
    test_res = evaluate(trained_nn, test_features, test_labels)
    test_probs, test_class = get_predictions(
        trained_nn, make_dataset(test_features))

    group_confusions(test_labels, test_features, )
