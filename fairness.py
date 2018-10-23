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


def examine(examples):
    '''Examine data in dataframe.'''
    for i in range(0, len(examples.columns)):
        print(examples.iloc[:, i:i+1].describe())


def num_cat(df):
    '''Return the numerical and categorical columns.'''
    num_cols = df.columns[np.logical_or.reduce((df.dtypes == "int64",
                                                df.dtypes == "float64",
                                                df.dtypes == "bool"))]

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


def make_dataset(features, labels):
    '''Create the tf dataset.'''
    fdict = {feature: features[feature] for feature in features}

    if labels is not None:
        ds = tf.data.Dataset.from_tensor_slices((fdict, np.array(labels)))
    else:
        ds = tf.data.Dataset.from_tensor_slices(fdict)

    return ds


def train_fn(ds, batch_size=1, shuffle=10000, repeat=None):
    '''Create input function for training, prediction, evaluation.'''
    if shuffle:
        return lambda: (ds.shuffle(shuffle).batch(batch_size).repeat(repeat)
                        .make_one_shot_iterator().get_next())
    return lambda: (ds.batch(batch_size).repeat(repeat)
                    .make_one_shot_iterator().get_next())


def train_model(examples, labels, steps=1000, batch_size=1,
                learning_rate=0.1, hidden_units=[10], show_loss=False,
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
                res = model.evaluate(train_fn(ds, batch_size=1,
                                              shuffle=False), steps=100)
                print(f"Steps: {res['global_step']}, "
                      f"Loss: {res['loss']}")
            except KeyboardInterrupt:
                print("\nTraining halted by user.")
                break
    else:
        model.train(train_fn(ds,
                             batch_size=batch_size, shuffle=10000),
                    steps=steps)
        evals = model.evaluate(
            train_fn(ds, shuffle=False), steps=eval_steps)
        print(f"Steps: {evals['global_step']:4}, Loss: {evals['loss']}")

    return model


def evaluate(model, features, labels, steps=1000):
    '''Check the mse on the validation set.'''

    ds = make_dataset(features, labels)

    results = model.evaluate(train_fn(ds, shuffle=False),
                             steps=steps)

    for stat_name, stat_value in results.items():
        print(f"{stat_name:>20} | {stat_value}")

    return results


def get_predictions(model, features, labels):
    '''Retrieve predictions from model.'''
    ds = make_dataset(features, labels)
    preds = model.predict(train_fn(ds, shuffle=False))
    preds = list(preds)
    probabilities = np.vstack(pred["probabilities"] for pred in preds)
    class_ids = np.hstack(pred["class_ids"] for pred in preds)
    return probabilities, class_ids


# Train a linear classifier.
tfs = ['marital_status', 'age', 'hours_per_week', 'education',
       'occupation', 'gender', 'capital_gain', 'capital_loss', 'race',
       'workclass', 'relationship']
trained_linear = train_model(train_features[tfs], train_labels,
                             optimizer=tf.train.AdagradOptimizer,
                             # model=trained,
                             # hidden_units=[20, 20],
                             # l1_strength=0.5,
                             show_loss=True,
                             learning_rate=1e-1, steps=1000, batch_size=1)
# Remove tf events.
list(map(os.remove,
         glob.glob(os.path.join(
             trained_linear.model_dir, "events.out.tfevents*"))))

# Validate.
print("Evaluated on validation set:")
res = evaluate(trained_linear, validate_features, validate_labels, steps=1000)


# Train a neural net classifier.
tfs = ['marital_status', 'age', 'hours_per_week', 'education',
       'occupation', 'gender', 'capital_gain', 'capital_loss', 'race',
       'workclass', 'relationship']
trained_nn = train_model(train_features[tfs], train_labels,
                         optimizer=tf.train.AdamOptimizer,
                         hidden_units=[1024, 512],
                         dropout=.1,
                         # buckets=buckets,
                         # l1_strength=0.5,
                         show_loss=True,
                         # embedding=2,
                         learning_rate=3e-3, steps=1000, batch_size=50)
# Remove tf events.
list(map(os.remove,
         glob.glob(os.path.join(
             trained_nn.model_dir, "events.out.tfevents*"))))

# Validate.
print("Evaluated on validation set:")
res = evaluate(trained_nn, validate_features, validate_labels, steps=1000)

# Get predictions for training data
probabilities, class_ids = get_predictions(
    trained_nn, train_features, train_labels)


for category in ('race', 'gender'):
    for group in train_features[category]:
        mask = (train_features[category] == group)
        masked_features = train_features[mask]
        masked_labels = train_labels[mask]
        masked_probabilities = probabilities[mask]
        cm = confusion_matrix(masked_labels, masked_probabilities)
        plt.matshow(cm)
        plt.title('Confusion matrix for {category}: {group}')


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
    tres = evaluate(trained_nn, test_features, test_labels)
