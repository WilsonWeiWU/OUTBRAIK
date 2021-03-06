# Copyright 2018-2019 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

PLEASE DELETE THIS FILE ONCE YOU START WORKING ON YOUR OWN PROJECT!
"""
# pylint: disable=invalid-name

from typing import Any, Dict


# General:
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import csv
import time
import shutil
import pickle
import logging

# Tensorflow:
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K

# Sklearn
from sklearn.model_selection import KFold

# Scikit-Optimise
from skopt import gp_minimize, dump
from skopt.space import Categorical, Integer
from skopt.utils import use_named_args
#from skopt.plots import plot_convergence

# Statistics:
import scipy
#from uncertainties import unumpy
import itertools
import statistics
itertools.imap = lambda *args, **kwargs: list(map(*args, **kwargs))






def predict(model: np.ndarray, test_x: pd.DataFrame) -> np.ndarray:
    """Node for making predictions given a pre-trained model and a test set.
    """
    X = test_x.values

    # Add bias to the features
    bias = np.ones((X.shape[0], 1))
    X = np.concatenate((bias, X), axis=1)

    # Predict "probabilities" for each class
    result = _sigmoid(np.dot(X, model))

    # Return the index of the class with max probability for all samples
    return np.argmax(result, axis=1)


def report_accuracy(predictions: np.ndarray, test_y: pd.DataFrame) -> None:
    """Node for reporting the accuracy of the predictions performed by the
    previous node. Notice that this function has no outputs, except logging.
    """
    # Get true class index
    target = np.argmax(test_y.values, axis=1)
    # Calculate accuracy of predictions
    accuracy = np.sum(predictions == target) / target.shape[0]
    # Log the accuracy of the model
    log = logging.getLogger(__name__)
    log.info("Model accuracy on test set: %0.2f%%", accuracy * 100)


def _sigmoid(z):
    """A helper sigmoid function used by the training and the scoring nodes."""
    return 1 / (1 + np.exp(-z))






def split_dataset(dataframe, n_splits):
    """Scikit-Learn KFold implementation for pandas DataFrame."""

    label_col = 'target'
    random_state = 2
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    kfolds = []
    global offset_col_name

    for train, validate in kf.split(dataframe):
        training = dataframe.iloc[train]
        train_labels = training[label_col]
        train_set = training.drop(label_col, axis=1)

        validating = dataframe.iloc[validate]
        validate_labels = validating[label_col]
        validate_set = validating.drop(label_col, axis=1)

        kfolds.append(
            [[train_set, validate_set],
             [train_labels, validate_labels]]
        )

    with open('./data/06_models/kfolds.json', "wb") as file:
        pickle.dump(kfolds, file)

    logging.info('Pickled kfolds nested list to JSON.')
    return kfolds



def create_model(num_dense_layers_base, num_dense_nodes_base,
                 num_dense_layers_end, num_dense_nodes_end,
                 activation, adam_b1, adam_b2, adam_eps):
    
    num_input_nodes = 11
    
    # Craete linear stack of layers.
    model = keras.Sequential()

    # Define input layer.
    model.add(keras.layers.Dense(
        num_input_nodes,  # N.umber of nodes
        input_shape=(num_input_nodes,)  # Tuple specifying data input dimensions only needed in first layer.
             ))

    # Define n number of hidden layers (base, i.e. first layers).
    for i in range(num_dense_layers_base):
        model.add(keras.layers.Dense(
            num_dense_nodes_base,
            activation=activation
        ))

    # Define n number of hidden layers (end, i.e. last layers).
    for i in range(num_dense_layers_end):
        model.add(keras.layers.Dense(
            num_dense_nodes_end,
            activation=activation
        ))

    # Add two output nodes.
    model.add(keras.layers.Dense(1, activation=keras.activations.linear))
    model.add(keras.layers.Activation("sigmoid"))

    # Define dam optimiser.
    optimizer = tf.keras.optimizers.Adam(
        lr=0.0001,  # Learning rate
        beta_1=adam_b1,  # Exponential decay rate for the first moment estimates.
        beta_2=adam_b2,  # Exponential decay rate for the second-moment estimates.
        epsilon=adam_eps  # Prevent any division by zero.
    )

    # Compile model.
    model.compile(
        loss='mae',  # Loss function
        optimizer=optimizer,  # Optimisaion function defined above.
        metrics=['mae']  # Metric to be recorded.
    )

    return model



def train_model(fold, fold_num, n_calls, epochs):
    """
    1. Unpack training data.
    2. Define hyper-perameter ranges.
    3. Define early stopping perameters.
    4. Optimise hyper-perameters and save best model.
    5. Save mae per call to CSV.
    """
    logging.info('Training fold {}.'.format(str(fold_num)))
    
    # Retrieve data sets and convert to numpy array.
    train_X = fold[0][0].values.astype(np.float32)
    validate_X = fold[0][1].values.astype(np.float32)
    train_y = fold[1][0].values.astype(np.float32)
    validate_y = fold[1][1].values.astype(np.float32)

    # Define hyper-perameters.
    # Layers
    dim_num_dense_layers_base = Integer(low=1, high=2, name='num_dense_layers_base')
    dim_num_dense_nodes_base = Categorical(categories=list(np.linspace(5, 261, 10, dtype=int)),
                                           name='num_dense_nodes_base')
    dim_num_dense_layers_end = Integer(low=1, high=2, name='num_dense_layers_end')
    dim_num_dense_nodes_end = Categorical(categories=list(np.linspace(5, 261, 10, dtype=int)),
                                          name='num_dense_nodes_end')

    # Optimiser
    dim_adam_b1 = Categorical(categories=list(np.linspace(0.8, 0.99, 11)), name='adam_b1')
    dim_adam_b2 = Categorical(categories=list(np.linspace(0.8, 0.99, 11)), name='adam_b2')
    dim_adam_eps = Categorical(categories=list(np.linspace(0.0001, 0.5, 11)), name='adam_eps')

    dimensions = [dim_num_dense_layers_base, dim_num_dense_nodes_base,
                  dim_num_dense_layers_end, dim_num_dense_nodes_end,
                  dim_adam_b1, dim_adam_b2, dim_adam_eps]

    # Set early stopping variable to prevent overfitting.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        mode='min',  # Monitoring loss
        patience=20,  # Large patience for small batch size
        verbose=0)  # Do not output to terminal

    best_mae = np.inf
    
    # Start hyper-perameter optimisation.
    @use_named_args(dimensions=dimensions)
    def fitness(num_dense_layers_base, num_dense_nodes_base,
                num_dense_layers_end, num_dense_nodes_end,
                adam_b1, adam_b2, adam_eps):

        # Create the neural network with these hyper-parameters.
        model = create_model(num_dense_layers_base=num_dense_layers_base,
                             num_dense_nodes_base=num_dense_nodes_base,
                             num_dense_layers_end=num_dense_layers_end,
                             num_dense_nodes_end=num_dense_nodes_end,
                             activation=tf.keras.activations.relu,
                             adam_b1=adam_b1, adam_b2=adam_b2, adam_eps=adam_eps)

        print('len(train_X.columns)')

        history = model.fit(train_X, train_y, # Training data
                            epochs=epochs,  # Number of forward and backward runs.
                            validation_data=(validate_X, validate_y),  # Validation data
                            verbose=1,
                            callbacks=[early_stopping],  # Prevent overfitting.
                            batch_size=30)  # Increase efficiency

        mae = history.history['val_mae'][-1]
        # If the regressor accuracy of the saved model is improved...
        nonlocal  best_mae
        if mae < best_mae:
            # Save the new model to harddisk.
            model.save('./data/06_models/fold_' + str(fold_num) + '_model.h5')
            # Update the regressor accuracy.
            best_mae = mae

        # Delete the Keras model with these hyper-parameters from memory.
        del model

        # Clear the Keras session, otherwise it will keep adding new
        # models to the same TensorFlow graph each time we create
        # a model with a different set of hyper-parameters.
        K.clear_session()

        # Reset best MAE.
        best_mae = np.inf

        return mae

    # A place for optimiser to start looking.
    default_parameters = [2, 261, 1, 61, 0.857, 0.933, 0.20006]

    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='EI',  # Expected Improvement
                                n_calls=n_calls,
                                x0=default_parameters)

    # Save skopt object.
    dump(search_result,
         './data/06_models/fold_' + str(fold_num) +  '_gp_minimize_result.pickle',
         store_objective=False)
    logging.info('Pickled fold {} Scikit-Optimise object.'.format(fold_num))

    logging.info('Fold {} final parameters: {}.'.format(str(fold_num), search_result.x))
    return search_result



def train_DNN(dataframe, n_splits, n_calls, epochs):
    
    kfolds = split_dataset(dataframe, n_splits)
    all_models = [train_model(fold, fold_num+1, n_calls, epochs) for fold_num, fold in enumerate(kfolds)]

    return all_models




def plot_convergence(all_models, n_calls, n_splits):
    
    mae_logger = [[fold_num + 1, x] for fold_num, result in enumerate(all_models) for x in result['func_vals']]
    mae_df = pd.DataFrame(mae_logger, columns=['Fold', 'MAE (kcal/mol)'])
    
    # x values
    x = np.linspace(1, n_calls, n_calls)

    # y values
    mae = [mae_df.loc[mae_df.iloc[:, 0] == fold, 'MAE (kcal/mol)'].cummin()
           for fold in range(1, n_splits + 1)]
    cumm_mae = list(zip(*mae))
    y = [statistics.mean(call) for call in cumm_mae]

    # standard devation
    std = [statistics.stdev(call) for call in cumm_mae]

    # standard devation bounds
    y1 = [i - sd for i, sd in zip(y, std)]
    y2 = [i + sd for i, sd in zip(y, std)]

    # plot mean line
    fig, ax = plt.subplots(figsize=[8, 6])
    for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(2)

    ax.plot(x, y,
            color='green',
            linewidth=2,
            label='Average MAE over {} folds'.format(n_splits))

    # plot standard deviation fill bounds
    ax.fill_between(x, y1, y2,
                    fc='lightsteelblue',
                    ec='lightsteelblue',
                    label='Standard deviation')

    ax.set_xlabel('Number of calls $n$', fontsize=18)
    ax.set_ylabel('MAE / kcal mol$^{-1}$', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)

    ax.legend(fontsize=18)
    plt.tight_layout()
    
    fig.savefig('.\data\08_reporting\convergence_plot.png')
    
    return ax



def model_predict(model_num, test_entry):
    """Load model from HDF5 and return model prediction on a given test_entry."""

    model = tf.keras.models.load_model('./data/06_models/fold_' + str(model_num) + '_model.h5')

    return model.predict(test_entry)


def test_model(test_set, n_splits):
    n_splits = 5
    test_X = test_set.iloc[:, :-1]
    test_y = test_set.iloc[:, -1]
    predict_per_fold = [model_predict(fold_num, test_X.to_numpy()) for fold_num in range(1, n_splits + 1)]
    predict_y = np.average(predict_per_fold, axis=0)

    predict_y = [x[0] for x in predict_y]
    results_dict = {'true': test_y, 'predicted': predict_y}
    df = pd.DataFrame.from_dict(results_dict, orient='columns')
    df.to_csv('/Users/wilsonwu/OUTBRAIK/outbraik/data/08_reporting/predictions.csv')

    return df