import os
import logging
import joblib
import numpy as np
import pandas as pd

from functools import partial

from tensorflow.python.keras.layers import Dense, Dropout, Activation
from scikeras.wrappers import KerasRegressor

import tensorflow as tf

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

__all__ = ['mean_absolute_percentage_error', 'StackedModel']

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate mean absolute percentage error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))*100

class StackedModel(BaseEstimator, RegressorMixin):
    """
    Ensemble model
    """

    def __init__(self, seed:int, epochs:int, batch_sizes = (128,64,32),
                 reg_learning_rate=0.001, loss="mean_absolute_error",patience=10,
                 min_delta=1e-3):

        self.verbose = 1
        self.fitted = False
        self.model_name = "stacked_model"
        self.seed = seed
        self.scaler = self._get_scaler

        self.epochs = epochs
        self.batch_sizes = batch_sizes
        self.learning_rate = reg_learning_rate
        self.loss = loss
        self.patience = patience
        self.min_delta = min_delta
        self.reg_earlystoppping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                                   mode="min",
                                                                   min_delta = self.min_delta,
                                                                   patience = self.patience)

    def _build_tf_refressor(self, reg_params: dict = {}, *args, **kwargs):
        tf_model = partial(
            self._deep_nn_build,
            input_dim = reg_params["input_dim"],
            loss = reg_params["loss"],
            seed = reg_params["random_state"],
            learning_rate = reg_params["learning_rate"]
        )

        return KerasRegressor(
            build_fn=tf_model,
            epochs=reg_params["epochs"],
            batch_size=reg_params["batch_size"],
            verbose = reg_params["verbose"]
        )

    @staticmethod
    def _deep_nn_build(input_dim, seed, loss, learning_rate=0.001):
        model = tf.keras.Sequential()

        model.add(Dense(256, input_dim=input_dim,
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(128, input_dim=input_dim,
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.5))


        model.add(Dense(64, input_dim=input_dim,
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.5))


        model.add(Dense(32, input_dim=input_dim,
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(1, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed), activation="linear"))
        model.compile(
            loss=loss, optimizer = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.98, beta_2=0.99, epsilon=None, decay=0.0,amsgrad=False)
        )

        return model