import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import regularizers
import matplotlib.pyplot as plt


def train_keras_model(x, y):
    model = Sequential()
    model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(1,)))
    model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1))

    model.compile(optimizer=Adam(), loss='mse')

    # fit the model, keeping 2,000 samples as validation set
    hist = model.fit(x, y,
                     validation_split=0.2,
                     epochs=15000,
                     batch_size=256)
    hist.save(f"Regressor_model")
    return hist

