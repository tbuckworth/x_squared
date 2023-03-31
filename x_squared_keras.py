import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import regularizers
import matplotlib.pyplot as plt

from image_generator import create_frame
from main import generated_x_squared_data


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
    model.save(f"Regressor_model")
    return model


def main():
    low = -50
    high = 50
    # x, y = generated_x_squared_data(low, high, 10000)
    # model = train_keras_model(x, y)
    model = keras.models.load_model("Regressor_model")
    x_test, y_test = generated_x_squared_data(-70, 70, 10000)

    y_pred = model.predict(x_test)

    create_frame(0, x_test, y_test, y_pred, low, high, 15000, out_file="x_squared_keras.png")


def test():
    model = keras.models.load_model("Regressor_model")
    x_in = np.array([50, 60, 70, 100, 1000, 100000])
    y_out = model.predict(x_in)
    grads = np.diag((y_out - y_out[0])[1:] / (x_in - x_in[0])[1:])



if __name__ == "__main__":
    main()
