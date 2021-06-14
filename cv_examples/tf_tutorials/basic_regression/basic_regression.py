#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   basic_regression.py
@Time    :   2021/06/12 21:49:39
@Author  :   lx-r 
@Version :   1.0
@Contact :   lixiang-85@foxmail.com
@License :   (C)Copyright
@Desc    :   None
'''

# here put the import lib
import pathlib
import matplotlib.pyplot as plt
from pandas.core.algorithms import mode
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

def norm(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']

def get_dataset():
    dataset_path = keras.utils.get_file("auto-msg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                    'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                                na_values="?", comment="\t",
                                sep=" ", skipinitialspace=True)
    dataset = raw_dataset.copy()
    print(dataset.tail())
    # data contains undeclared value, filter these data
    print(dataset.isna().sum())
    dataset = dataset.dropna()
    # Encode Origin cloumn to One-Hot format
    origin = dataset.pop("Origin")
    dataset["USA"] = (origin == 1)*1.0
    dataset["Europe"] = (origin == 2)*1.0
    dataset["Japan"] = (origin == 3)*1.0
    print(dataset.tail())
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]],
                diag_kind="kde")
    train_labels = train_dataset.pop("MPG")
    test_labels = test_dataset.pop("MPG")
    return train_dataset, train_labels, test_dataset, test_labels


def build_model(train_dataset):
    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", 
        input_shape=[len(train_dataset.keys())]),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1)
    ])
    optimizer = keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


# example_batch = normed_train_data[:10]
# example_result = model.predict(example_batch)

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print("")
        print(".", end="")

EPOCHS = 1000
# model = build_model()
# print(model.summary())
# history = model.fit(
#     normed_train_data, train_labels,
#     epochs = EPOCHS, validation_split = 0.2, verbose = 0,
#     callbacks = [ PrintDot() ])

# hist = pd.DataFrame(history.history)
# hist["epoch"] = history.epoch
# print(hist.tail())

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Abs Error [MPG]")
    plt.plot(hist["epoch"], hist["mae"], label="Train Error")
    plt.plot(hist["epoch"], hist["val_mae"], label="Val Error")
    plt.ylim([0, 5])
    plt.legend()
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Square Error [ $MPG^2$ ]")
    plt.plot(hist["epoch"], hist["mse"], label="Train Error")
    plt.plot(hist["epoch"], hist["val_mse"], label = "Val Error")
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


def plot_prediction_error(test_labels, test_predictions):
    plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    error = test_predictions - test_labels
    plt.subplot(1,2,2)
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")
    plt.savefig("PredictionResult.png", dpi=90)
    plt.show()


if __name__ == '__main__':
    train_dataset, train_labels, test_dataset, test_labels = get_dataset()
    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()
    normed_train_data = norm(train_dataset, train_stats)
    normed_test_data = norm(test_dataset, train_stats)
    model = build_model(train_dataset)
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10
    )
    history = model.fit(
        normed_train_data, train_labels, epochs = EPOCHS,
        validation_split = 0.2, verbose = 0, callbacks = [early_stop, PrintDot()])
    plot_history(history)
    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
    test_predictions = model.predict(normed_test_data).flatten()
    plot_prediction_error(test_labels, test_predictions)
