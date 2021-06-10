#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   basic_image_classification.py
@Time    :   2021/06/10 23:24:08
@Author  :   lx-r 
@Version :   1.0
@Contact :   lixiang-85@foxmail.com
@License :   (C)Copyright
@Desc    :   None
'''

# here put the import lib
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.engine.training import Model


print(tf.__version__)


def get_dataset(class_names, show_detail=False):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print(f"train_images.shape: {train_images.shape}, train_labels.shape: {train_labels.shape}")
    print(f"labels content: {set(train_labels)}")
    train_images = train_images/255.
    test_images = test_images/255.
    if show_detail:
        plt.figure()
        plt.imshow(train_images[0], cmap='gray')
        plt.colorbar()
        plt.grid(False)
        # plt.show()

        plt.figure(figsize=(10,8))
        for i in range(25):
            plt.subplot(5, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.xlabel(class_names[train_labels[i]])

        plt.show()
    return train_images, train_labels, test_images, test_labels


def get_trained_model(train_images, train_labels, test_images, test_labels):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10),
        keras.layers.Softmax()
    ])

    model.compile(optimizer = 'adam',
                loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                metrics = ['accuracy'])

    model.fit(train_images, train_labels, epochs=10)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Test Accuracy: {test_acc}")
    return model


def plot_image(i, predictions_array, true_label, img, class_names):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(
        class_names[predicted_label],
        100*np.max(predictions_array),
        class_names[true_label],
        color = color
    ))


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def num_plot_predictions(num_rows, num_cols, predictions, test_images, test_labels, class_names):
    num_images = num_cols*num_rows
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], test_labels, test_images, class_names)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_images, train_labels, test_images, test_labels = get_dataset(class_names)
    model = get_trained_model(train_images, train_labels, test_images, test_labels)
    predictions = model.predict(test_images)
    print(f"predictions[0]: { np.argmax(predictions[0]) }")
    num_plot_predictions(5, 3, predictions, test_images, test_labels, class_names)

