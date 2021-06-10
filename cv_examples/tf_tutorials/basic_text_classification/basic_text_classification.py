#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   basic_text_classification.py
@Time    :   2021/06/09 21:58:49
@Author  :   lx-r 
@Version :   1.0
@Contact :   lixiang-85@foxmail.com
@License :   (C)Copyright
@Desc    :   None
'''

# here put the import lib
from functools import partial
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.python.ops.gen_math_ops import mod
import matplotlib.pyplot as plt

print(f"tensorflow.__version__: {tf.__version__}")


def get_reverse_word_index(imdb):
    word_index = imdb.get_word_index()
    word_index = {k:v+3 for k,v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([val, key] for (key, val) in word_index.items())
    return reverse_word_index, word_index


def decode_review(txt, reverse_word_index):
    return ' '.join([ reverse_word_index.get(i, '?') for i in txt ])


def get_history_dict():
    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    print(f"size of train_data: {len(train_data)}, size of test_data: {len(test_labels)}")
    reverse_word_index, word_index = get_reverse_word_index(imdb)
    train_data = keras.preprocessing.sequence.pad_sequences(
        train_data,
        value = word_index["<PAD>"],
        padding = "post",
        maxlen = 256 
    )
    test_data = keras.preprocessing.sequence.pad_sequences(
        test_data,
        value = word_index["<PAD>"],
        padding = "post",
        maxlen = 256
    )

    vocab_size = 10000
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(
        optimizer = 'adam',
        loss = "binary_crossentropy",
        metrics = ['accuracy'] 
    )
    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]
    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    history = model.fit(
        partial_x_train,
        partial_y_train,
        epochs=40,
        batch_size=512,
        validation_data=(x_val, y_val),
        verbose=1
    )

    results = model.evaluate(test_data, test_labels, verbose=2)
    print(results)
    history_dict = history.history
    history_dict.keys()
    return history_dict


def plot_training_process(history_dict):
    plt.figure(24)
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]
    epoches = range(1, len(acc)+1)
    plt.subplot(121)
    plt.plot(epoches, loss, 'bo', label="train loss")
    plt.plot(epoches, val_loss, 'b', label="validation loss")
    plt.title("train and validation loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    
    plt.subplot(122)
    plt.plot(epoches, acc, 'bo', label="training acc")
    plt.plot(epoches, val_acc, 'b', label="validation acc")
    plt.title("train and validation acc")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.legend

    plt.show()


if __name__=="__main__":
    history_dict = get_history_dict()
    plot_training_process(history_dict)