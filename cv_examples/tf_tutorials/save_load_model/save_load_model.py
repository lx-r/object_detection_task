#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   save_load_model.py
@Time    :   2021/06/14 17:28:20
@Author  :   lx-r 
@Version :   1.0
@Contact :   lixiang-85@foxmail.com
@License :   (C)Copyright
@Desc    :   None
'''

# here put the import lib
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, models
from tensorflow.python.ops.gen_array_ops import transpose

def get_dataset():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]
    print(f"shape of train_images: {train_images.shape}")
    train_images = train_images[:1000].reshape(-1, 28*28)/255.
    test_images = test_images[:1000].reshape(-2, 28*28)/255.
    return train_images, train_labels, test_images, test_labels


def create_model():
    model = keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape = (784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(
        optimizer='adam',
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    return model


def save_model_checkpoints(train_images,
                           train_labels, 
                           test_images, 
                           test_labels,
                           model,
                           checkpoint_path):
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # svae the model weights with call back function, cp abbreviation for checkpoints
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )
    model.fit(
        train_images,
        train_labels,
        epochs=10,
        validation_data=(test_images, test_labels),
        callbacks=[cp_callback]
    )

def save_model_checkpoints_period(train_images, train_labels, test_images, test_labels, model):
    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=5
    )
    model.fit(
        train_images,
        train_labels,
        epochs=30,
        callbacks=[cp_callback],
        validation_data=(test_images, test_labels),
        verbose=0
    )
    return checkpoint_dir



train_images, train_labels, test_images, test_labels = get_dataset()
model = create_model()
model.summary()
checkpoint_path = "training_1/train_1.ckpt"
# save_model_checkpoints(train_images, train_labels, test_images, test_labels, model, checkpoint_path)
checkpoint_dir = save_model_checkpoints_period(train_images, train_labels, test_images, test_labels, model)
latest_weights = tf.train.latest_checkpoint(checkpoint_dir)
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Accuracy of untrained model: {acc*100:5.2f}%")
model.load_weights(latest_weights)
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Accuracy of trained model: {acc*100:5.2f}%")
model.save_weights("./checkpoints/mycheckpoints")
model.load_weights("./checkpoints/mycheckpoints")
# Saved Model方式
model.save("saved_model/model") # save the whole model
new_model = tf.keras.models.load_model("saved_model/model")
loss,acc = new_model.evaluate(test_images, test_labels, verbose=2)
print(f"Accuracy of model with manually saved weights: {acc*100:5.2f}%")
# 使用HDF5标准提供的一种基本保存格式
new_model.save("h5_model/my_model.h5")
h5_model = keras.models.load_model("h5_model/my_model.h5")
h5_model.summary()
loss,acc = new_model.evaluate(test_images, test_labels, verbose=2)
print(f"Accuracy of model with manually saved h5 model: {acc*100:5.2f}%")
