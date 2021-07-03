from re import X
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import mod
import matplotlib.pyplot as plt


def mnist_test():
    minist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = minist.load_data()
    x_train, x_test = x_train/255, x_test/255

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=1)
    model.evaluate(x_test, y_test)
    plt.figure("Images")
    plt.subplot(1,2,1)
    plt.imshow(x_train[1], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(x_train[2], cmap='gray')
    plt.title("Images")
    plt.show()

if __name__=="__main__":
    mnist_test()