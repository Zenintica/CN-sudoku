# coding=utf-8
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Model
import numpy as np


class AlexNet(Model):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv = Sequential([
            # unit1 [b,28,28,1] => [b,14,14,16]
            Conv2D(16, (3, 3), padding='same', strides=1, activation=tf.nn.relu),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            BatchNormalization(),  # 使用bn层代替LRN

            # unit2 [b,14,14,16] => [b,7,7,32]
            layers.Conv2D(32, (3, 3), padding='same', strides=1, activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
            layers.BatchNormalization(),

            # unit3 [b,7,7,32] => [b,7,7,64]
            layers.Conv2D(64, (3, 3), padding='same', strides=1, activation=tf.nn.relu),

            # unit4 [b,7,7,64] => [b,7,7,128]
            layers.Conv2D(128, (3, 3), padding='same', strides=1, activation=tf.nn.relu),

            # unit5 [b,7,7,128] => [b,4,4,256]
            layers.Conv2D(256, (3, 3), padding='same', strides=1, activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
            layers.BatchNormalization(),

        ])

        self.fc = Sequential([
            # fc1
            layers.Dense(4096, activation=tf.nn.relu),
            layers.Dropout(0.4),
            # fc2
            layers.Dense(2048, activation=tf.nn.relu),
            layers.Dropout(0.4),
            # fc3
            layers.Dense(1024, activation=tf.nn.relu),
            layers.Dropout(0.4),
            # fc4
            layers.Dense(10, activation=tf.nn.relu)
        ])

    def call(self, inputs, training=None):
        x = inputs
        out = self.conv(x)
        out = np.reshape(out, (-1, 4*4*256))
        out = self.fc(out)
        return out

model = AlexNet()


# seed = 7
# np.random.seed(seed)
#
# model = Sequential()
# model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=(227, 227, 3), padding='valid', activation='relu',
#                  kernel_initializer='uniform'))
# model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# model.add(Flatten())
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1000, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
