from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        input_shape = (height, width, depth)
        # C1 Convolutional Layer
        model.add(Conv2D(32, (5, 5), padding="same",
                         input_shape=input_shape))
        model.add(Activation("relu"))
        # S2 Pooling Layer
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # C3 Convolutional Layer
        model.add(Conv2D(64, (5, 5), padding="same"))
        model.add(Activation("relu"))
        # S4 Pooling Layer
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # C5 Fully Connected Convolutional Layer
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Activation("relu"))
        # F6 Fully Connected Layer
        model.add(Dense(84))
        model.add(Activation("relu"))
        # OUTPUT
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        # return the constructed network architecture
        return model
