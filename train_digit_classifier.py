# USAGE
# python train_digit_classifier.py --epoch [num_epoch] --net [name_net] --version [number] --debug [true for n > 0]
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from utils import load_cn_datasets
from net import SudokuNet, LeNet, lr_schedule, resnet_v1

# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-e",
                "--epoch",
                required=True,
                help="number of epoches")
ap.add_argument("-n",
                "--net",
                required=True,
                help="type of network")
ap.add_argument("-v",
                "--version",
                required=True,
                help="version info")
ap.add_argument("-d",
                "--debug",
                type=int,
                required=True,
                help="debug mode")
args = vars(ap.parse_args())

# Initialize the initial learning rate, number of epochs to train.
INIT_LR = 1e-3
EPOCHS = int(args["epoch"])
BS = 128

# Load the MNIST dataset. Normalization not performed yet (in line 35 - line 37).
print("[INFO] accessing MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# Add a channel (i.e., grayscale) dimension to the digits.
trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

# Scale vanilla MNIST data to the range of [0, 1]
trainData = trainData / 255
testData = testData / 255

# Add CN dataset (Normalization already performed).
print("[INFO] accessing CN dataset...")
if args["debug"] > 0:
    debug_inner_flag = True
else:
    debug_inner_flag = False
trainData_CN, trainLabels_CN, testData_CN, testLabels_CN = load_cn_datasets("EI339_CN", debug=debug_inner_flag)

# Concatenate vanilla MNIST data and CN datasets.
trainData = np.vstack([trainData, trainData_CN])
testData = np.vstack([testData, testData_CN])
trainLabels = np.concatenate([trainLabels, trainLabels_CN])
testLabels = np.concatenate([testLabels, testLabels_CN])

# Convert the labels from integers to vectors. Transform the digital labels to one-hot vectors.
le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.transform(testLabels)

# initialize the optimizer and model
print("[INFO] compiling model...")

if args["net"] == "sudokunet":
    # Implementation of SudokuNet
    opt = Adam(lr=INIT_LR)
    model = SudokuNet.build(width=28, height=28, depth=1, classes=10)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])

elif args["net"] == "lenet":
    # Implementation of LeNet
    model = LeNet.build(width=28, height=28, depth=1, classes=10)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=["accuracy"])

elif args["net"][0:6] == "resnet":
    # Implementation of resnet
    n = int(args["net"][7])
    if n < 3 or n > 8:
        raise ValueError("resnet arguments wrong. please set 3 <= n <= 8 and depth = 6n+2. ")
    depth = n * 6 + 2
    input_shape = trainData.shape[1:]
    model = resnet_v1(input_shape=input_shape, depth=depth, num_classes=10)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])
    model.summary()

else:
    raise NotImplementedError("Wrong network argument. Supported: \"sudokunet\", \"lenet\", \"resnet\".")

# train the network
print("[INFO] training network...")
H = model.fit(
    trainData, trainLabels,
    validation_data=(testData, testLabels),
    batch_size=BS,
    epochs=EPOCHS,
    verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testData)
print(classification_report(
    testLabels.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=[str(x) for x in le.classes_]))

# serialize the model to disk
print("[INFO] serializing digit model...")
model.save("./model/{}_epoch={}_ver={}.h5".format(args["net"], args["epoch"], args["version"]), save_format="h5")

print("[INFO] visualizing training and validating results...")
x_axis = np.arange(1, EPOCHS + 1)
plt.title('Result Analysis: Accuracy')
plt.plot(x_axis, H.history["accuracy"], color='red', label='training accuracy')
plt.plot(x_axis, H.history["val_accuracy"], color='blue', label='testing accuracy')
plt.legend()
plt.xlabel('epoches')
plt.ylabel('rate')
plt.show()

plt.title('Result Analysis: Losses')
plt.plot(x_axis, H.history["loss"], color='green', label='losses')
plt.plot(x_axis, H.history["val_loss"], color='yellow', label='val_losses')
plt.legend()
plt.xlabel('epoches')
plt.ylabel('rate')
plt.show()
