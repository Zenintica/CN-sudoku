import os
import random
import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array


def load_cn_datasets(root_name, debug=False):

    testing_img = []
    training_img = []

    directories = os.listdir(root_name)
    directories.remove("10")

    for directory in directories:
        digit_path = os.path.join(root_name, directory)
        training_path = os.path.join(digit_path, 'training')
        testing_path = os.path.join(digit_path, 'testing')
        for img in os.listdir(training_path):
            training_img.append([int(directory), os.path.join(training_path, img)])
        random.shuffle(training_img)
        for img in os.listdir(testing_path):
            testing_img.append([int(directory), os.path.join(testing_path, img)])
        random.shuffle(testing_img)

    testing = []
    training = []
    testing_label = []
    training_label = []
    count_training_total = len(training_img)
    count_training_loaded = 0
    count_testing_total = len(testing_img)
    count_testing_loaded = 0

    for img in training_img:
        count_training_loaded += 1
        if count_training_loaded % 500 == 0:
            print("{} training images loaded in CN dataset, {} in total."
                  .format(count_training_loaded, count_training_total))
        training_label.append(np.array([img[0]]))

        image = cv2.imread(img[1])
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        if image_binary.shape != (28, 28):
            image_binary = cv2.resize(image_binary, (28, 28), interpolation=cv2.INTER_CUBIC)
        if debug:
            cv2.imshow("input training image", image_binary)
            cv2.waitKey(0)

        roi = image_binary / 255
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        training.append(roi)

    training = np.vstack(training)
    training_label = np.concatenate(training_label)

    for img in testing_img:
        count_testing_loaded += 1
        if count_testing_loaded % 500 == 0:
            print("{} testing images loaded in CN dataset, {} in total."
                  .format(count_testing_loaded, count_testing_total))
        testing_label.append(np.array([img[0]]))

        image = cv2.imread(img[1])
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        if image_binary.shape != (28, 28):
            image_binary = cv2.resize(image_binary, (28, 28), interpolation=cv2.INTER_CUBIC)

        roi = image_binary / 255
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        testing.append(roi)

    testing = np.vstack(testing)
    testing_label = np.concatenate(testing_label)
    return training, training_label, testing, testing_label
