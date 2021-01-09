# USAGE
# python solve_sudoku_puzzle.py --model model/digit_classifier_experimental.h5 --image sudoku_puzzle.jpg

# import the necessary packages
from utils import find_puzzle, extract_digit
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from skimage.segmentation import clear_border
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained digit classifier")
ap.add_argument("-i", "--image", required=True,
                help="path to input sudoku puzzle image")
ap.add_argument("-d", "--debug", type=int, default=-1,
                help="whether or not we are visualizing each step of the pipeline")
args = vars(ap.parse_args())

# load the digit classifier from disk
print("[INFO] loading digit classifier...")
model = load_model(args["model"])

# load the input image from disk and resize it
print("[INFO] processing image...")
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = imutils.resize(image, width=600)

# crop the cell from the warped transform image and then
# extract the digit from the cell
cell = image
# digit = extract_digit(cell, debug=args["debug"] > 0)

# # for debugging
# cv2.imshow("Cell/Digit", cell)
# cv2.waitKey(0)

thresh = clear_border(cell)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# if no contours were found than this is an empty cell

thresh = cv2.drawContours(thresh, cnts, -1, 255, -1)

# compute the percentage of masked pixels relative to the total
# area of the image
(h, w) = thresh.shape
percent_filled = cv2.countNonZero(thresh) / float(w * h)


# apply the mask to the thresholded cell
# digit = cv2.bitwise_and(thresh, thresh, mask=mask)
digit = thresh.copy()

# verify that the digit is not empty
if digit is not None:
    foo = np.hstack([cell, digit])

    # for debugging
    # cv2.imshow("Cell/Digit", foo)
    # cv2.waitKey(0)

    # resize the cell to 28x28 pixels and then prepare the
    # cell for classification
    roi = cv2.resize(digit, (28, 28))
    roi = roi.astype("float") / 255.0

    # # for debugging
    # cv2.imshow("ROI", roi)
    # cv2.waitKey(0)

    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    # classify the digit and update the sudoku board with the
    # prediction
    pred_list = model.predict(roi)
    pred = model.predict(roi).argmax(axis=1)[0]
    # if 11 <= pred <= 19:
    #     pred -= 10
    print("predicted digit is {}.".format(pred))
    print(pred_list)
