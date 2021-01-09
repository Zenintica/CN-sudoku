# import the necessary packages
import numpy as np
import imutils
import cv2

from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border


def find_puzzle(image, debug=False):
    # convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # apply adaptive thresholding and then invert the threshold map
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    # check to see if we are visualizing each step of the image
    # processing pipeline (in this case, thresholding)
    if debug:
        cv2.imshow("Puzzle Thresh", thresh)
        cv2.waitKey(0)

    # find contours in the thresholded image and sort them by size in
    # descending order
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # initialize a contour that corresponds to the puzzle outline
    puzzle_cnt = None

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we can
        # assume we have found the outline of the puzzle
        if len(approx) == 4:
            puzzle_cnt = approx
            break

    # if the puzzle contour is empty then our script could not find
    # the outline of the sudoku puzzle so raise an error
    if puzzle_cnt is None:
        raise Exception(("Could not find sudoku puzzle outline. "
                         "Try debugging your thresholding and contour steps."))

    # check to see if we are visualizing the outline of the detected
    # sudoku puzzle
    if debug:
        # draw the contour of the puzzle on the image and then display
        # it to our screen for visualization/debugging purposes
        output = image.copy()
        cv2.drawContours(output, [puzzle_cnt], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)

    # apply a four point perspective transform to both the original
    # image and grayscale image to obtain a top-down birds eye view
    # of the puzzle
    puzzle = four_point_transform(image, puzzle_cnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzle_cnt.reshape(4, 2))

    # check to see if we are visualizing the perspective transform
    if debug:
        # show the output warped image (again, for debugging purposes)
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)

    # return a 2-tuple of puzzle in both RGB and grayscale
    return puzzle, warped


def clear_border_pre(img, percent):
    """
    Pre-operations before calling clear_border function of skimage package.
    :param img: the image to be operated.
    :param percent: the cutting percent.
    :return: img: the refined image.
    """
    h, w = img.shape[:2]
    for y in range(0, w):
        for x in range(0, h):
            if y < int(w * percent):
                img[x, y] = 255
            if y > w - int(w * percent):
                img[x, y] = 0
            if x < int(h * percent):
                img[x, y] = 255
            if x > h - int(h * percent):
                img[x, y] = 0
    return img


def extract_digit(cell, debug=False):
    """
    To check whether there is a digit in the given, RGB-channels, not-(28*28) image.
    If so, return a binary, (28*28) image of the given cell;
    Else, return none.
    :param cell: the given, RGB-channels, not-(28*28) image.
    :param debug: the boolean value to decide whether to show some intermediate images.
    :return: digit: a binary, (28*28) image of the given cell (if there exists);
    """

    # apply automatic thresholding to the cell and then clear any
    # connected borders that touch the border of the cell

    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    thresh_pred = clear_border_pre(thresh, percent=0.125)

    thresh_clear = clear_border(thresh_pred)

    # # dilate the poor handwritings
    # kernel = np.ones((3, 3), np.uint8)
    # thresh_clear = cv2.dilate(thresh_clear, kernel, iterations=1)

    # find contours in the thresholded cell
    cnts = cv2.findContours(thresh_clear.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # if no contours were found than this is an empty cell
    if len(cnts) == 0:
        return None

    # otherwise, find all contours in the cell and fill every piece of area within the contour
    thresh_contoured = cv2.drawContours(thresh_clear, cnts, -1, 255, 1)

    # compute the percentage of masked pixels relative to the total area of the image
    (h, w) = thresh_contoured.shape
    percent_filled = cv2.countNonZero(thresh_contoured) / float(w * h)

    # if less than 0.01% of the mask is filled then we are looking at noise and can safely ignore the contour
    if percent_filled < 0.0001:
        return None

    # make digits in the middle of the image
    coordinates = np.nonzero(thresh_contoured)
    xmin = coordinates[0][0]
    xmax = coordinates[0][-1]
    coordinates[1].sort()
    ymin = coordinates[1][0]
    ymax = coordinates[1][-1]
    if (xmax - xmin) * (ymax - ymin) != 0:
        clip = thresh_contoured[xmin: xmax, ymin: ymax]
        digit_raw = np.pad(clip,
                           ((int((thresh_contoured.shape[0] - (xmax - xmin)) / 4),
                             int((thresh_contoured.shape[0] - (xmax - xmin)) / 4)),
                            (int((thresh_contoured.shape[1] - (ymax - ymin)) / 4),
                             int((thresh_contoured.shape[1] - (ymax - ymin)) / 4))),
                           "constant")
    else:
        return None

    digit = digit_raw

    if debug:
        foo_1 = np.hstack([thresh, cv2.resize(thresh_pred, thresh.shape[::-1])])
        foo_2 = np.hstack([cv2.resize(thresh_clear, thresh.shape[::-1]), cv2.resize(digit, thresh.shape[::-1])])
        foo = np.vstack([foo_1, foo_2])
        cv2.imshow("digit", foo)
        cv2.waitKey(0)

    # return the digit to the calling function
    return digit
