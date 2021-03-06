# USAGE
# python solve_sudoku_puzzle.py --model [path_model] --image [path_image] --debug [true for n > 0]
import numpy as np
import argparse
import imutils
import cv2

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from utils import find_puzzle, extract_digit, Sudoku

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m",
                "--model",
                required=True,
                help="path to trained digit classifier")
ap.add_argument("-i",
                "--image",
                required=True,
                help="path to input sudoku puzzle image")
ap.add_argument("-d",
                "--debug",
                type=int,
                default=-1,
                help="whether or not we are visualizing each step of the pipeline")
args = vars(ap.parse_args())

# load the digit classifier from disk
print("[INFO] loading digit classifier...")
model = load_model(args["model"])

# load the input image from disk and resize it
print("[INFO] processing image...")
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)

if args["debug"] > 0:
    cv2.imshow("raw image", image)
    cv2.waitKey(0)

# find the puzzle in the image and then
(puzzleImage, warped) = find_puzzle(image, debug=args["debug"] > 0)

if args["debug"] > 0:
    cv2.imshow("warped area", warped)
    cv2.waitKey(0)

# initialize our 9x9 sudoku board
board = np.zeros((9, 9), dtype="int")

# a sudoku puzzle is a 9x9 grid (81 individual cells), so we can infer the location of each cell by
# dividing the warped image into a 9x9 grid
stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9

# initialize a list to store the (x, y)-coordinates of each cell location
cellLocs = []

# loop over the grid locations
predicted_zone = []
for y in range(0, 9):

    # initialize the current list of cell locations
    row = []

    for x in range(0, 9):

        # compute the starting and ending (x, y)-coordinates of the current cell
        startX = x * stepX
        startY = y * stepY
        endX = (x + 1) * stepX
        endY = (y + 1) * stepY

        # add the (x, y)-coordinates to our cell locations list
        row.append((startX, startY, endX, endY))

        # crop the cell from the warped transform image and then extract the digit from the cell
        cell = warped[startY:endY, startX:endX]
        digit = extract_digit(cell, debug=args["debug"] > 0)

        # verify that the digit is not empty
        if digit is not None:
            roi = cv2.resize(digit, (28, 28))
            roi = roi / 255
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # classify the digit and update the sudoku board with the prediction
            # pred = model.predict(roi).argmax(axis=1)[0]
            # zero excluded
            pred = int(model.predict(roi)[0][1:].argmax()+1)

            if args["debug"] > 0:
                print("[DEBUG] count: {}; prediction list: {}; predicted digit: {}"
                      .format(str((x, y)), str(model.predict(roi)), pred))

            board[y, x] = pred

            textX = int((endX - startX) * 0.33)
            textY = int((endY - startY) * -0.2)
            textX += startX
            textY += endY
            predicted_zone.append((textX, textY))

    # add the row to our cell locations
    cellLocs.append(row)

# board = np.array([[6, 0, 0, 0, 0, 0, 7, 0, 0],
#                   [9, 0, 0, 4, 2, 0, 0, 0, 0],
#                   [0, 1, 2, 0, 5, 0, 0, 0, 0],
#                   [5, 0, 0, 8, 0, 9, 0, 0, 0],
#                   [8, 0, 4, 0, 0, 0, 9, 0, 3],
#                   [0, 0, 0, 7, 0, 3, 0, 0, 1],
#                   [0, 0, 0, 0, 7, 0, 2, 8, 0],
#                   [0, 0, 0, 0, 9, 1, 0, 0, 6],
#                   [0, 0, 3, 0, 0, 0, 0, 0, 4]], dtype="int")

# construct a sudoku puzzle from the board
print("[INFO] OCR'd sudoku board:")
puzzle = Sudoku(3, 3, board=board.tolist())

puzzle.show()

if puzzle.is_legal_puzzle():
    # solve the sudoku puzzle
    print("[INFO] solving sudoku puzzle...")
    solution = puzzle.solve()
    solution.show_full()

    if solution.difficulty == 0:

        # loop over the cell locations and board
        for (cellRow, boardRow) in zip(cellLocs, solution.board):

            # loop over individual cell in the row
            for (box, digit) in zip(cellRow, boardRow):

                # unpack the cell coordinates
                startX, startY, endX, endY = box

                # compute the coordinates of where the digit will be drawn on the output puzzle image
                textX = int((endX - startX) * 0.33)
                textY = int((endY - startY) * -0.2)
                textX += startX
                textY += endY

                # draw the result digit on the sudoku puzzle image
                if (textX, textY) in predicted_zone:
                    cv2.putText(puzzleImage, str(digit), (textX, textY),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                else:
                    cv2.putText(puzzleImage, str(digit), (textX, textY),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # show the output image
        cv2.imshow("Sudoku Result", puzzleImage)
        cv2.waitKey(0)
    else:
        print("[INFO] given puzzle has no solution.")
else:
    raise ValueError("[ERROR] given puzzle is illegal.")
