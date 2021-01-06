# mock
from model import ChessboardCalibration
import cv2
from utils import imshow
import os

frame = cv2.imread('/Volumes/ROND/chess/dataset/v2/resized/IMG_0357.jpg')

chessboard_calibration = ChessboardCalibration(chessboard_img=frame)
found, board = chessboard_calibration.loadMapping()
if not found:
  raise Exception('No mapping found. Run calibration mapping')

img = chessboard_calibration.applyMapping(frame)

# board.state(img)
for (r_idx, row) in enumerate(board.squares):
  for (c_idx, square) in enumerate(row):

    x1 = round(square.x1)
    y1 = round(square.y1)
    x2 = round(square.x2)
    y2 = round(square.y2)

    cropped = img[y1:y2, x1:x2]
    cv2.imwrite('casas/{}_{}.jpg'.format(r_idx, c_idx), cropped)