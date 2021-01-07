# mock
from model import ChessboardCalibration
import cv2
from utils import (
  imshow,
  imshow
)

chessboard_calibration = ChessboardCalibration(debug=True)
board = chessboard_calibration.mapping(
  chessboard_img=cv2.imread('/Volumes/ROND/chess/dataset/v2/resized/IMG_0365.jpg'),
  smooth_ksize=(13, 13),
  add_padding=True
)

frame = cv2.imread('/Volumes/ROND/chess/dataset/v2/resized/IMG_0408.jpg')
img = chessboard_calibration.applyMapping(frame)

board.state(img)

# board.state(img)
for (r_idx, row) in enumerate(board.squares):
  for (c_idx, square) in enumerate(row):

    x1 = round(square.x1)
    y1 = round(square.y1)
    x2 = round(square.x2)
    y2 = round(square.y2)

    cropped = img[y1:y2, x1:x2]
    cv2.imwrite('casas/{}_{}.jpg'.format(r_idx, c_idx), cropped)