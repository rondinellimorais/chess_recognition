# mock
from model import ChessboardCalibration
import cv2
from utils import (
  imshow
)

chessboard_calibration = ChessboardCalibration(debug=True)
board = chessboard_calibration.mapping(
  chessboard_img=cv2.imread('/Volumes/ROND/chess/dataset/v2/resized/IMG_0365.jpg'),
  smooth_ksize=(13, 13),
  add_padding=True
)

chessboard_calibration.saveMapping()

frame = cv2.imread('/Volumes/ROND/chess/dataset/v2/resized/IMG_0408.jpg')
found, board = chessboard_calibration.loadMapping()

if found:
  img = chessboard_calibration.applyMapping(frame)
  board.state(img)
  imshow('img', img)