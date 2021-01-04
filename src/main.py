from model import ChessboardCalibration, Camera, Square, Piece, Darknet
from enumeration import Color
from utils import imshow
import numpy as np
import cv2

#####
# python3 src/main.py --mapping
#
# Captura uma frame da camera e faz o mapeamento, ao final saldo o mapeamento
#####

# get a camera frame
camera = Camera('http://192.168.0.105:4747/video')
frame = camera.capture()

# do calibration
chessboard_calibration = ChessboardCalibration(chessboard_img=frame, debug=True)
board = chessboard_calibration.mapping(
  smooth_ksize=(13, 13),
  add_padding=True
)

camera.destroy()

###########################
#
# with camera
##############

def didCaptureFrame(frame):
  # apply a mapping
  img = chessboard_calibration.applyMapping(frame)

  # show board state
  board.state(img)

  # show image
  cv2.imshow("preview", img)
  if cv2.waitKey(1) & 0xFF == ord('q'):
      camera2.stopRunning()
      cv2.destroyAllWindows()

camera2 = Camera('http://192.168.0.105:4747/video')

# start capture
camera2.startRunning(didCaptureFrame)