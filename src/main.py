from argparse import ArgumentParser, BooleanOptionalAction
from model import ChessboardCalibration, Camera, Square, Piece, Darknet
from enumeration import Color
from utils import imshow
import numpy as np
import cv2

# define arguments
parser = ArgumentParser()
parser.add_argument("-m", "--mapping",
                    action=BooleanOptionalAction,
                    default=False,
                    help="Starts the mapping of the board")
parser.add_argument("-s", "--start",
                    action=BooleanOptionalAction,
                    default=False,
                    help="Chess game starts")

args = vars(parser.parse_args())

if __name__ == "__main__":
  # start a mapping
  if args['mapping']:
    camera = Camera('http://192.168.0.105:4747/video')
    frame = camera.capture()

    # do calibration
    chessboard_calibration = ChessboardCalibration(chessboard_img=frame, debug=True)

    found, board = chessboard_calibration.loadMapping()
    if not found:
      board = chessboard_calibration.mapping()
      chessboard_calibration.saveMapping()

    camera.destroy()

  # start a game
  if args['start']:
    def didCaptureFrame(frame):
      img = chessboard_calibration.applyMapping(frame)

      board.state(img)

      cv2.imshow("preview", img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          camera.stopRunning() 
          cv2.destroyAllWindows()

    # start capture
    camera = Camera('http://192.168.0.105:4747/video')
    camera.startRunning(didCaptureFrame)
