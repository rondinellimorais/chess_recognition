from argparse import ArgumentParser, BooleanOptionalAction
from model import Game

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
  # calibration mapping
  if args['mapping']:
    Game().mapping()

  # start a game
  if args['start']:
    Game().start()


# mock
from model import ChessboardCalibration
import cv2
from utils import (
  imshow
)

chessboard_calibration = ChessboardCalibration(debug=True)
frame = cv2.imread('/Volumes/ROND/chess/dataset/board/real_life/01.jpg')
found, board = chessboard_calibration.loadMapping()
if found:
  img = chessboard_calibration.applyMapping(frame)
  board.state(img)
  board.print()
  imshow('img', img)