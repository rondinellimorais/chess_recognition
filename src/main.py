import sys
from argparse import ArgumentParser, BooleanOptionalAction
from model import Game
from pyqtgraph.Qt import QtGui

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
    app = QtGui.QApplication(sys.argv)
    game = Game()
    game.mapping()

  # start a game
  if args['start']:
    app = QtGui.QApplication(sys.argv)
    game = Game()
    game.start()
    sys.exit(app.exec_())