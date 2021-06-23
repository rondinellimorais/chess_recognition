import numpy as np

from typing import Dict
from model.square import Square
from model.darknet import Darknet
from enumeration import Color
from utils import intersect_area
from shapely.geometry import box
from dotenv import dotenv_values

class Board:
  contours: list = None
  squares: list[Square] = None
  network: Darknet = Darknet.instance()
  __squaresAverage: int
  __config: Dict = None

  def __init__(self) -> None:
    self.__config = dotenv_values()

  def colorBuild(self):
    """
    Build the color of the squares
    """
    if self.squares != None:
      curr_color = Color.UNDEFINED
      for (row_idx, row) in enumerate(self.squares):
        curr_color = Color.WHITE if (row_idx % 2) == 0 else Color.BLACK
        for (col_idx, square) in enumerate(row):
          square.color = curr_color
          curr_color = Color.BLACK if curr_color == Color.WHITE else Color.WHITE
  
  def addSquares(self, squares=None):
    """
    Add a list of squares into a board
    """
    if squares is None:
      raise Exception('List of squares cannot be null')

    if len(squares) == 0:
      raise Exception('List of squares cannot be empty')

    self.squares = squares
    self.colorBuild()
    self.computeSquaresAverage()

  def computeSquaresAverage(self):
    """
    Compute the average of squares
    """
    square_areas = []
    for row in self.squares:
      for square in row:
        square_areas.append(box(square.x1, square.y1, square.x2, square.y2).area)

    self.__squaresAverage = np.average(square_areas)
    print('Squares Average......: {}'.format(self.__squaresAverage))

  def scan(self, img):
    """
    Scan chess board using computer vision

    @return
    A list of `Square` with position of each piece on image
    """
    # previews all image classes
    thresh = float(self.__config.get('PREDICTION_THRESHOLD'))
    res = int(self.__config.get('RESOLUTION'))
    detections = self.network.predict(img=img, size=(res * 32, res * 32), thresh=thresh)

    self.resetState()

    # The intersection area must be greater
    # than 25% of the average square area
    minArea = self.__squaresAverage * 0.25

    # check if the boxes intersect with any square
    for (name, bounding_box, accuracy, classID) in detections:
      old_area = 0
      for row in self.squares:
        for square in row:
          area = intersect_area(
            np.array([square.x1, square.y1, square.x2, square.y2], dtype=object),
            np.array(bounding_box, dtype=object)
          )

          if area is not None and area > minArea and area > old_area:
            square.createPiece(name, accuracy, classID)

    return (self.squares, detections)

  def resetState(self):
    """
    Reset board state
    """
    for (r_idx, row) in enumerate(self.squares):
      for (c_idx, square) in enumerate(row):
        new_square = Square(
          x1=square.x1,
          y1=square.y1,
          x2=square.x2,
          y2=square.y2,
          color=square.color
        )
        self.squares[r_idx][c_idx] = new_square

  def print(self):
    print('\n==================')
    if self.squares is not None:
      for (r_idx, row) in enumerate(self.squares):
        for (c_idx, square) in enumerate(row):
          if not square.isEmpty:
            print('[{}, {}] {} | {} | {:.2f}%'.format(r_idx, c_idx, square.piece.color, square.piece.name, square.piece.acc))
          else:
            print('[{}, {}] Empty'.format(r_idx, c_idx))

  def toMatrix(self, squares: list[Square]) -> np.array:
    """
    parse a list of `Square` object to a `numpy` matrix
    """
    piece_map = np.ones(64) * -1
    board_state = piece_map.reshape((8, 8)).astype(np.int32)

    if squares is not None:
      for (r_idx, row) in enumerate(squares):
        for (c_idx, square) in enumerate(row):
          if not square.isEmpty:
            board_state[r_idx][c_idx] = square.piece.classID
    
    return board_state