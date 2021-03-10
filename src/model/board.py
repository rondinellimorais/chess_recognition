from model.square import Square
from model.darknet import Darknet
from enumeration import Color
from utils import intersect_area
import numpy as np

class Board:
  contours: list = None
  squares: list = None
  network: Darknet = Darknet.instance()

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

  def state(self, img):
    # previews all image classes
    detections = self.network.predict(img=img, size=(416, 416), thresh=0.8, draw_and_save=True)

    # check if the boxes intersect with any square
    for (name, bounding_box, accuracy, _) in detections:
      old_area = 0
      for row in self.squares:
        for square in row:
          area = intersect_area(
            np.array([square.x1, square.y1, square.x2, square.y2], dtype=object),
            np.array(bounding_box, dtype=object)
          )

          if area is not None and area > old_area:
            square.createPiece(name, accuracy)

    return self.squares

  def print(self):
    print('\n==================')
    if self.squares is not None:
      for (r_idx, row) in enumerate(self.squares):
        for (c_idx, square) in enumerate(row):
          if not square.isEmpty:
            print('[{}, {}] {} | {} | {:.2f}%'.format(r_idx, c_idx, square.piece.color, square.piece.name, square.piece.acc))
          else:
            print('[{}, {}] Empty'.format(r_idx, c_idx))