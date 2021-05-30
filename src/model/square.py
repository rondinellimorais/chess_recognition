import cv2
import numpy as np
from model.piece import Piece
from enumeration import Color

class Square:
  """
  Create a representation object of a individual chess board square

  **Coordinates**
  
  (x1, y1)
      •-------+
      |  IMG  |
      +-------•
            (x2, y2)
  """

  color: Color
  x1: int
  y1: int
  x2: int
  y2: int
  isEmpty: bool=True
  piece: Piece=None

  def __init__(self, x1, y1, x2, y2, color=Color.UNDEFINED):
    self.x1 = x1
    self.y1 = y1
    self.x2 = x2
    self.y2 = y2
    self.color = color

  def createPiece(self, class_name, accuracy, classID):
    self.piece = Piece(class_name, accuracy, classID)
    self.isEmpty = False

  def toJson(self):
    return {
      "color": self.color.value,
      "x1": self.x1,
      "y1": self.y1,
      "x2": self.x2,
      "y2": self.y2,
    }