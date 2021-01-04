import cv2
import numpy as np
from model.piece import Piece
from model.darknet import Darknet
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
  network: Darknet = Darknet.instance()

  def __init__(self, x1, y1, x2, y2):
    self.x1 = x1
    self.y1 = y1
    self.x2 = x2
    self.y2 = y2

  def piece(self, frame) -> Piece:
    """
    Make a chess piece prediction base on square coordinates
    """    
    x1 = self.x1
    y1 = self.y1
    x2 = self.x2
    y2 = self.y2

    cropped = frame[y1:y2, x1:x2]
    found, class_name, acc = self.network.predict(img=cropped, size=(64, 64), thresh=0.90)

    if found:
      return Piece(class_name, acc)
    return None