import cv2
import os
from utils import random_color

class Debugable:
  def __init__(self, debug=False):
    self.debug = debug
    if self.debug:
      self.default_path = os.path.join(f"{os.environ['PWD']}/debug")
      if not os.path.exists(self.default_path):
        os.mkdir(self.default_path)

  def save(self, filename, img):
    if self.debug and len(img) > 0:
      print(os.path.join(self.default_path, filename))
      cv2.imwrite(os.path.join(self.default_path, filename), img)

  def drawChessPoints(self, points, img):
    if self.debug:
      img = img.copy()
      for (idx, point) in enumerate(points):
        color = random_color()
        cv2.putText(img, str(idx), point, cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
        cv2.circle(img, point, 3, color, -1)
      self.save('4_squares_corners.jpg', img)

  def drawBiggestContours(self, biggest_cnt, img):
    if self.debug:
      imagedrawed = img.copy()
      imagedrawed = cv2.drawContours(imagedrawed, [biggest_cnt], -1, random_color(), 2)
      self.save('2_biggest_cnt.jpg', imagedrawed)
