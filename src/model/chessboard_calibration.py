import cv2
import numpy as np
import imutils
import json
import os
import sys

from typing import Dict, Tuple
from utils import (
  perspective_transform,
  rotate_image,
  canny_edge,
  hough_line,
  h_v_lines,
  cluster_points,
  line_intersections,
  draw_chessboard_mapping
)
from model.board import Board
from model.debugable import Debugable
from model.square import Square
from dotenv import dotenv_values

class ChessboardCalibration(Debugable):
  """
  Build a chess board calibration
  """
  board: Board
  apply_kdilate: bool
  fix_rotate: bool
  add_padding: bool
  chessboard_img: tuple
  rotate_val: int
  smooth_ksize: tuple

  __out_size: tuple = None
  __padding_val: tuple = (15, 20)
  __matrix: list

  def __init__(self, debug=False):
    super().__init__(debug=debug)
    self.board = Board()
    config = dotenv_values()

    res = int(config.get('RESOLUTION'))
    self.__out_size = (res * 32, res * 32)
    print('frame resolution: {}'.format(self.__out_size))
  
  def mapping(self, chessboard_img=None, fix_rotate=False, rotate_val=-90, add_padding=False, apply_kdilate=True, smooth_ksize=(11, 11)):
    """
    Make a chess board mapping to generate a 8x8 matrix.

    @param chessboard_img A chess board image

    @param add_padding A boolean indicate if image need a padding. Default is `False`

    @param fix_rotate A boolean indicate if image need a rotate. Default is `False`
    
    @param rotate_val The values of rotate if `fix_rotate` is `True`. Default is `-90`
    
    @param apply_kdilate A booelan indicate if image need expand your contours. Default is `True`
    
    @param smooth_ksize A tuple of Gaussian Blur ksize. Default is `(11, 11)`
    """
    self.chessboard_img = chessboard_img
    self.add_padding = add_padding
    self.fix_rotate = fix_rotate
    self.rotate_val = rotate_val
    self.apply_kdilate = apply_kdilate
    self.smooth_ksize = smooth_ksize

    if self.chessboard_img is not None:
      self.save('1_raw.jpg', self.chessboard_img)
    else:
      raise Exception('chess board image cannot be null')

    # get chess board playable area. Playable area is a image of board and only the 64
    # squares clipping borders, backgrounds and something else that
    # make bug on the squares recognitions
    pa_image = self.playableArea(self.chessboard_img)

    # get chess board squares corners
    corners = self.squaresCorners(pa_image)

    # convert corners into a matrix of points
    if corners:
      self.__matrix = self.convertCorners2Matrix(corners)
      self.board.addSquares(self.parseMatrix(self.__matrix))

      # save only works when debug flag is True
      self.save(
        '5_mapping.jpg',
        draw_chessboard_mapping(pa_image, self.__matrix) if self.debug else None
      )

    return self.board

  def playableArea(self, chessboard_image):
    # convert image to gray scale
    gray = cv2.cvtColor(chessboard_image, cv2.COLOR_BGR2GRAY)

    # Blur the image a little. This smooths out the noise a bit and
    # makes extracting the grid lines easier.
    smooth = cv2.GaussianBlur(gray, self.smooth_ksize, 0)

    # The image can have varying illumination levels, the adaptive threshold 
    # calculates a threshold level several small windows in the image.
    # It calculates a mean over a 11x1 window and subtracts 2 from the mean.
    # This is the threshold level for every pixel.
    thresh = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Since we're interested in the borders, and they are black, we invert the image color.
    # Then, the borders of the chessboard are white (along with other noise).
    thresh = cv2.bitwise_not(thresh)

    if self.apply_kdilate:
      kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
      thresh = cv2.dilate(thresh, kernel, iterations=1)

    self.board.contours = self.biggestContours(thresh)
    
    # only works when debug flag is True
    self.drawBiggestContours(self.board.contours, chessboard_image)

    # crop image using biggest contours
    cropped = self.cropImage(chessboard_image,  self.board.contours)

    # only works when debug flag is True
    self.save('3_playable_area.jpg', cropped)

    # crop image with padding to remove chess board borders
    if self.add_padding:
      cropped = self.addPadding(
        cropped,
        self.__padding_val[0],
        self.__padding_val[1],
        self.__out_size
      )
      self.save('3.1_padding.jpg', cropped)

    return cropped
  
  def biggestContours(self, thresh):
    """
    find biggest square area
    """
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse = True)

    peri = cv2.arcLength(cnts[0], True)
    return cv2.approxPolyDP(cnts[0], 0.025 * peri, True)

  def cropImage(self, chessboard_image, biggest_cnt):
    """
    Crop image using contours
    """
    transformed = perspective_transform(chessboard_image.copy(), biggest_cnt)
    
    # check if image need fix a rotate
    if self.fix_rotate:
      transformed = rotate_image(transformed, self.rotate_val)

    return transformed

  def addPadding(self, img, padding_horizontal, padding_vertical, output_image_size):
    """
    Add image padding and return a new cropped image
    """
    h, w = img.shape[:2]
    output_img_h, output_img_w, = output_image_size

    pts1 = np.float32([
      (padding_horizontal, padding_vertical),
      (w - padding_horizontal, padding_vertical),
      (padding_horizontal, h - padding_vertical),
      (w - padding_horizontal, h - padding_vertical)
    ])

    pts2 = np.float32([
      [0, 0],
      [output_img_w, 0],
      [0, output_img_h],
      [output_img_w, output_img_h]
    ])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, output_image_size)

  def squaresCorners(self, pa_image):
    try:
      # based on
      # https://towardsdatascience.com/board-game-image-recognition-using-neural-networks-116fc876dafa
      
      gray = cv2.cvtColor(pa_image, cv2.COLOR_BGR2GRAY)

      # Blur the image a little. This smooths out the noise a bit and
      # makes extracting the grid lines easier.
      gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

      # Canny algorithm
      edges = canny_edge(gray_blur)

      # Hough Transform
      lines = hough_line(edges)

      # Separate the lines into vertical and horizontal lines
      h_lines, v_lines = h_v_lines(lines)

      # Find and cluster the intersecting
      intersection_points = line_intersections(h_lines, v_lines)
      points = cluster_points(intersection_points)

      # only works when debug flag is True
      self.drawChessPoints(points, pa_image)
      
      return points
    except:
      print('[ERROR] Could not get the corners of the board. Try change add_padding parameter.')

  def convertCorners2Matrix(self, corners, thresh_val=10.0):
    """
    Convert chess board squares corners into 9x9 matrix of points.

    matrix:

    [
      [coord0_0, coord0_1, ...., coord0_N]
      [coord1_0, coord1_1, ...., coord1_N]
      [coord2_0, coord2_1, ...., coord2_N]
    ]
    """
    # we will start with first corner
    target_corner = corners[0]

    chess_matrix = []
    while len(corners) != 0:
      # get all corners where axis y between min_thresh and max_thresh
      min_thresh = target_corner[1] - thresh_val
      max_thresh = target_corner[1] + thresh_val
      line_corners = [p for p in corners if p[1] >= min_thresh and p[1] <= max_thresh]
      line_corners.sort(key=lambda p: p[0])

      # add corners of the line into matrix
      if len(line_corners) > 0:
        chess_matrix.append(line_corners)

      # remove finded corners of the main array
      corners = [p for p in corners if p not in line_corners]

      # define a new target corner
      if len(corners) != 0:
        target_corner = corners[0]

    return chess_matrix

  def parseMatrix(self, matrix):
    """
    Parse matrix 9x9 of points into a matrix 8x8 of `Square` object

    matrix:
    
    [
      [Square0_0, Square0_1, ...., Square0_N]
      [Square1_0, Square1_1, ...., Square1_N]
      [Square2_0, Square2_1, ...., Square2_N]
    ]
    """
    if len(matrix) != 9:
      print("[ERROR] No 9x9 dimension matrix found")
      return

    new_matrix = []
    for (row_idx, points) in enumerate(matrix[:len(matrix) - 1]):
      squares = []
      for (col_idx, pt1) in enumerate(points[:len(points) - 1]):
        pt2 = matrix[row_idx + 1][col_idx + 1]

        x1 = round(pt1[0])
        y1 = round(pt1[1])
        x2 = round(pt2[0])
        y2 = round(pt2[1])

        squares.append(Square(x1, y1, x2, y2))
      new_matrix.append(squares)
    return new_matrix

  def applyMapping(self, img):
    """
    Applies the mapping done previously to any image

    @param `img` The image you want to apply the mapping to

    Returns the mapped image
    """

    if img is None:
      raise Exception('img cannot be null')

    if self.board.contours is None:
      raise Exception('Cannot find a contours')

    img = self.cropImage(img, self.board.contours)

    if self.add_padding:
      img = self.addPadding(
        img,
        self.__padding_val[0],
        self.__padding_val[1],
        self.__out_size
      )

    return img

  def __rootDir(self):
    return os.path.join(os.path.dirname(sys.modules['__main__'].__file__), '../')

  def saveMapping(self):
    """
    Save the chess board calibration mapping
    """
    if self.board.contours is None:
      raise Exception('Cannot find a contours')

    if self.board.squares is None:
      raise Exception('Cannot find a squares')

    if len(self.__matrix) == 0:
      raise Exception('No 8x8 matrix found to save')

    dictionary = {
      "fix_rotate": self.fix_rotate,
      "rotate_val": self.rotate_val,
      "add_padding": self.add_padding,
      "matrix": [
        [(float(pt1), float(pt2)) for (pt1, pt2) in row]
        for row in self.__matrix
      ],
      "board": {
        "contours": self.board.contours.tolist(),
        "squares": [
          [s.toJson() for s in row ]
          for row in self.board.squares
        ]
      }
    }

    # save mapping file
    with open(os.path.join(self.__rootDir(), 'chessboard-mapping.json'), 'w') as f:
      f.write('%s' % json.dumps(dictionary))

  def loadMapping(self) -> Tuple[bool, Board]:
    """
    Load the chess board calibration mapping if exists.
    """
    chessboard_mapping_path = os.path.join(self.__rootDir(), 'chessboard-mapping.json')
    if not os.path.exists(chessboard_mapping_path):
      return (False, None)

    json_str = None
    with open(chessboard_mapping_path, 'r') as f:
      json_str = f.read().strip()

    # deserialize json object 
    dictionary = json.loads(json_str)

    self.fix_rotate = dictionary['fix_rotate']
    self.rotate_val = dictionary['rotate_val']
    self.add_padding = dictionary['add_padding']

    self.__matrix = [
      [
        (np.float32(pt1), np.float32(pt2))
        for (pt1, pt2) in row
      ]
      for row in dictionary['matrix']
    ]

    self.board = Board()
    self.board.contours = np.array(dictionary['board']['contours'], dtype=object)
    self.board.addSquares([
      [
        Square(item["x1"], item["y1"], item["x2"], item["y2"], item["color"])
        for item in row
      ]
      for row in dictionary['board']['squares']
    ])

    return (True, self.board)