from typing import Dict, Tuple

from pyqtgraph.graphicsItems.ScatterPlotItem import Symbols
from model.chessboard_calibration import ChessboardCalibration
from model.board import Board
from model.agent import Agent
from model.camera import Camera
from model.gui import GUI
from dotenv import dotenv_values
from PIL import Image
from io import BytesIO
from pyqtgraph.Qt import QtCore, QtGui

import cv2
import chess
import numpy as np
import chess.svg
import cairosvg
import time

import imutils

# indexes do arquivos darknet.names
SYMBOLS = {
  0: "♙",
  1: "♖",
  2: "♗",
  3: "♘",
  4: "♔",
  5: "♕",
  6: "♟",
  7: "♜",
  8: "♝",
  9: "♚",
  10: "♛",
  11: "♞"
}

# colors constants
COLORS = np.random.randint(0, 255, size=(len(SYMBOLS), 3), dtype="uint8")

class Game(GUI):
  __cam_address: str
  __running_calibration: ChessboardCalibration
  __board: Board
  __config: Dict
  __agent: Agent
  __camera: Camera = None
  __fps: float
  __lastupdate: float

  def __init__(self, **kwargs):
    super(Game, self).__init__(**kwargs)
    self.__config = dotenv_values()
    self.__cam_address = self.__config.get('CAM_ADDRESS')
    self.__agent = Agent()

    # frame rate metrics
    self.__fps = 0.
    self.__lastupdate = time.time()

  def mapping(self):
    """
    Start mapping chess board
    """
    camera = Camera(self.__cam_address)
    frame = camera.capture()

    # do calibration mapping
    chessboard_calibration = ChessboardCalibration(debug=True)
    chessboard_calibration.mapping(
      chessboard_img=frame,
      fix_rotate=True,
      rotate_val=90,
      add_padding=True
    )
    chessboard_calibration.saveMapping()
    
    # release camera
    camera.destroy()
    print('Done!')

  def start(self):
    """
    Start game
    """
    self.__camera = Camera(self.__cam_address)

    self.__running_calibration = ChessboardCalibration()
    found, self.__board = self.__running_calibration.loadMapping()
    if not found:
      raise Exception('No mapping found. Run calibration mapping')

    self.setKeyPressEvent(self.__keyPressEvent)
    self.__captureFrame()
    self.__runScan(only_prediction=True)
    self.show()

  def __captureFrame(self):
    frame = self.__camera.capture()
    self.__processed_image = self.__running_calibration.applyMapping(frame)
    
    result = self.__addFakeBoundingBoxes()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    self.setImage(result, index=0)
    self.__updateFrameRate()

    QtCore.QTimer.singleShot(1, self.__captureFrame)

  def __addFakeBoundingBoxes(self):
    # Isso mesmo: fake bounding boxes!!
    # Esse método gera as caixas fake utilizando rastreamento de cor HSV
    # Tudo isso para dar a impressão que tudo está acontecendo real time ;)

    # Mais tarde o darknet vai passar e de fato detectar os objetos, até lá
    # precisamos mostrar algo incrível para o público

    inverted = cv2.bitwise_not(self.__processed_image.copy())
    hsv = cv2.cvtColor(inverted, cv2.COLOR_BGR2HSV)

    # ========
    # eliminar as casas
    # ========

    # ---- verdes
    lower = np.array([135, 6, 91])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(hsv.copy(), lower, upper)
    mask = 255-mask
    green_squares_mask = mask.copy()

    # ---- brancas
    hsv = cv2.cvtColor(self.__processed_image.copy(), cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 81])
    upper = np.array([255, 43, 255])
    mask = cv2.inRange(hsv.copy(), lower, upper)
    mask = 255-mask
    white_squares_mask = mask.copy()

    # ---- resultado final sem as casas
    image_final = self.__processed_image.copy()
    image_final = cv2.bitwise_and(image_final, image_final, mask=green_squares_mask)
    image_final = cv2.bitwise_and(image_final, image_final, mask=white_squares_mask)

    # ========
    # Seleciona as peças
    # ========
    inverted = cv2.bitwise_not(image_final.copy())
    hsv = cv2.cvtColor(inverted, cv2.COLOR_BGR2HSV)

    # ---- peças brancas
    target = self.__processed_image.copy()
    lower = np.array([76, 87, 50])
    upper = np.array([255, 255, 255])
    white_pieces_mask = cv2.inRange(hsv.copy(), lower, upper)

    # ---- peças pretas
    lower = np.array([0, 0, 159])
    upper = np.array([55, 255, 255])
    black_pieces_mask = cv2.inRange(hsv.copy(), lower, upper)

    hand_is_detected, hand_contours = self.__hand_detected(image_final, white_pieces_mask, black_pieces_mask)
    if hand_is_detected:
      self.__drawHand(target, hand_contours)
    else:
      res = cv2.bitwise_and(image_final, image_final, mask=white_pieces_mask)
      self.drawPiecesBoundingBoxes(res, target, (0, 255, 0))

      res = cv2.bitwise_and(image_final, image_final, mask=black_pieces_mask)
      self.drawPiecesBoundingBoxes(res, target, (255, 0, 0))

    return target

  def drawPiecesBoundingBoxes(self, res, target, color):
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
    thresh = cv2.bitwise_not(thresh)

    # find contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    boxes = []
    confidences = []

    for cnt in cnts:
      area = cv2.contourArea(cnt)
      if area > 200:
        peri = cv2.arcLength(cnt, True)
        biggest_cnt = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        x, y, w, h = cv2.boundingRect(biggest_cnt)
        boxes.append([x, y, int(x+w), int(y+h)])
        confidences.append(float(0.6))

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.35, 0.7)
    if len(idxs) > 0:
      for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        cv2.rectangle(target, (x,y), (w, h), color, 2)

  def __drawHand(self, target, hand_contours):
    peri = cv2.arcLength(hand_contours, True)
    biggest_cnt = cv2.approxPolyDP(hand_contours, 0.015 * peri, True)
    x, y, w, h = cv2.boundingRect(biggest_cnt)
    cv2.rectangle(target, (x,y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(target, 'HUMAN HAND', (x, y - 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

  def __hand_detected(self, no_houses_frame, white_pieces_mask, black_pieces_mask) -> Tuple[bool, list]:
    """
    return `True` or `False` if hand is detected
    """
    white_pieces_mask = 255-white_pieces_mask
    black_pieces_mask = 255-black_pieces_mask

    no_houses_frame = cv2.bitwise_and(no_houses_frame, no_houses_frame, mask=white_pieces_mask)
    no_houses_frame = cv2.bitwise_and(no_houses_frame, no_houses_frame, mask=black_pieces_mask)

    # convert image to gray scale
    gray = cv2.cvtColor(no_houses_frame, cv2.COLOR_BGR2GRAY)

    # This is the threshold level for every pixel.
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=8)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if cnts is not None and len(cnts) > 0:
      # Estou assumindo que a coisa maior na imagem diferente das casas e das peças
      # é uma mão, mas isso não é uma vdd absoluta.
      cnt = max(cnts, key=cv2.contourArea)
      return (True, cnt)
    else:
      return (False, None)

  def __updateFrameRate(self):
    now = time.time()
    dt = (now - self.__lastupdate)
    if dt <= 0:
      dt = 0.000000000001

    fps2 = 1.0 / dt
    self.__lastupdate = now
    self.__fps = self.__fps * 0.9 + fps2 * 0.1
    self.print('Mean Frame Rate:  {:.2f} FPS'.format(self.__fps), index=0)

  def __toPNGImage(self):
    out = BytesIO()
    bytestring = chess.svg.board(self.__agent.board, size=640).encode('utf-8')
    cairosvg.svg2png(bytestring=bytestring, write_to=out)
    image = Image.open(out)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)

  def __keyPressEvent(self, e):
    if type(e) == QtGui.QKeyEvent:
      if e.key() == QtCore.Qt.Key_Return or e.key() == QtCore.Qt.Key_Enter:
        self.__runScan()
      e.accept()
    else:
      e.ignore()

  def __runScan(self, only_prediction: bool = False):
    squares, detections = self.__board.scan(self.__processed_image)
    board_state = self.__board.toMatrix(squares)

    cvImage = self.__cvPredictionImage(self.__processed_image.copy())
    self.setImage(cvImage, index=1)
    self.addBoundingBoxes(
      detections,
      viewIndex=1,
      class_colors=COLORS,
      symbols=SYMBOLS
    )

    if not only_prediction:
      human_move = self.__agent.state2Move(board_state)
      if human_move is not None:
        self.print('HUMAN: {}'.format(human_move.uci()))
        self.__agent.makeMove(human_move)
        self.__agent.updateState(board_state)

      cpu_move = self.__agent.chooseMove()
      if cpu_move is not None:
        self.print('BOT: {}'.format(cpu_move.uci()))
        self.__agent.makeMove(cpu_move)
        self.__agent.updateState(self.__agent.board.state())

  def __cvPredictionImage(self, src) -> np.array:
    rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    inverted = cv2.bitwise_not(rgb)
    color_map = cv2.applyColorMap(inverted, cv2.COLORMAP_OCEAN)
    gray = cv2.cvtColor(color_map, cv2.COLOR_RGB2GRAY)
    return gray