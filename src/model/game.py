from typing import Dict
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
    camera = Camera(self.__cam_address, fps=self.__fps)
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

    self.__captureFrame()
    self.setKeyPressEvent(self.__keyPressEvent)
    self.show()

  def __captureFrame(self):
    frame = self.__camera.capture()
    self.__processed_image = self.__running_calibration.applyMapping(frame)
    rgb = cv2.cvtColor(self.__processed_image.copy(), cv2.COLOR_BGR2RGB)
    self.setImage(rgb, index=0)
    self.__updateFrameRate()

    QtCore.QTimer.singleShot(1, self.__captureFrame)

  def __updateFrameRate(self):
    now = time.time()
    dt = (now - self.__lastupdate)
    if dt <= 0:
      dt = 0.000000000001

    fps2 = 1.0 / dt
    self.__lastupdate = now
    self.__fps = self.__fps * 0.9 + fps2 * 0.1
    self.setConsoleText('Mean Frame Rate:  {:.2f} FPS'.format(self.__fps), index=0)

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

  def __runScan(self):
    squares, detections = self.__board.scan(self.__processed_image)
    board_state = self.__board.toMatrix(squares)

    cvImage = self.__cvPredictionImage(self.__processed_image.copy(), detections)
    self.setImage(cvImage, index=1)

    human_move = self.__agent.state2Move(board_state)
    if human_move is not None:
      self.setConsoleText('HUMAN: {}'.format(human_move.uci()))
      self.__agent.makeMove(human_move)
      self.__agent.updateState(board_state)

    cpu_move = self.__agent.chooseMove()
    if cpu_move is not None:
      self.setConsoleText('BOT: {}'.format(cpu_move.uci()))
      self.__agent.makeMove(cpu_move)
      self.__agent.updateState(self.__agent.board.state())

  def __cvPredictionImage(self, src, detections: list) -> np.array:
    rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    inverted = cv2.bitwise_not(rgb)
    color_map = cv2.applyColorMap(inverted, cv2.COLORMAP_TWILIGHT)
    gray = cv2.cvtColor(color_map, cv2.COLOR_RGB2GRAY)
    return gray

  def __drawBoundingBoxes(self, src, detections: list) -> np.array:
    COLORS = np.random.randint(0, 255, size=(12, 3), dtype="uint8")

    for (_, bounding_box, _, class_id) in detections:
      color = [int(c) for c in COLORS[class_id]]
      x,y,w,h = bounding_box
      cv2.rectangle(src, (x, y), (w, h), color, 2)
      cv2.putText(src, str(class_id), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return src