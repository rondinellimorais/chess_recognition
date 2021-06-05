from typing import Dict
from model.chessboard_calibration import ChessboardCalibration
from model.board import Board
from model.agent import Agent
from model.camera import Camera
from model.gui import GUI
from dotenv import dotenv_values
from PIL import Image
from io import BytesIO
from pyqtgraph.Qt import QtCore

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

    self.show()
    self.__captureFrame()

  def __captureFrame(self):
    frame = self.__camera.capture()
    img = self.__running_calibration.applyMapping(frame)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.setImage(img)
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
    self.setConsoleText('Mean Frame Rate:  {:.2f} FPS'.format(self.__fps))

  def __toPNGImage(self):
    out = BytesIO()
    bytestring = chess.svg.board(self.__agent.board, size=640).encode('utf-8')
    cairosvg.svg2png(bytestring=bytestring, write_to=out)
    image = Image.open(out)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)

  # def __didCaptureFrame(self, frame, camera):
  #   """
  #   Camera frame delegate
  #   """
  #   img = self.__running_calibration.applyMapping(frame)

  #   key_pressed = cv2.waitKey(1)
  #   if key_pressed & 0xFF == ord('q'):
  #     camera.stopRunning()
  #   elif key_pressed == 13: # Enter(13)
  #     squares = self.__board.scan(img)
  #     board_state = self.__board.toMatrix(squares)
  #     human_move = self.__agent.state2Move(board_state)
  #     if human_move is not None:
  #       self.__agent.makeMove(human_move)
  #       self.__agent.updateState(board_state)

  #     cpu_move = self.__agent.chooseMove()
  #     if cpu_move is not None:
  #       self.__agent.makeMove(cpu_move)
  #       self.__agent.updateState(self.__agent.board.state())

  #   virtualBoardImage = self.__toPNGImage()
  #   img = np.hstack((img, virtualBoardImage)) # np.hstack tem um performance bem ruim :(
  #   cv2.imshow('chess board computer vision', img)