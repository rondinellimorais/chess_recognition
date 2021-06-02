from typing import Dict
from model.chessboard_calibration import ChessboardCalibration
from model.board import Board
from model.agent import Agent
from model.camera import Camera
from dotenv import dotenv_values
from PIL import Image
from io import BytesIO

import cv2
import time
import chess
import numpy as np
import chess.svg
import cairosvg

class Game:
  __fps: int
  __cam_address: str
  __running_calibration: ChessboardCalibration
  __board: Board
  __previousMillis: int = 0
  __config: Dict
  __virtualBoard: chess.Board
  __agent: Agent

  def __init__(self):
    self.__config = dotenv_values()
    self.__fps = int(self.__config.get('CAM_FPS'))
    self.__cam_address = self.__config.get('CAM_ADDRESS')
    self.__virtualBoard = chess.Board()
    self.__agent = Agent()

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
    camera = Camera(self.__cam_address, fps=self.__fps)

    self.__running_calibration = ChessboardCalibration()
    found, self.__board = self.__running_calibration.loadMapping()
    if not found:
      raise Exception('No mapping found. Run calibration mapping')

    camera.startRunning(self.__didCaptureFrame)

  def __toPNGImage(self):
    out = BytesIO()
    bytestring = chess.svg.board(self.__virtualBoard, size=640).encode('utf-8')
    cairosvg.svg2png(bytestring=bytestring, write_to=out)
    image = Image.open(out)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)

  def __didCaptureFrame(self, frame, camera):
    """
    Camera frame delegate
    """
    img = self.__running_calibration.applyMapping(frame)

    currentMillis = round(time.time() * 1000)

    # check to see if it's time to running vision computer; that is, if the difference
    # between the current time and last time we running vision computer is bigger than
    # the interval at which we want to running vision computer.
    if currentMillis - self.__previousMillis >= int(self.__config.get('CHECK_BOARD_STATE_INTERVAL')):
      self.__previousMillis = currentMillis
      self.__board.state(img)
      board_state = self.__board.toMatrix()
      self.__agent.setState(board_state)

    virtualBoardImage = self.__toPNGImage()
    hstack = np.hstack((img, virtualBoardImage))

    cv2.imshow('chess board computer vision', hstack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        camera.stopRunning() 
        cv2.destroyAllWindows()