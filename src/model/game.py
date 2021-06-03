from typing import Dict
from model.chessboard_calibration import ChessboardCalibration
from model.board import Board
from model.agent import Agent
from model.camera import Camera
from model.virtual_board import VirtualBoard
from dotenv import dotenv_values
from PIL import Image
from io import BytesIO

import cv2
import chess
import numpy as np
import chess.svg
import cairosvg

class Game:
  __fps: int
  __cam_address: str
  __running_calibration: ChessboardCalibration
  __board: Board
  __config: Dict
  __virtualBoard: VirtualBoard
  __agent: Agent

  def __init__(self):
    self.__config = dotenv_values()
    self.__fps = int(self.__config.get('CAM_FPS'))
    self.__cam_address = self.__config.get('CAM_ADDRESS')
    self.__virtualBoard = VirtualBoard()
    self.__agent = Agent(self.__virtualBoard)

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

    key_pressed = cv2.waitKey(1)
    if key_pressed & 0xFF == ord('q'):
      camera.stopRunning()
    elif key_pressed == 13: # Enter(13)
      squares = self.__board.scan(img)
      board_state = self.__board.toMatrix(squares)

      human_move = self.__agent.state2Move(board_state)
      if human_move is not None:
        self.__agent.makeMove(human_move)
        self.__agent.updateState(board_state)

      cpu_move = self.__agent.chooseMove()
      if cpu_move is not None:
        self.__agent.makeMove(cpu_move)
        self.__agent.updateState(self.__virtualBoard.state())

    virtualBoardImage = self.__toPNGImage()
    img = np.hstack((img, virtualBoardImage)) # np.hstack tem um performance bem ruim :(
    cv2.imshow('chess board computer vision', img)