from typing import Dict
from model.chessboard_calibration import ChessboardCalibration
from model.board import Board
from model.camera import Camera
from dotenv import dotenv_values
import cv2
import time

class Game:
  fps: int
  cam_address: str
  running_calibration: ChessboardCalibration
  board: Board
  previousMillis: int = 0
  __config: Dict

  def __init__(self):
    self.__config = dotenv_values()
    self.fps = int(self.__config.get('CAM_FPS'))
    self.cam_address = self.__config.get('CAM_ADDRESS')

  def mapping(self):
    """
    Start mapping chess board
    """
    camera = Camera(self.cam_address, fps=self.fps)
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
    camera = Camera(self.cam_address, fps=self.fps)

    self.running_calibration = ChessboardCalibration()
    found, self.board = self.running_calibration.loadMapping()
    if not found:
      raise Exception('No mapping found. Run calibration mapping')

    camera.startRunning(self.didCaptureFrame)

  def didCaptureFrame(self, frame, camera):
    """
    Camera frame delegate
    """
    img = self.running_calibration.applyMapping(frame)

    currentMillis = round(time.time() * 1000)

    # check to see if it's time to running vision computer; that is, if the difference
    # between the current time and last time we running vision computer is bigger than
    # the interval at which we want to running vision computer.
    if currentMillis - self.previousMillis >= int(self.__config.get('CHECK_BOARD_STATE_INTERVAL')):
      self.previousMillis = currentMillis
      self.board.state(img)

    cv2.imshow("preview", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        camera.stopRunning() 
        cv2.destroyAllWindows()