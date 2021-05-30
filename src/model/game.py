from model.chessboard_calibration import ChessboardCalibration
from model.board import Board
from model.camera import Camera
import cv2

class Game:
  fps: int
  cam_address: str
  running_calibration: ChessboardCalibration
  board: Board

  def __init__(self):
    self.fps = 30
    self.cam_address = 'http://192.168.0.110:4747/video'

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

    # self.board.state(img)

    cv2.imshow("preview", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        camera.stopRunning() 
        cv2.destroyAllWindows()