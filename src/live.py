import sys
import time

import cv2
from model.camera import Camera
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
from model.chessboard_calibration import ChessboardCalibration

# import pyqtgraph.examples
# pyqtgraph.examples.run()

class App(QtGui.QMainWindow):
  def __init__(self, parent=None):
    super(App, self).__init__(parent)

    #### Create Gui Elements ###########
    self.canvas = pg.GraphicsLayoutWidget(size=(540, 540))
    self.canvas.setWindowTitle('pyqtgraph example: ImageItem')
    self.canvas.ci.layout.setContentsMargins(0, 0, 0, 0)
    
    self.view = self.canvas.addViewBox()
    self.view.setBackgroundColor('r')
    self.view.suggestPadding = lambda *_: 0.0
    self.view.installEventFilter(self)

    self.img = pg.ImageItem()
    self.view.addItem(self.img)

    #### Set Data  #####################
    self.counter = 0
    self.fps = 0.
    self.lastupdate = time.time()

    #### Start  #####################
    self.camera = Camera('http://192.168.0.111:4747/video', fps=30)
    self.__running_calibration = ChessboardCalibration()
    found, self.board = self.__running_calibration.loadMapping()
    self._update()

  def eventFilter(self, watched, event):
    if event.type() == QtCore.QEvent.GraphicsSceneWheel:
      return True
    return super().eventFilter(watched, event)

  def show(self):
    self.canvas.show()

  def _update(self):
    frame = self.camera.capture()
    img = self.__running_calibration.applyMapping(frame)
    
    self.board.scan(img)
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.img.setImage(cv2.rotate(rgb_img, cv2.ROTATE_90_CLOCKWISE))

    now = time.time()
    dt = (now - self.lastupdate)
    if dt <= 0:
      dt = 0.000000000001
    fps2 = 1.0 / dt
    self.lastupdate = now
    self.fps = self.fps * 0.9 + fps2 * 0.1
    tx = 'Mean Frame Rate:  {:.3f} FPS'.format(self.fps)
    # self.label.setText(tx)
    QtCore.QTimer.singleShot(1, self._update)
    self.counter += 1


if __name__ == '__main__':
  app = QtGui.QApplication(sys.argv)
  thisapp = App()
  thisapp.show()
  sys.exit(app.exec_())