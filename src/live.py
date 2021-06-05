import sys
import time

import cv2
from pyqtgraph import Qt
from pyqtgraph.graphicsItems.ViewBox.ViewBox import ViewBox
from model.camera import Camera
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
from model.chessboard_calibration import ChessboardCalibration

# import pyqtgraph.examples
# pyqtgraph.examples.run()

class App(QtGui.QMainWindow):
  __canvas: pg.GraphicsLayoutWidget = None

  def __init__(self, parent=None):
    super(App, self).__init__(parent)

    #### Create Gui Elements ###########
    self.__canvas = pg.GraphicsLayoutWidget(size=(500 * 2, 500))
    self.__canvas.setWindowTitle('pyqtgraph example: ImageItem')
    self.__canvas.ci.layout.setContentsMargins(0, 0, 0, 0)

    ## add grid
    self.imgs:list[pg.ImageItem] = []
    views = self.__addGridLayout(numrows=1, numcols=2)
    for view in views:
      imgItem = pg.ImageItem()
      self.imgs.append(imgItem)
      view.addItem(imgItem)

    ## legends
    self.__createConsole()

    #### Set Data  #####################
    self.counter = 0
    self.fps = 0.
    self.lastupdate = time.time()

    #### Start  #####################
    self.camera = Camera('http://192.168.0.111:4747/video', fps=30)
    self.__running_calibration = ChessboardCalibration()
    found, self.board = self.__running_calibration.loadMapping()
    self.__update()

  def __addGridLayout(self, numrows: int, numcols: int) -> list[pg.ViewBox]:
    views: list[pg.ViewBox] = []
    for r in range(numrows):
      for c in range(numcols):
        view: pg.ViewBox = self.__canvas.addViewBox(row=r, col=c)
        view.suggestPadding = lambda *_: 0.0
        view.installEventFilter(self)
        view.setBackgroundColor('r')
        views.append(view)
    return views

  def __createConsole(self) -> pg.ViewBox:
    self.label = QtGui.QLabel(self.__canvas)
    self.label.setStyleSheet('QLabel { color: yellow; margin: 10px; font-weight: bold }')

  def eventFilter(self, watched, event):
    if event.type() == QtCore.QEvent.GraphicsSceneWheel:
      return True
    return super().eventFilter(watched, event)

  def setImage(self, img):
    for imgItem in self.imgs:
      imgItem.setImage(img)

  def show(self):
    self.__canvas.show()

  def __update(self):
    frame = self.camera.capture()
    img = self.__running_calibration.applyMapping(frame)
    
    # self.board.scan(img)
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.setImage(cv2.rotate(rgb_img, cv2.ROTATE_90_CLOCKWISE))

    now = time.time()
    dt = (now - self.lastupdate)
    if dt <= 0:
      dt = 0.000000000001
    fps2 = 1.0 / dt
    self.lastupdate = now
    self.fps = self.fps * 0.9 + fps2 * 0.1
    tx = 'Mean Frame Rate:  {:.2f} FPS'.format(self.fps)
    self.label.setText(tx)
    self.label.adjustSize()
    QtCore.QTimer.singleShot(1, self.__update)
    self.counter += 1

if __name__ == '__main__':
  app = QtGui.QApplication(sys.argv)
  thisapp = App()
  thisapp.show()
  sys.exit(app.exec_())