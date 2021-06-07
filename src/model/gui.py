from typing import Optional
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui

import pyqtgraph as pg

class GUI(QtGui.QMainWindow):
  __canvas: pg.GraphicsLayoutWidget = None
  __layout: pg.GraphicsLayout = None
  __grid_size = (2, 1)
  __window_size = (640 * __grid_size[0], 640 * __grid_size[1])
  __console_texts: list[str] = []
  __max_buffer_size = 10
  __views: list[pg.ViewBox] = None
  __imageItems: list[pg.ImageItem] = None

  def __init__(self, title: str = 'preview'):
    super(GUI, self).__init__(parent=None)
    self.__canvas = pg.GraphicsLayoutWidget(size=self.__window_size)
    self.__canvas.setWindowTitle(title)

    self.__layout = pg.GraphicsLayout()
    self.__layout.setContentsMargins(0, 0, 0, 0)
    self.__canvas.setCentralItem(self.__layout)

    ## add image grid
    self.__imageItems:list[pg.ImageItem] = []
    self.__views = self.__addGridLayout(numrows=self.__grid_size[1], numcols=self.__grid_size[0])
    for view in self.__views:
      imgItem = pg.ImageItem(axisOrder='row-major')
      self.__imageItems.append(imgItem)
      view.addItem(imgItem)

    ## setup console log
    self.__createConsole()
  
  def setKeyPressEvent(self, callback):
    if callback is not None:
      self.__canvas.keyPressEvent = callback

  def __addGridLayout(self, numrows: int, numcols: int) -> list[pg.ViewBox]:
    views: list[pg.ViewBox] = []
    for r in range(numrows):
      for c in range(numcols):
        view = pg.ViewBox(enableMouse=False)
        view.suggestPadding = lambda *_: 0.0
        view.invertY()
        views.append(view)
        self.__layout.addItem(view, row=r, col=c)
    return views

  def __createConsole(self):
    self.label = QtGui.QLabel(self.__canvas)
    self.label.setStyleSheet('QLabel { color: yellow; margin: 10px; font-weight: bold }')

  def setConsoleText(self, text: str = '', index: Optional[int] = None):
    if index is None:
      self.__console_texts.append(text)
    else:
      if len(self.__console_texts) > 0:
        self.__console_texts.pop(index)
      self.__console_texts.insert(index, text)

    if len(self.__console_texts) > self.__max_buffer_size:
      self.__console_texts.pop(1)

    self.__showConsoleText()

  def __showConsoleText(self):
    self.label.setText('\n'.join(self.__console_texts))
    self.label.adjustSize()

  def setImage(self, img, index=None):
    if index is not None:
      imgItem = self.__imageItems[index]
      imgItem.setImage(img)
    else:
      for imgItem in self.__imageItems:
        imgItem.setImage(img)

  def show(self):
    """
    Show application window
    """
    self.__canvas.show()

  def addBoundingBoxes(self, detections: list, viewIndex: int = 0):
    QtGui.QToolTip.setFont(QtGui.QFont('Helvetica', 14))
    COLORS = np.random.randint(0, 255, size=(12, 3), dtype="uint8")

    for (name, bounding_box, accuracy, class_id) in detections:
      color = [int(c) for c in COLORS[class_id]]
      x,y,w,h = bounding_box

      box = pg.ViewBox(parent=self.__views[viewIndex], border='r')
      box.setGeometry(x, y, w-x, h-y)
      box.setZValue(1)
      box.setBorder(pg.mkPen(color, width=6))
      box.setToolTip('{}\naccuracy: {:.2f}%'.format(name, accuracy * 100)) 