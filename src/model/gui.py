from typing import Optional, Tuple
from pyqtgraph.Qt import QtCore, QtGui

import pyqtgraph as pg

class GUI(QtGui.QMainWindow):
  __canvas: pg.GraphicsLayoutWidget = None
  __grid_size = (2, 1)
  __window_size = (500 * __grid_size[0], 500 * __grid_size[1])
  __console_texts: list[str] = []
  __max_buffer_size = 10

  def __init__(self, title: str = 'preview'):
    super(GUI, self).__init__(parent=None)

    #### Create Gui Elements ###########
    self.__canvas = pg.GraphicsLayoutWidget(size=self.__window_size)
    self.__canvas.setWindowTitle(title)
    self.__canvas.ci.layout.setContentsMargins(0, 0, 0, 0)

    ## add image grid
    self.imageItems:list[pg.ImageItem] = []
    views = self.__addGridLayout(numrows=self.__grid_size[1], numcols=self.__grid_size[0])
    for view in views:
      imgItem = pg.ImageItem(axisOrder='row-major')
      self.imageItems.append(imgItem)
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
        view: pg.ViewBox = self.__canvas.addViewBox(row=r, col=c)
        view.suggestPadding = lambda *_: 0.0
        view.installEventFilter(self)
        view.setBackgroundColor('r')
        view.invertY()
        views.append(view)
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

  def eventFilter(self, watched, event):
    if event.type() == QtCore.QEvent.GraphicsSceneWheel:
      return True
    return super().eventFilter(watched, event)

  def setImage(self, img, index=None):
    if index is not None:
      imgItem = self.imageItems[index]
      imgItem.setImage(img)
    else:
      for imgItem in self.imageItems:
        imgItem.setImage(img)

  def show(self):
    """
    Show application window
    """
    self.__canvas.show()