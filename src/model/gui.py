import sys

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

class GUI(QtGui.QMainWindow):
  __canvas: pg.GraphicsLayoutWidget = None
  __grid_size = (2, 1)
  __window_size = (500 * __grid_size[0], 500 * __grid_size[1])

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
      imgItem = pg.ImageItem()
      self.imageItems.append(imgItem)
      view.addItem(imgItem)

    ## legends
    self.__createConsole()

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

  def setConsoleText(self, text: str = ''):
    self.label.setText(text)
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