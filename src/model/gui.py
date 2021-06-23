from typing import Optional
from pyqtgraph.Qt import QtGui

import pyqtgraph as pg

# Aviso: Toda parte de interface está aqui
#   - console log
#   - main grid
#
# Eu poderia ter separado em classes mas não fiz :(

class GUI(QtGui.QMainWindow):
  __canvas: pg.GraphicsLayoutWidget = None
  __layout: pg.GraphicsLayout = None
  __window_size = (640, 640)
  __console_texts: list[str] = []
  __max_buffer_size = 10
  __view: pg.ViewBox = None
  __image_item: pg.ImageItem = None

  def __init__(self, title: str = 'preview'):
    super(GUI, self).__init__(parent=None)
    self.__canvas = pg.GraphicsLayoutWidget(size=self.__window_size)
    self.__canvas.setWindowTitle(title)

    self.__layout = pg.GraphicsLayout()
    self.__layout.setContentsMargins(0, 0, 0, 0)
    self.__canvas.setCentralItem(self.__layout)

    ## add view and image
    self.__view = pg.ViewBox(enableMouse=False)
    self.__view.suggestPadding = lambda *_: 0.0
    self.__view.invertY()
    self.__layout.addItem(self.__view)

    self.__image_item = pg.ImageItem(axisOrder='row-major')
    self.__view.addItem(self.__image_item)

    ## setup console log
    self.__createConsole()

    ## define tool tip settings
    QtGui.QToolTip.setFont(QtGui.QFont('Helvetica', 18))

  def __createConsole(self):
    self.label = QtGui.QLabel(self.__canvas)
    self.label.setStyleSheet('QLabel { color: yellow; margin: 10px; font-weight: bold }')

  def __showConsoleText(self):
    self.label.setText('\n'.join(self.__console_texts))
    self.label.adjustSize()

  def setImage(self, img):
    self.__image_item.setImage(img)

  def setKeyPressEvent(self, callback):
    if callback is not None:
      self.__canvas.keyPressEvent = callback

  def print(self, text: str = '', index: Optional[int] = None):
    if index is None:
      self.__console_texts.append(text)
    else:
      if len(self.__console_texts) > 0:
        self.__console_texts.pop(index)
      self.__console_texts.insert(index, text)

    if len(self.__console_texts) > self.__max_buffer_size:
      self.__console_texts.pop(1)

    self.__showConsoleText()

  def show(self):
    """
    Show application window
    """
    self.__canvas.show()