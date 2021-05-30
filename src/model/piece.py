from enumeration import Color, PieceName

class Piece:
  name:PieceName = None
  color:Color = None
  acc: float
  classID: int = None

  def __init__(self, name=None, acc=None, classID=None):
    self.acc = acc

    arr = name.split('-')
    self.color = Color.BLACK if arr[0] == 'black' else Color.WHITE
    self.name = PieceName(arr[1])
    self.classID = classID