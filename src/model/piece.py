from enumeration import Color, PieceName

class Piece:
  name:PieceName = None
  color:Color = None
  acc: float

  def __init__(self, name=None, acc=None):
    self.acc = acc

    arr = name.split('-')
    self.color = Color.BLACK if arr[0] == 'black' else Color.WHITE
    self.name = PieceName(arr[1])