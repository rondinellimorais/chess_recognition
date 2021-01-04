from model.square import Square
from enumeration import Color

class Board:
  contours: []
  squares: list = None

  def colorBuild(self):
    """
    Build the color of the squares
    """
    if self.squares != None:
      curr_color = Color.UNDEFINED
      for (row_idx, row) in enumerate(self.squares):
        curr_color = Color.WHITE if (row_idx % 2) == 0 else Color.BLACK
        for (col_idx, square) in enumerate(row):
          square.color = curr_color
          curr_color = Color.BLACK if curr_color == Color.WHITE else Color.WHITE
  
  def state(self, img):
    """
    docstring
    """
    print('\n==================')
    if self.squares is not None:
      for (r_idx, row) in enumerate(self.squares):
        for (c_idx, square) in enumerate(row):
          piece:Piece = square.piece(img)

          if piece is not None:
            print('[{}, {}] {} | {}'.format(r_idx, c_idx, piece.color, piece.name))
          else:
            print('[{}, {}] Empty'.format(r_idx, c_idx))