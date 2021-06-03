from typing import Optional
import numpy as np
import chess

class Agent:
  """
  Smart chess agent
  """
  ___initial_board_state = np.array([
    [7, 11, 8, 10, 9, 8, 11, 7],
    [6, 6, 6, 6, 6, 6, 6, 6],
    [-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 3, 2, 5, 4, 2, 3, 1]
  ])

  __current_board_state: np.array = None
  __board: chess.Board = None
  __indexes = sorted(np.arange(1, 9), reverse=True)
  __columns = list("abcdefgh")

  def __init__(self, board):
    self.__current_board_state = self.___initial_board_state
    self.__board = board

  def setState(self, board_state) -> Optional[chess.Move]:
    """
    update chess board state matrix

    @return
    The `chess.Move` made based on state matrices
    """
    # check if state change
    result = np.abs(self.__current_board_state - board_state)

    # save new state
    self.__current_board_state = board_state

    coordinates = []
    for ridx, row in enumerate(result):
      for cidx, col in enumerate(row):
        if col > 0:
          coordinates.append('{}{}'.format(self.__columns[cidx], self.__indexes[ridx]))

    if len(coordinates) > 0:
      return self.__validMove(coordinates)
    return None

  def makeMove(self, move: chess.Move):
    """
    Make a move on virtual chess board
    """
    self.__board.push(move)

  def __validMove(self, coordinates) -> chess.Move:
    """
    find a valid `chess.Move` base on generated coordinates
    """
    moviments = []
    for coord_i in coordinates:
      for coord_j in coordinates:
        if coord_i != coord_j:
          moviments.append(coord_i+coord_j)

    if len(moviments) > 0:
      for uci_str in moviments:
        move = chess.Move.from_uci(uci_str)
        if move in self.__board.legal_moves:
          print('Move coordinate ........: {}'.format(uci_str))
          return move
    return None
