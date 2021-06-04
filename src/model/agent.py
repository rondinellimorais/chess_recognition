from model.virtual_board import VirtualBoard
from typing import Optional
import numpy as np
import chess
import chess.svg
import chess.pgn
import chess.polyglot
import chess.engine
import time

class Agent:
  """
  Smart chess agent
  """
  board: VirtualBoard = None

  __current_board_state: np.array = None
  __indexes = sorted(np.arange(1, 9), reverse=True)
  __columns = list("abcdefgh")

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

  __pawnEvalWhite = np.reshape([
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0],
    [1.0,  1.0,  2.0,  3.0,  3.0,  2.0,  1.0,  1.0],
    [0.5,  0.5,  1.0,  2.5,  2.5,  1.0,  0.5,  0.5],
    [0.0,  0.0,  0.0,  2.0,  2.0,  0.0,  0.0,  0.0],
    [0.5, -0.5, -1.0,  0.0,  0.0, -1.0, -0.5,  0.5],
    [0.5,  1.0, 1.0,  -2.0, -2.0,  1.0,  1.0,  0.5],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]
  ], 64)

  __pawnEvalBlack = np.flip(__pawnEvalWhite, axis=0)

  __knightEval = np.reshape([
    [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0],
    [-4.0, -2.0,  0.0,  0.0,  0.0,  0.0, -2.0, -4.0],
    [-3.0,  0.0,  1.0,  1.5,  1.5,  1.0,  0.0, -3.0],
    [-3.0,  0.5,  1.5,  2.0,  2.0,  1.5,  0.5, -3.0],
    [-3.0,  0.0,  1.5,  2.0,  2.0,  1.5,  0.0, -3.0],
    [-3.0,  0.5,  1.0,  1.5,  1.5,  1.0,  0.5, -3.0],
    [-4.0, -2.0,  0.0,  0.5,  0.5,  0.0, -2.0, -4.0],
    [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0]
  ], 64)

  __bishopEvalWhite = np.reshape([
    [ -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0],
    [ -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0],
    [ -1.0,  0.0,  0.5,  1.0,  1.0,  0.5,  0.0, -1.0],
    [ -1.0,  0.5,  0.5,  1.0,  1.0,  0.5,  0.5, -1.0],
    [ -1.0,  0.0,  1.0,  1.0,  1.0,  1.0,  0.0, -1.0],
    [ -1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, -1.0],
    [ -1.0,  0.5,  0.0,  0.0,  0.0,  0.0,  0.5, -1.0],
    [ -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0]
  ], 64)

  __bishopEvalBlack = np.flip(__bishopEvalWhite, axis=0)

  __rookEvalWhite = np.reshape([
    [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [  0.5,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.5],
    [ -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
    [ -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
    [ -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
    [ -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
    [ -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
    [  0.0,   0.0, 0.0,  0.5,  0.5,  0.0,  0.0,  0.0]
  ], 64)

  __rookEvalBlack = np.flip(__rookEvalWhite, axis=0)

  __evalQueen = np.reshape([
    [ -2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0],
    [ -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0],
    [ -1.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0],
    [ -0.5,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5],
    [  0.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5],
    [ -1.0,  0.5,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0],
    [ -1.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.0, -1.0],
    [ -2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]
  ], 64)

  __kingEvalWhite = np.reshape([
    [ -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
    [ -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
    [ -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
    [ -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
    [ -2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0],
    [ -1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0],
    [  2.0,  2.0,  0.0,  0.0,  0.0,  0.0,  2.0,  2.0 ],
    [  2.0,  3.0,  1.0,  0.0,  0.0,  1.0,  3.0,  2.0 ]
  ], 64)

  __kingEvalBlack = np.flip(__kingEvalWhite, axis=0)
  __position_count = 0

  def __init__(self):
    self.__current_board_state = self.___initial_board_state
    self.board = VirtualBoard()

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
        if move in self.board.legal_moves:
          print('Move coordinate ........: {}'.format(uci_str))
          return move
    return None

  def chooseMove(self) -> Optional[chess.Move]:
    best_move = self.__getBestMove()
    if self.board.is_game_over():
      print('Game over')
      return None
    return best_move

  def __getBestMove(self, depth = 3) -> Optional[chess.Move]:
    if self.board.is_game_over():
      print('Game over')
      return

    self.__position_count = 0

    d = time.time()
    best_move = self.__minimaxRoot(depth, True)
    d2 = time.time()
    moveTime = (d2 - d);
    positionsPerS = (self.__position_count * 1000 / moveTime)

    print('Positions evaluated: {}'.format(self.__position_count))
    print('Time: {:.3f}s'.format(moveTime/1000))
    print('Positions/s: {}\n\n'.format(positionsPerS))

    return best_move

  def __minimaxRoot(self, depth: int, is_maximising_player: bool) -> Optional[chess.Move]:
    best_move = -9999
    best_move_found: Optional[chess.Move] = None

    for move in self.board.legal_moves:
      self.board.push(move)
      value = self.__miniMax(depth - 1, -10000, 10000, not is_maximising_player)
      self.board.pop()
      if value >= best_move:
        best_move = value
        best_move_found = move
    return best_move_found

  def __miniMax(self, depth: int, alpha: int, beta: int, is_maximising_player: bool) -> int:
    self.__position_count += 1
    if depth == 0:
      return -self.__evaluateBoard()

    if is_maximising_player:
      best_move = -9999
      for move in self.board.legal_moves:
        self.board.push(move)
        best_move = max(best_move, self.__miniMax(depth - 1, alpha, beta, not is_maximising_player))
        self.board.pop()
        alpha = max(alpha, best_move);
        if beta <= alpha:
          return best_move
      return best_move
    else:
      best_move = 9999
      for move in self.board.legal_moves:
        self.board.push(move)
        best_move = min(best_move, self.__miniMax(depth - 1, alpha, beta, not is_maximising_player))
        self.board.pop()
        beta = min(beta, best_move)
        if beta <= alpha:
          return best_move
      return best_move

  def __evaluateBoard(self):
    total_evaluation = 0
    for i in range(64):
      total_evaluation = total_evaluation + self.__getPieceValue(self.board.piece_at(i), i)
    return total_evaluation

  def __getAbsoluteValue(self, piece: chess.Piece, is_white: bool, index: int) -> int:
    if piece.symbol().lower() == 'p':
      return 10 + (is_white if self.__pawnEvalWhite[index] else self.__pawnEvalBlack[index])
    elif piece.symbol().lower() == 'r':
      return 50 + (is_white if self.__rookEvalWhite[index] else self.__rookEvalBlack[index])
    elif piece.symbol().lower() == 'n':
      return 30 + self.__knightEval[index]
    elif piece.symbol().lower() == 'b':
      return 30 + (is_white if self.__bishopEvalWhite[index] else self.__bishopEvalBlack[index])
    elif piece.symbol().lower() == 'q':
      return 90 + self.__evalQueen[index]
    elif piece.symbol().lower() == 'k':
      return 900 + (is_white if self.__kingEvalWhite[index] else self.__kingEvalBlack[index])

  def __getPieceValue(self, piece: Optional[chess.Piece], index: int) -> int:
    if piece is None:
      return 0
    
    absolute_value = self.__getAbsoluteValue(piece, piece.color == chess.WHITE, index)
    return absolute_value if piece.color == chess.WHITE else -absolute_value

  def updateState(self, board_state):
    self.__current_board_state = board_state

  def state2Move(self, board_state: np.array) -> Optional[chess.Move]:
    """
    generate a `chess.Move` based on state matrices

    @return
    A valid `chess.Move` object
    """
    coordinates = []
    result = np.abs(self.__current_board_state - board_state)

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
    self.board.push(move)