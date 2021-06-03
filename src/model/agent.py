from typing import Optional
import numpy as np
import chess
import chess.svg
import chess.pgn
import chess.polyglot
import chess.engine

class Agent:
  """
  Smart chess agent
  """
  __current_board_state: np.array = None
  __board: chess.Board = None
  __indexes = sorted(np.arange(1, 9), reverse=True)
  __columns = list("abcdefgh")
  __move_history: list[Optional[chess.Move]] = []

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

  __pawntable = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0]

  __knightstable = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
  ]

  __bishopstable = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
  ]

  __rookstable = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0
  ]

  __queenstable = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20
  ]

  __kingstable = [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30
  ]

  def __init__(self, board):
    self.__current_board_state = self.___initial_board_state
    self.__board = board

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

  def __evaluateBoard(self):
    if self.__board.is_checkmate():
      if self.__board.turn:
        return -9999
      else:
        return 9999
    if self.__board.is_stalemate():
      return 0
    if self.__board.is_insufficient_material():
      return 0

    wp = len(self.__board.pieces(chess.PAWN, chess.WHITE))
    bp = len(self.__board.pieces(chess.PAWN, chess.BLACK))
    wn = len(self.__board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(self.__board.pieces(chess.KNIGHT, chess.BLACK))
    wb = len(self.__board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(self.__board.pieces(chess.BISHOP, chess.BLACK))
    wr = len(self.__board.pieces(chess.ROOK, chess.WHITE))
    br = len(self.__board.pieces(chess.ROOK, chess.BLACK))
    wq = len(self.__board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(self.__board.pieces(chess.QUEEN, chess.BLACK))

    material = 100 * (wp - bp) + 320 * (wn - bn) + 330 * (wb - bb) + 500 * (wr - br) + 900 * (wq - bq)

    pawnsq = sum([self.__pawntable[i] for i in self.__board.pieces(chess.PAWN, chess.WHITE)])
    pawnsq = pawnsq + sum([-self.__pawntable[chess.square_mirror(i)] for i in self.__board.pieces(chess.PAWN, chess.BLACK)])
    knightsq = sum([self.__knightstable[i] for i in self.__board.pieces(chess.KNIGHT, chess.WHITE)])
    knightsq = knightsq + sum([-self.__knightstable[chess.square_mirror(i)] for i in self.__board.pieces(chess.KNIGHT, chess.BLACK)])
    bishopsq = sum([self.__bishopstable[i] for i in self.__board.pieces(chess.BISHOP, chess.WHITE)])
    bishopsq = bishopsq + sum([-self.__bishopstable[chess.square_mirror(i)] for i in self.__board.pieces(chess.BISHOP, chess.BLACK)])
    rooksq = sum([self.__rookstable[i] for i in self.__board.pieces(chess.ROOK, chess.WHITE)])
    rooksq = rooksq + sum([-self.__rookstable[chess.square_mirror(i)] for i in self.__board.pieces(chess.ROOK, chess.BLACK)])
    queensq = sum([self.__queenstable[i] for i in self.__board.pieces(chess.QUEEN, chess.WHITE)])
    queensq = queensq + sum([-self.__queenstable[chess.square_mirror(i)] for i in self.__board.pieces(chess.QUEEN, chess.BLACK)])
    kingsq = sum([self.__kingstable[i] for i in self.__board.pieces(chess.KING, chess.WHITE)])
    kingsq = kingsq + sum([-self.__kingstable[chess.square_mirror(i)] for i in self.__board.pieces(chess.KING, chess.BLACK)])

    eval = material + pawnsq + knightsq + bishopsq + rooksq + queensq + kingsq
    if self.__board.turn:
      return eval
    else:
      return -eval

  def __alphaBeta(self, alpha, beta, depthleft):
    bestscore = -9999
    if (depthleft == 0):
      return self.__quiesce(alpha, beta)
    for move in self.__board.legal_moves:
      self.makeMove(move)
      score = -self.__alphaBeta(-beta, -alpha, depthleft - 1)
      self.__board.pop()
      if (score >= beta):
        return score
      if (score > bestscore):
        bestscore = score
      if (score > alpha):
        alpha = score
    return bestscore

  def __quiesce(self, alpha, beta ):
    stand_pat = self.__evaluateBoard()
    if( stand_pat >= beta ):
      return beta
    if( alpha < stand_pat ):
      alpha = stand_pat

    for move in self.__board.legal_moves:
      if self.__board.is_capture(move):
        self.makeMove(move)
        score = -self.__quiesce(-beta, -alpha )
        self.__board.pop()

        if( score >= beta ):
          return beta
        if( score > alpha ):
          alpha = score
    return alpha

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

  def selectMove(self, depth: int = 3):
    bestMove = chess.Move.null()
    bestValue = -99999
    alpha = -100000
    beta = 100000

    for move in self.__board.legal_moves:
      self.makeMove(move)
      boardValue = -self.__alphaBeta(-beta, -alpha, depth-1)
      if boardValue > bestValue:
        bestValue = boardValue
        bestMove = move
      if( boardValue > alpha ):
        alpha = boardValue
      self.__board.pop()

    self.__move_history.append(bestMove)
    return bestMove