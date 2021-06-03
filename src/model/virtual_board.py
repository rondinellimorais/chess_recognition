from typing import Optional, Tuple
import chess
import numpy as np

class VirtualBoard(chess.Board):
  # these values are based on the computer vision
  # project ids (darknet data.names)
  __piece_values_dict = {
    "P": 0,
    "R": 1,
    "B": 2,
    "N": 3,
    "K": 4,
    "Q": 5,
    "p": 6,
    "r": 7,
    "b": 8,
    "k": 9,
    "q": 10,
    "n": 11
  }

  def __init__(self, **kwargs) -> None:
    super().__init__(**kwargs)

  def state(self) -> np.array:
    piece_map = np.ones(64) * -1
    for square, piece in zip(self.piece_map().keys(), self.piece_map().values()):
      piece_map[square] = self.__piece_values_dict[piece.symbol()]
    
    board_state = piece_map.reshape((8, 8)).astype(np.int32)
    return np.flip(board_state, axis=0)