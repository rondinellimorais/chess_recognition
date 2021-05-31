import numpy as np

class Agent:
  """
  Smart chess agent
  """
  __current_board_state: np.array = None

  def __init__(self):
    pass

  def setState(self, board_state):
    print(board_state)
    if self.__current_board_state is None:
      self.__current_board_state = board_state
    else:
      # check if state change
      result = np.abs(self.__current_board_state - board_state)

      # save new state
      self.__current_board_state = board_state

      coordinates = []
      index = np.arange(1, 9)
      columns = list("abcdefgh")

      for ridx, row in enumerate(result):
        for cidx, col in enumerate(row):
          if col > 0:
            coordinates.append('{}{}'.format(columns[cidx], index[ridx]))


      # Paramos aqui...
      # o estado do tabuleiro está oscilando demais, precisamos verificar
      # uma forma de tornar a deteccao mais estável

      print(coordinates)