# Usamos o código abaixo para obter as jogadas de xadrez do 
# aplicativo de xadrez do macOS
import time
from PIL import ImageGrab
import cv2
import numpy as np

class Screen:
  previousMillis: int = 0
  counter: int = 0
  old_print_srceen = None

  def print(self):
    return ImageGrab.grab()

  def imsave(self, snapshot):
    self.counter += 1
    save_path = "/Volumes/ROND/chess/jogadas_xadrez/02/JOGADA_{}.png".format(self.counter)
    snapshot.save(save_path)

  def start(self):
    while True:
      currentMillis = round(time.time() * 1000)
      if currentMillis - self.previousMillis >= 2000: # 2s
        self.previousMillis = currentMillis
        
        # get print screen
        snapshot = self.print()

        # compare images
        if self.old_print_srceen is not None:
          print_srceen = cv2.cvtColor(np.array(snapshot), cv2.COLOR_BGRA2BGR)
          difference = cv2.subtract(print_srceen, self.old_print_srceen)
          b, g, r = cv2.split(difference)

          if cv2.countNonZero(b) != 0 and cv2.countNonZero(g) != 0 and cv2.countNonZero(r) != 0:
            print("Imagens diferentes salva")
            self.imsave(snapshot)
            self.old_print_srceen = print_srceen
        else:
          self.imsave(snapshot)
          self.old_print_srceen = cv2.cvtColor(np.array(snapshot), cv2.COLOR_BGRA2BGR)

Screen().start()