# ---
# Quando for rodar isso, jogue dentro da src
# 
import time
import cv2
import imutils
import numpy as np
from model import *

def nothing(x):
  pass

cv2.namedWindow("Tracking")

# lower
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)

# upper
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

def drawContours(res, target, color):
  gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

  thresh = cv2.adaptiveThreshold(gray.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
  thresh = cv2.bitwise_not(thresh)

  # find contours
  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

  for cnt in cnts:
    # area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    biggest_cnt = cv2.approxPolyDP(cnt, 0.025 * peri, True)

    x, y, w, h = cv2.boundingRect(biggest_cnt)
    cv2.rectangle(target, (x,y), (x+w, y+h), color, 2)

camera = Camera(cam_address='http://192.168.0.111:4747/video')
calibration = ChessboardCalibration()
calibration.loadMapping()

while True:
  frame = camera.capture()
  frame = calibration.applyMapping(frame)

  l_h = cv2.getTrackbarPos("LH", "Tracking")
  l_s = cv2.getTrackbarPos("LS", "Tracking")
  l_v = cv2.getTrackbarPos("LV", "Tracking")
  print("config =====")
  print("lower => {}".format([l_h, l_s, l_v]))

  u_h = cv2.getTrackbarPos("UH", "Tracking")
  u_s = cv2.getTrackbarPos("US", "Tracking")
  u_v = cv2.getTrackbarPos("UV", "Tracking")
  print("upper => {}\n\n".format([u_h, u_s, u_v]))

  # selecionar as peças
  # ========
  image_final = frame.copy()
  inverted = cv2.bitwise_not(image_final)
  hsv = cv2.cvtColor(inverted, cv2.COLOR_BGR2HSV)

  lower = np.array([l_h, l_s, l_v])
  upper = np.array([u_h, u_s, u_v])
  mask = cv2.inRange(hsv.copy(), lower, upper)
  res = cv2.bitwise_and(image_final, image_final, mask=mask)

  drawContours(res, frame, (255,0,0))

  if cv2.waitKey(1) == ord("q"):
    break

  cv2.imshow('original frame', frame)
  cv2.imshow('mask', mask)
  cv2.imshow('resultado', res)