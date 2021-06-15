import sys
sys.path.append('/Users/rondinellimorais/Desktop/projetos/chess_recognition/src')

import time
import cv2
import imutils
import numpy as np
from model import *

def adjust_gamma(image, gamma=1.0):
  invGamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
  return cv2.LUT(image, table)

def drawPiecesBoundingBoxes(res, target, color):
  gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

  thresh = cv2.adaptiveThreshold(gray.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
  thresh = cv2.bitwise_not(thresh)

  # find contours
  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

  boxes = []
  confidences = []

  for cnt in cnts:
    area = cv2.contourArea(cnt)
    if area > 200:
      peri = cv2.arcLength(cnt, True)
      biggest_cnt = cv2.approxPolyDP(cnt, 0.025 * peri, True)
      x, y, w, h = cv2.boundingRect(biggest_cnt)
      boxes.append([x, y, int(x+w), int(y+h)])
      confidences.append(float(0.6))

  # apply non-maxima suppression to suppress weak, overlapping bounding
  # boxes
  idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.35, 0.7)
  if len(idxs) > 0:
    for i in idxs.flatten():
      (x, y) = (boxes[i][0], boxes[i][1])
      (w, h) = (boxes[i][2], boxes[i][3])
      cv2.rectangle(target, (x,y), (w, h), color, 2)

camera = Camera(cam_address='/Volumes/ROND/chess/video/fake_cam.mp4')
calibration = ChessboardCalibration()
calibration.loadMapping()


while True:
  original_frame = camera.capture()
  original_frame = calibration.applyMapping(original_frame)

  frame = original_frame.copy()
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # to RGB

  lower = np.array([0, 0, 0])
  upper = np.array([65, 89, 255])
  mask = cv2.inRange(frame, lower, upper)
  res = cv2.bitwise_and(frame, frame, mask=mask)
  cv2.imshow('res', res)

  # drawPiecesBoundingBoxes(res, original_frame, (0,255,0))
  # cv2.imshow('original_frame', original_frame)

  if cv2.waitKey(1) == ord("q"):
    break