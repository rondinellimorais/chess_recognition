import cv2
import numpy as np
from cv2.dnn import readNetFromDarknet, blobFromImage

class Darknet:

  __instance = None
  __net = None
  __out_layers = None
  __labels = []

  def __init__(self):
    config_path = 'assets/dnn/yolov4.custom.cfg'
    weights_path = 'assets/dnn/yolov4_best.weights'

    # load model
    self.__net = readNetFromDarknet(config_path, weights_path)

    # determine only the *output* layer names that we need from YOLO
    self.__out_layers = self.__net.getLayerNames()
    self.__out_layers = [self.__out_layers[i[0] - 1] for i in self.__net.getUnconnectedOutLayers()]

    # load our labels
    self.__labels = self.__loadLabels()

  @classmethod
  def instance(cls):
    if cls.__instance is None:
      cls.__instance = cls()
    return cls.__instance

  def __loadLabels(self):
    """
    docstring
    """
    labels_path = 'assets/dnn/data.names'
    return open(labels_path).read().strip().split('\n')

  def predict(self, img=None, size=(416, 416), thresh=0.35) -> tuple:
    """
    Deep neural network predict
    """
    if img is None:
      raise Exception('img cannot be null')

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = blobFromImage(img, 1/255.0, size=size, swapRB=True, crop=False)
    self.__net.setInput(blob)

    # initialize our lists of confidences and class IDs
    confidences = []
    classIDs = []

    for output in self.__net.forward(self.__out_layers):
      for detection in output:
        # extract the class ID and confidence (i.e., probability) of
        # the current object detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > thresh:
          confidences.append(float(confidence))
          classIDs.append(classID)

    if len(confidences) != 0:
      max_conf_idx = np.argmax(confidences)
      class_idx = classIDs[np.argmax(confidences)]
      return (True, self.__labels[class_idx], confidences[max_conf_idx])
    return (False, None, None)