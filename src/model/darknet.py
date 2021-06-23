import cv2
import numpy as np
from cv2.dnn import readNetFromDarknet, blobFromImage
from dotenv import dotenv_values

class Darknet:

  __instance = None
  __net = None
  __out_layers = None
  __labels = []

  def __init__(self):
    config = dotenv_values()

    # load model
    self.__net = readNetFromDarknet(config.get('CFG_FILE_PATH'), config.get('WEIGHTS_FILE_PATH'))
    
    # only GPU
    # self.__net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # self.__net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

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

  def predict(self, img=None, size=(416, 416), thresh=0.35, nms_threshold=0.6) -> list:
    """
    Deep neural network predict

    Params
    ---
    `img` image file to running darknet prediction

    `size` width and height of network size. Default is `416x416`
    
    `thresh` a threshold used in darknet. Default is `0.35`
    
    `nms_threshold` a threshold used in non maximum suppression. Default is `0.6`
    """
    if img is None:
      raise Exception('img cannot be null')

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = blobFromImage(img, 1/255.0, size=size, swapRB=True, crop=False)
    self.__net.setInput(blob)

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    detections = []

    # grab input image dimensions
    (H, W) = img.shape[:2]

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
          # scale the bounding box coordinates back relative to the
          # size of the image, keeping in mind that YOLO actually
          # returns the center (x, y)-coordinates of the bounding
          # box followed by the boxes' width and height
          box = detection[0:4] * np.array([W, H, W, H])
          (centerX, centerY, width, height) = box.astype("int")

          # use the center (x, y)-coordinates to derive the top and
          # and left corner of the bounding box
          x = int(centerX - (width / 2))
          y = int(centerY - (height / 2))

          # update our list of bounding box coordinates, confidences,
          # and class IDs
          boxes.append([x, y, int(width), int(height)])
          confidences.append(float(confidence))
          classIDs.append(classID)
    
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, thresh, nms_threshold)

    if len(idxs) > 0:
      for i in idxs.flatten():
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        detections.append((
          self.__labels[classIDs[i]],
          (x, y, x+w, y+h),
          confidences[i],
          classIDs[i]
        ))

    return detections