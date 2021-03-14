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
    weights_path = 'assets/dnn/yolov4_best_v2.weights'

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

  def predict(self, img=None, size=(416, 416), thresh=0.35, nms_threshold=0.6, draw_and_save=False) -> list:
    """
    Deep neural network predict

    Params
    ---
    `img` image file to running darknet prediction

    `size` width and height of network size. Default is `416x416`
    
    `thresh` a threshold used in darknet. Default is `0.35`
    
    `nms_threshold` a threshold used in non maximum suppression. Default is `0.6`
    
    `draw_and_save` a flag to debug predictions, when `True` the predictions is saved as a file. Default is `False`
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
    
    # to debug
    if draw_and_save and len(detections) > 0:
      self.__save(img, detections)

    return detections

  def __save(self, img, detections):
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(self.__labels), 3), dtype="uint8")

    file_content = []
    for (name, bounding_box, accuracy, class_id) in detections:
      color = [int(c) for c in COLORS[class_id]]
      x,y,w,h = bounding_box
      cv2.rectangle(img, (x, y), (w, h), color, 2)
      cv2.putText(img, str(class_id), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
      file_content.append("{}\t| {}\t| {:.4f}".format(class_id, name, accuracy))
  
    # save prediction image
    cv2.imwrite('debug/predictions.jpg', img)

    # save prediction logs
    names = list(map(lambda p: p[0], detections))
    classes = list(dict.fromkeys(names))
    metrics = list(map(lambda c: "{} | {}".format(str(names.count(c)), c), classes))
    with open('debug/predictions.log', 'w') as txt_file:
      txt_file.write('PREDICTIONS\n=================\n')
      txt_file.write('\n'.join(file_content))
      txt_file.write('\n\n\n')
      txt_file.write('METRICS\n=================\n')
      txt_file.write('\n'.join(metrics))