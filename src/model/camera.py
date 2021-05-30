import cv2
import time

class Camera:
  cap: cv2.VideoCapture
  resize_percent: float
  fps: int
  resolution: tuple
  previousMillis: int = 0
  stop: bool

  def __init__(self, cam_address, resize_percent=0.35, fps=30):
    """
    Initialize a `Camera` object

    @param cam_address The path of the camera can number, video file name or ip address
    @param resize_percent The percent of resize frame
    @param fps The number of frames that can get in 1 second
    """
    self.resize_percent = resize_percent
    self.fps = fps
    
    self.cap = cv2.VideoCapture(cam_address)
    if self.cap.isOpened():
      width = self.cap.get(cv2.cv2.CAP_PROP_FRAME_WIDTH)
      height = self.cap.get(cv2.cv2.CAP_PROP_FRAME_HEIGHT)
      self.resolution = (height, width)
    else:
      raise Exception('Could not open camera.')

  def startRunning(self, callback):
    """
    Running a loop getting the next frame of the video camera

    @param callback The return function where frame captured is sent

    **OBS**
    This method lock execution the nexts lines of the caller
    """
    self.stop = False
    while True:
      currentMillis = round(time.time() * 1000)

      # check to see if it's time to capture the frame; that is, if the difference
      # between the current time and last time we capture the frame is bigger than
      # the interval at which we want to capture the frame.
      # The interval is a 1s divide by number of frames
      if currentMillis - self.previousMillis >= (1000/self.fps):
        self.previousMillis = currentMillis

        # capture the next frame
        frame = self.capture()
        callback(frame, self)
      
      # check to see if it's time to break
      if self.stop:
        break

  def capture(self):
    """
    Return a single next frame of the capture camera
    """
    try:
      retval, frame = self.cap.read()

      if not retval:
        raise Exception('Could not get the next video frame.')

      # define new resolution
      height, width = self.resolution
      new_w = int(width - (self.resize_percent * width))
      new_h = int(height - (self.resize_percent * height))

      frame = cv2.resize(frame, (new_w, new_h))
      return frame
    except:
      self.stopRunning()
      raise Exception('Could not get the next video frame.')

  def stopRunning(self):
    """
    Stop running and clean the session
    """
    self.stop = True
    if self.cap:
      self.destroy()

  def destroy(self):
    """
    Release the camera
    """
    self.cap.release()