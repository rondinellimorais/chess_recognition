import time
import cv2

from model.camera import Camera

vidwrite = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, (1920, 1080))
camera = Camera('http://192.168.0.111:4747/video', resize_percent=0)

while True:
  frame = camera.capture()
  vidwrite.write(frame)

  cv2.imshow('as', frame)
  if cv2.waitKey(1) == ord('q'):
    break

vidwrite.release()
camera.destroy()
cv2.destroyAllWindows()