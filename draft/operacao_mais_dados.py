# Aqui vamos tentar criar mais dado para o darknet.
# Inves de fotograr tudo e anotar manualmente, vamos utilizar
# a propria rede para gerar os dados e mais tarde corrigir o que deu
# errado
#
# **Editado**
# Funcionou muito bem essa operação, o modelo está mais preciso
# embora ainda esteja errando em alguns casos mas é evidênte a melhora
# que teve.
#
# Isso mostra que realmente nosso problema era quantidade de dados e variações.
#
# Vou jogar esse arquivo dentro do diretório `draft` mas se quiser roda-lo novamente
# precisa jogar ele na raiz da `src`
# 
from model import ChessboardCalibration, Darknet, Camera
import cv2
import time

darknet = Darknet.instance()

def cria_dado(img, network_size, filename, folder):
  detections = darknet.predict(img=img, size=network_size, thresh=0.8)

  im_height, im_width, _ = img.shape
  cv2.imwrite('data/{}/{}.jpg'.format(folder, filename), img)

  with open('data/{}/{}.txt'.format(folder, filename), 'w') as txt_file:
    for (name, bbox, _, class_id) in detections:
      w = bbox[2] - bbox[0]
      h = bbox[3] - bbox[1]
      abs_x = bbox[0] + (w / 2)
      abs_y = bbox[1] + (h / 2)

      center_x = abs_x/im_width
      center_y = abs_y/im_height
      width = w/im_width
      height = h/im_height

      txt_file.write("{} {} {} {} {}\n".format(class_id, center_x, center_y, width, height))

# run
def didCaptureFrame(frame, camera):
  # Aqui vai da todos os erros de escopo possível
  # pois didCaptureFrame está em outro escopo
  # por isso as gambis abaixo
  
  identifier = time.time()
  full_img = frame
  cropped_img = chessboard_calibration.applyMapping(full_img) # 416 × 416

  cria_dado(
    img=cropped_img,
    network_size=(416, 416),
    filename='{}'.format(identifier),
    folder='cropped'
  )

  camera.stopRunning()
  input('Press any key to continue...')
  camera = Camera('http://192.168.0.109:4747/video', fps=1)
  camera.startRunning(didCaptureFrame)

chessboard_calibration = ChessboardCalibration()
found, _ = chessboard_calibration.loadMapping()
if found:
  camera = Camera('http://192.168.0.109:4747/video', fps=1)
  camera.startRunning(didCaptureFrame)