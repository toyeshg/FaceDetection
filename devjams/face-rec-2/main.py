from pathlib import Path
from time import sleep
from cv2 import VideoCapture, imwrite, imshow, waitKey, destroyAllWindows
from shutil import rmtree

IMAGES_FOLDER = Path('data') / 'images'
rmtree(IMAGES_FOLDER)
IMAGES_FOLDER.mkdir()
NUMBER_OF_IMAGES = 5

capturer = VideoCapture(0)

for image_number in range(NUMBER_OF_IMAGES):
  return_value, frame = capturer.read()

  imwrite(str(IMAGES_FOLDER / (str(image_number) + '.jpg')), frame)
  sleep(0.5)

  if waitKey(1) & 0xFF == ord('q'):
    break

capturer.release()
destroyAllWindows()