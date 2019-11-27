from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from os import listdir
from os.path import isfile, join
import numpy as np
import dataset
import cv2

w, h = 224, 224

model = load_model('model.h5')
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

my_path = 'prediction_target/'

only_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]

name_of_class = dataset.name_of_class

# for f in only_files:
#   img = image.load_img(my_path + f, target_size=(w, h))
#   x = image.img_to_array(img)
#   x = np.expand_dims(x, axis=0)

#   images = np.vstack([x])
#   print(f, model.predict(x))


def preprocess_image(img):
  if (img.shape[0] != 224 or img.shape[1] != 224):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
    img = (img/127.5)
    img = img - 1
    img = np.expand_dims(img, axis=0)
  return img
  
img = cv2.imread('prediction_target/horse1.jpg')
pred = model.predict(preprocess_image(img))
result = name_of_class[np.argmax(pred)]
print(result)
