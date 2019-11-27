from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from os import listdir
from os.path import isfile, join
import numpy as np
import dataset

w, h = 224, 224

model = load_model('model.h5')
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

my_path = 'prediction_target/'

only_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]

for f in only_files:
  img = image.load_img(my_path + f, target_size=(w, h))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  print(f, model.predict(x))
  # dataset.train_generator.class_indices