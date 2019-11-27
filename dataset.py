from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import preprocess_input


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
  './train',
  target_size=(224, 224),
  color_mode='rgb',
  batch_size=32,
  class_mode='categorical',
  shuffle=True
)