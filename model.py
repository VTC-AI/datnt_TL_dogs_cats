from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
import keras



# import mobinet model
base_model = MobileNetV2(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)

preds = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)

for layer in model.layers[:20]:
  layer.trainable = False

for layer in model.layers[20:]:
  layer.trainable = True

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
