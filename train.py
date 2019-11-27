import dataset
from model import model


step_size_train = dataset.train_generator.n // dataset.train_generator.batch_size
epoch = 5

model.fit_generator(generator=dataset.train_generator, steps_per_epoch=step_size_train,epochs=epoch)

model.save('model.h5')