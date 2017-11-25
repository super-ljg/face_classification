import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import plot_model
from shufflenet import ShuffleNet
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
import numpy as np


def preprocess(x):
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x /= 255.0
    x -= 0.5
    x *= 2.0
    return x


if __name__ == '__main__':
    groups = 3
    batch_size = 64
    inital_epoch = 0
    ds = '/home/ljg/Desktop'

    model = ShuffleNet(groups=groups, pooling='avg')
    #plot_model(model, 'model.png', show_shapes=True)
    # model.load_weights('%s.hdf5' % model.name, by_name=True)
    csv_logger = CSVLogger('%s.log' % model.name, append=(inital_epoch is not 0))
    checkpoint = ModelCheckpoint(filepath='5000_{val_acc:.4f}.hdf5', verbose=1,
                                 save_best_only=True, monitor='val_acc', mode='max')

    learn_rates = [0.05, 0.01, 0.005, 0.001, 0.0005]
    lr_scheduler = LearningRateScheduler(lambda epoch: learn_rates[epoch // 30])

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess,
                                       zoom_range=0.1,
                                       width_shift_range=0.05,
                                       height_shift_range=0.05,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess)

    train_generator = train_datagen.flow_from_directory(
            '%s/train5000/' % ds,
            target_size=(224, 224),
            batch_size=batch_size)

    test_generator = test_datagen.flow_from_directory(
            '%s/val5000/' % ds,
            target_size=(224, 224),
            batch_size=batch_size)

    model.compile(
              optimizer= keras.optimizers.SGD(lr=.1, decay=5e-4, momentum=0.9),
              metrics=['accuracy'],
              loss='categorical_crossentropy')
    model.summary()
    model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=400,
            workers=8,
            initial_epoch=inital_epoch,
            use_multiprocessing=True,
            validation_data=test_generator,
            validation_steps=test_generator.samples // batch_size,
            callbacks=[csv_logger, checkpoint, lr_scheduler])
