import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Lambda, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from load_data import upload_dataset

PATH_TO_CONTENT = './model/content/'  # by default data is loaded here
TRAIN_DIR = PATH_TO_CONTENT + 'seg_train/seg_train/'
VAL_DIR = PATH_TO_CONTENT + 'seg_test/seg_test/'
DOWNLOAD_DATA = False
PRINT_INFO_MODEL = True


def train():
    # Download data if not already done
    if DOWNLOAD_DATA:
        upload_dataset(path_=PATH_TO_CONTENT)

    class_folder_paths = [PATH_TO_CONTENT + 'seg_train/seg_train/' + x for x in
                          os.listdir(PATH_TO_CONTENT + 'seg_train/seg_train/')]

    # Print info about amount of imgs in each class
    for class_folder_path in class_folder_paths:
        print('{0}: '.format(class_folder_path), len(os.listdir(class_folder_path)))

    # Create data generators
    train_generator, val_generator = create_data_gen()

    # Create labels
    labels = train_generator.class_indices
    class_mapping = dict((v, k) for k, v in labels.items())

    # Create model
    model = create_model()
    opt = Adam(learning_rate=0.00001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.build(((None, 150, 150, 3)))

    if PRINT_INFO_MODEL:
        model.summary()

    # Train
    train_cb = ModelCheckpoint('model/temp/', save_best_only=True)
    model.fit(train_generator, validation_data=val_generator, callbacks=[train_cb], epochs=1)

    # Convert and save model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)


def create_data_gen():
    train_data_gen = ImageDataGenerator(horizontal_flip=True)
    train_generator = train_data_gen.flow_from_directory(TRAIN_DIR,
                                                         target_size=(150, 150),
                                                         color_mode='rgb',
                                                         batch_size=32,
                                                         class_mode='categorical',
                                                         shuffle=True)
    val_data_gen = ImageDataGenerator()
    val_generator = val_data_gen.flow_from_directory(VAL_DIR,
                                                     target_size=(150, 150),
                                                     color_mode='rgb',
                                                     batch_size=32,
                                                     class_mode='categorical',
                                                     shuffle=False)
    return train_generator, val_generator


def create_model():
    before_mobilenet = Sequential([Input((150, 150, 3)),
                                   Lambda(preprocess_input)])

    mobilenet = MobileNetV2(input_shape=(150, 150, 3), include_top=False)

    after_mobilente = Sequential([GlobalAveragePooling2D(),
                                  Dropout(0.3),
                                  Dense(6, activation='softmax')])

    model = Sequential([before_mobilenet, mobilenet, after_mobilente])
    return model


# For test
if __name__ == "__main__":
    train()
