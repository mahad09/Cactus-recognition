import os
import cv2
import shutil
import random
import pdb
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, MobileNet
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Activation, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm



TRAINING_DATASET_PATH = 'dataset/training_set/training_set/'
VALIDATION_DATASET_PATH = 'dataset/validation_set/validation_set/'
BATCH_SIZE = 64
NUMBER_OF_CLASSES = 2
INPUT_SHAPE = (64, 64)
EPOCHS=15


def data_generators():
  print('defining generators of training and validation sets...')
  datagen = ImageDataGenerator(rescale=1./255)

  train_generator = datagen.flow_from_directory(
    TRAINING_DATASET_PATH, target_size=INPUT_SHAPE, batch_size=BATCH_SIZE, class_mode='binary')

  validation_generator = datagen.flow_from_directory(
    VALIDATION_DATASET_PATH, target_size=INPUT_SHAPE, batch_size=BATCH_SIZE, class_mode='binary')

  return train_generator, validation_generator


def model_architecture_compilation():
  print('model compilation...')
  vgg16 = MobileNet(weights = 'imagenet', 
              include_top = False, 
              input_shape = (64, 64, 3))

  for layer in vgg16.layers:
    layer.trainable = False

  model = keras.models.Sequential()
  model.add(vgg16)
  model.add(Flatten())
  model.add(Dense(256, activation = 'relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  model.add(Dense(128, activation = 'relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  model.add(Dense(1, activation = 'sigmoid'))

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  return model


def model_training(train_generator, validation_generator, model):
  print('model training...')
  history = model.fit(
      train_generator,
      steps_per_epoch = train_generator.samples // BATCH_SIZE,
      validation_data = validation_generator, 
      validation_steps = validation_generator.samples // BATCH_SIZE,
      epochs = EPOCHS,
      workers=4)

  model.save('transfer_learning_stats/retrained_model.h5')

  return history, model


def model_evaluation(validation_generator):
  trained_model = keras.models.load_model('transfer_learning_stats/retrained_model.h5')
  print(validation_generator.class_indices)
  Y_pred = trained_model.predict(validation_generator)
  y_pred = np.argmax(Y_pred, axis=1)

  testing = []
  counter = 0
  actual = []
  for folder in tqdm(os.listdir(VALIDATION_DATASET_PATH)):
    for image in os.listdir(VALIDATION_DATASET_PATH + folder + '/'):
      img = cv2.imread(VALIDATION_DATASET_PATH + folder + '/' + image)
      img = cv2.resize(img, (64, 64))
      img = np.array(img)
      img = img.reshape(1, 64, 64, 3)
      # img = img.flatten()
      img *= 255
      predict = trained_model.predict(img)
      # print(counter, 'actual: ', validation_generator.class_indices[folder],'  ', 'predicted: ', np.argmax(predict, axis=1))
      testing.append(np.argmax(predict))
      actual.append(validation_generator.class_indices[folder])
      counter += 1
    print(counter)

  cf_matrix(actual, testing)
  cls_report(actual, testing)


def cf_matrix(predicted_labels, correct_labels):
  cf_matrix = confusion_matrix(predicted_labels, correct_labels)
  fig, ax = plt.subplots(figsize=(8, 6))
  sns.heatmap(cf_matrix, cmap="YlGnBu", annot=True, linewidths=.5, ax=ax, fmt=".0f")
  plt.show()

def cls_report(predicted_labels, correct_labels):
  print(classification_report(predicted_labels, correct_labels))


def plot_accuracy(history):
    plt.title("Accuracy Graph")
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')
    plt.show()


def plot_loss(history):
    plt.title("Loss Graph")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'validation_loss'], loc='best')
    plt.show()




train_generator, validation_generator = data_generators()
# model = model_architecture_compilation()
# history, trained_model = model_training(train_generator, validation_generator, model)
model_evaluation(validation_generator)
# plot_accuracy(history)
# plot_loss(history)
