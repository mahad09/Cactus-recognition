import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import datetime
import pdb


DATA_DIR = 'dataset/training_set/training_set/'
TEST_DIR = 'dataset/validation_set/validation_set/'

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 64
EPOCHS = 10
NUM_OF_CLASSES = 2


def data_loading_and_preprocessing():
  train_data = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
  )

  test_data = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
  )

  # class_names = train_ds.class_names
  # print(class_names)

  return train_data, test_data


def custom_CNN():
  model = keras.Sequential([
    keras.layers.experimental.preprocessing.Rescaling(1./255),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(NUM_OF_CLASSES, activation="softmax"),
    ]
  )

  return model


def model_compilation(model):
  model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
  )

  return model


def model_training(model, train_data, test_data):

  log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard = keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_images=True
  )

  history = model.fit(train_data, validation_data=test_data, epochs=EPOCHS, callbacks=[tensorboard])
  model.save('custom_model_stats/cactus_classifier.h5')

  return history, model


def model_evaluation(test_data):
  trained_model = keras.models.load_model('custom_model_stats/cactus_classifier.h5')
  overall_result = trained_model.evaluate(test_data)
  print(dict(zip(trained_model.metrics_names, overall_result)))

  y_pred = []  # store predicted labels
  y_true = []  # store true labels

  # iterate over the test dataset
  for image_batch, label_batch in test_data:   # use dataset.unbatch() with repeat
    # append true labels
    y_true.append(label_batch)
    # compute predictions
    preds = trained_model.predict(image_batch)
    # append predicted labels
    y_pred.append(np.argmax(preds, axis = - 1))

  # convert the true and predicted labels into tensors
  correct_labels = tf.concat([item for item in y_true], axis = 0)
  predicted_labels = tf.concat([item for item in y_pred], axis = 0)

  cf_matrix(predicted_labels, correct_labels)
  cls_report(predicted_labels, correct_labels)


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



train_data, test_data = data_loading_and_preprocessing()
model = custom_CNN()
compiled_model = model_compilation(model)
history, trained_model = model_training(model, train_data, test_data)
predictions = model_evaluation(test_data)
plot_accuracy(history)
plot_loss(history)




  


