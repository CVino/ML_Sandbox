
# Load Libraries
import tensorflow as tf
import pandas
import numpy as np
import os
import matplotlib.pyplot as plt
import string
import re
import config
from PyQt5 import QtCore, QtGui, QtWidgets

#Function to Standardize Text and strip out HTML code
@tf.keras.utils.register_keras_serializable()
def custom_standarization(input_data):
  lowercase_data = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase_data,"<*>", " ")
  return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

if config.Generate_Model == True:
  #Prepare Train, Validate and Test Datasets
  raw_train_dataset = tf.keras.utils.text_dataset_from_directory(
      os.path.dirname(os.path.abspath(__file__))+'/DataSet_Imdb/train/',
      batch_size = 32,
      validation_split = 0.2,
      subset = 'training',
      seed = 42)

  if config.Preview_Reviews == True:
    for text_batch, label_batch in raw_train_dataset.take(1):
      for i in range(3):
        print("Review", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])

  raw_val_dataset = tf.keras.utils.text_dataset_from_directory(
      os.path.dirname(os.path.abspath(__file__))+'/DataSet_Imdb/train/',
      batch_size = 32,
      validation_split = 0.2,
      subset = 'validation',
      seed = 42)

  raw_test_dataset = tf.keras.utils.text_dataset_from_directory(
      os.path.dirname(os.path.abspath(__file__))+'/DataSet_Imdb/test/',
      batch_size = 32)

  #Create Vectorization Layer
  vectorize_layer = tf.keras.layers.TextVectorization(
    standardize = custom_standarization,
    max_tokens = 10000,
    output_mode = 'int',
    output_sequence_length = 250)

  #Setup vectorization Lazer
  train_text_dataset = raw_train_dataset.map(lambda x , y : x)
  vectorize_layer.adapt(train_text_dataset)

  #Preview vectorized text
  text_batch, label_batch = next(iter(raw_train_dataset))
  first_raw_review, first_label = text_batch[0], label_batch[0]
  print("Review: " + first_raw_review)
  print("Label: " + raw_train_dataset.class_names[first_label])
  print("Vectorized Text : ", vectorize_layer(first_raw_review))

  #print("1 ---> ",vectorize_layer.get_vocabulary()[566])

  #Configure cache and prefetch to optimize performance
  Autotune = tf.data.AUTOTUNE

  raw_test_dataset = raw_test_dataset.cache().prefetch(buffer_size = Autotune)
  raw_train_dataset = raw_train_dataset.cache().prefetch(buffer_size = Autotune)
  raw_val_dataset = raw_val_dataset.cache().prefetch(buffer_size = Autotune)

  #Define the NN Model
  Embedded_Dimension = 16

  checkpoint_dir = os.path.dirname(config.Checkpoint_Path)

  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = config.Checkpoint_Path,
                                                      save_weights_only=True,
                                                      verbose=1)

  NN_Model = tf.keras.Sequential([
    vectorize_layer,
    tf.keras.layers.Embedding(10000, Embedded_Dimension),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)])

  #Compile model
  NN_Model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer = 'adam',
    metrics = tf.metrics.BinaryAccuracy(threshold=0.0))

  history = NN_Model.fit(raw_train_dataset,
              validation_data = raw_val_dataset,
              epochs = 10,
              callbacks = [cp_callback])
  
  Final_Model = tf.keras.Sequential([
    NN_Model,
    tf.keras.layers.Activation('sigmoid')
  ])

  Final_Model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer="adam",
    metrics=['accuracy'])

  Final_Model.save("Text_Classification/"+config.Model_Name)

  #Plot accuracy and loss
  history_dict = history.history
  acc = history_dict['binary_accuracy']
  val_acc = history_dict['val_binary_accuracy']
  loss = history_dict['loss']
  val_loss = history_dict['val_loss']

  epochs = range(1, len(acc) + 1)

  # "bo" is for "blue dot"
  plt.plot(epochs, loss, 'bo', label='Training loss')
  # b is for "solid blue line"
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  plt.show()

  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(loc='lower right')

  plt.show()



else:
    #Load Model
    Final_Model = tf.keras.models.load_model("Text_Classification/"+config.Model_Name)

if config.Evaluate_Model == True:
  model_loss, model_acc = Final_Model.evaluate(raw_test_dataset)

if config.Test_Model == True:
  Test_Result = Final_Model.predict(config.Test_Text)
  print("The Rate of your evaluation is: {:.2f}".format(Test_Result[0][0]))

  if Test_Result[0][0] < 0.4:
    Result_String = "You did not like the movie :("

  elif Test_Result[0][0] > 0.6:
    Result_String = "You liked the movie :)"

  else:
    Result_String = "Your review is in between good and bad."

  app = QtWidgets.QApplication([])
  Result_Window = QtWidgets.QMessageBox()
  Result_Window.setWindowTitle("Review evaluation result")
  Result_Window.setText("Your Review:  " + config.Test_Text[0] + "\r\r" + "Your Rate Score: " + str(Test_Result[0][0]) + "\r\r" + Result_String)
  
  Result_Window.exec()