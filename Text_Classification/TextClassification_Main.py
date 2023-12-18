
# Load Libraries
import tensorflow as tf
import pandas
import numpy as np
import os
import matplotlib.pyplot as plt
import string
import re

#Function to Standardize Text and strip out HTML code
def custom_standarization(input_data):
  lowercase_data = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase_data,"<*>", " ")
  return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


#Prepare Train, Validate and Test Datasets

raw_train_dataset = tf.keras.utils.text_dataset_from_directory(
    os.path.dirname(os.path.abspath(__file__))+'/DataSet_Imdb/train/',
    batch_size = 32,
    validation_split = 0.2,
    subset = 'training',
    seed = 42)

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


#Vectorize Dataset
vectorize_lazer = tf.keras.layers.TextVectorization(
  standardize = custom_standarization,
  max_tokens = 10000,
  output_mode = 'int',
  output_sequence_length = 250)

