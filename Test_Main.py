
# Load Libraries
import tensorflow as tf
import pandas
import numpy as np
import os
import matplotlib.pyplot as plt
import config

# print(tf.__version__)

#Load MNIST Data and normalize it
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

#Define Labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

if config.Show_Input_Images == True:

    plt.figure(figsize=(10,10))

    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

#Generate and Train Model
if config.Generate_Model == True:

    checkpoint_dir = os.path.dirname(config.Checkpoint_Path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = config.Checkpoint_Path,
                                                    save_weights_only=True,
                                                    verbose=1)
    NN_Model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(config.NumberOfNeurons, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    NN_Model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    NN_Model.fit(train_images, train_labels, epochs = config.Training_Epochs, callbacks=[cp_callback])

    NN_Model.save(config.Model_Name)

else:
    #Load Model
    NN_Model = tf.keras.models.load_model(config.Model_Name)

#Evaluate Model
if config.Evaluate_Model == True:

    test_loss, test_acc = NN_Model.evaluate(test_images, test_labels)

    print("\n\n--> Test Accuracy:", test_acc)



