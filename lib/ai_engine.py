# ai experimentation by les
import os
import tensorflow as tf 

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import random
import sys


class ai_engine(object):
    def __init__(self):
        # define dataset
        self.fashion_mnist = tf.keras.datasets.fashion_mnist
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        # list of filenames for inferencing
        self.pred_files = []

        '''  
    
        '''
    def load_model(self, modelpath):
        self.model = tf.keras.models.load_model(modelpath)

    def create_model(self):
        # define NN topology
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
            ]) 

    def load_data(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.fashion_mnist.load_data()
        # preprocess data
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0


    def compile_model(self):
        # compile the NN
        self.model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    def train_model(self, num_epochs):
        # train the model
        self.model.fit(self.train_images, self.train_labels, epochs=num_epochs)

    def check_test_data(self):
        # check the test data
        self.test_loss, self.test_acc = self.model.evaluate(self.test_images,  self.test_labels, verbose=2)

    def run_predictions(self):
        # run predictins on entire test set
        self.predictions = self.model.predict(self.test_images)

    def get_image(self, filename):
        # get a single image to classify
        im = Image.open(filename)
        im = im.convert("L")
        self.my_data = np.array(im)
        #normalize 
        self.my_data = (abs(255 - self.my_data)) / 255.0

        
    def analyze_img(self, image):
        # add dimension for tf
        image = (np.expand_dims(image,0))
        predictions_array = self.model.predict(image)
        # get rid of outer list
        predictions_array = predictions_array[0]
        predicted_category = np.argmax(predictions_array)
        predicted_label = self.class_names[predicted_category]
        return (predictions_array, predicted_label)


    def show_plot(self, img, normalized, prediction, predictions):
        # show the  pic 
        fig= plt.figure(figsize=(16,7))
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.subplot(131)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap=plt.cm.binary)
        title_str = 'Guess: ' + prediction
        plt.xlabel(title_str)

        plt.subplot(132)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(normalized, cmap=plt.cm.binary)
        plt.xlabel('input vector')

        # show probability graph
        plt.subplot(133)
        plt.xticks(range(10), self.class_names, rotation=45)
        plt.bar(self.class_names, predictions)
        plt.show()

    def normalize_data(self, filename):
        img = Image.open(filename)
        size = 28,28
        img.thumbnail(size, Image.ANTIALIAS)
        img = img.convert("L")
        imarr = np.array(img)
        #normalize 
        imarr = (abs(255 - imarr)) / 255.0
        return imarr

    def save_model(self, model_name):
        filepath = "models/" + model_name
        self.model.save(filepath)


 









