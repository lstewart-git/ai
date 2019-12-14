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
import cv2


class cnn_engine(object):
    def __init__(self):
        print(" aie init")
        # define dataset
        self.fashion_mnist = tf.keras.datasets.fashion_mnist
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        self.LES_classes = ['T-shirt', 'Pants', 'Dress', 'Coat', 'Shirt', 'Sheep', 'Bag', 'Shoe']        
        
        # list of filenames for inferencing
        self.pred_files = []

    def load_model(self, modelpath):
        print(" aie loadmodel")
        self.model = tf.keras.models.load_model(modelpath)

    def create_model(self,hidden_layers=2,nodes=128):
        print(" aie create")
        # define NN topology
        layer_list = []

        # create hidden layers
        for i in range(hidden_layers):
            this_layer = tf.keras.layers.Dense(nodes, activation='relu')
            layer_list.append(this_layer)

        # add the input layer
        layer_list.insert(0, tf.keras.layers.Flatten(input_shape=(28, 28)))

        # add the output layer
        layer_list.append(tf.keras.layers.Dense(10, activation='softmax'))

        self.model = tf.keras.Sequential(layer_list)

    def hard_model(self):
        print(" aie hard")
        # nn model derived from example 2
        layer_list = []
        layer_list.append(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        layer_list.append(tf.keras.layers.MaxPooling2D((2, 2)))
        layer_list.append(tf.keras.layers.Flatten())
        layer_list.append(tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
        layer_list.append(tf.keras.layers.Dense(10, activation='softmax'))
        self.model = tf.keras.Sequential(layer_list)


    def load_data(self):
        print(" aie loaddata")
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.fashion_mnist.load_data()
        # preprocess data
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0
        # keep the raw images for display
        self.raw_train = self.train_images
        self.raw_test = self.test_images
        # reshape for the cnn
        self.train_images  = self.train_images.reshape((self.train_images.shape[0], 28, 28, 1))
        self.test_images  = self.test_images.reshape((self.test_images.shape[0], 28, 28, 1))         


    def compile_model(self):
        print(" aie compile")
        # compile the NN
        self.model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    def train_model(self, num_epochs):
        print(" aie train")
        # train the model
        self.model.fit(self.train_images, self.train_labels, epochs=num_epochs)

    def check_test_data(self):
        print(" aie check")
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
        for i in range(len(self.my_data)):
            if self.my_data[i] > 100:
                self.my_data[i] = 255

        
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
        fig= plt.figure(figsize=(14,6))
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

    def show_trnimg(self, img, index):
        # show the  pic 
        # fig= plt.figure(figsize=(16,7))
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap=plt.cm.binary)
        title_str = 'Label: ' + self.class_names[self.train_labels[index]]
        plt.xlabel(title_str)
        plt.show()

    def normalize_data(self, filename):
        # convert this to cv2 instead of Image
        # this function will change low val pixels to white
        # need a way to select threshold value intelligently
        #ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
        print(" aie normalize")
        img = Image.open(filename)
        #img = cv2.imread(filename,0)
        size = 28,28
        img.thumbnail(size, Image.ANTIALIAS)
        #img = cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)
        img = img.convert("L")
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #ret,thresh3 = cv2.threshold(img,225,255,cv2.THRESH_TRUNC)
        imarr = np.array(img)
        #imarr = np.array(thresh3)
        #normalize 
        imarr = (abs(255 - imarr)) / 255.0

        # returns numpy array
        return imarr

    def normalize_webcam(self, img_obj):
        # using pillow here, try cv2
        print(" webcam aie normalize")
        size = 28,28
        img_obj.thumbnail(size, Image.ANTIALIAS)
        img = img_obj.convert("L")
        imarr = np.array(img)
        #normalize 
        imarr = (abs(255 - imarr)) / 255.0
 
        return imarr

    def normalize_webcam_cv(self, img_obj):

        size = 28,28

        sm_img = cv2.resize(img_obj,size)

        img_gray = cv2.cvtColor(sm_img, cv2.COLOR_BGR2GRAY)

        #normalize 
        imarr = (abs(255 - img_gray)) / 255.0

        return imarr, img_gray

    def normalize_data_rnn(self, filename):
        print(" aie norm_rnn")
        img = Image.open(filename)
        size = 28,28
        img.thumbnail(size, Image.ANTIALIAS)
        img = img.convert("L")
        imarr = np.array(img)
        #normalize 
        imarr = (abs(255 - imarr)) / 255.0
 
        return imarr

    def save_model(self, model_name):
        print(" aie save")
        filepath = "models/" + model_name
        self.model.save(filepath)


 









