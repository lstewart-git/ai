# ai experimentation by les
import os
import tensorflow as tf 

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import random


class ai_engine(object):
    def __init__(self):
        # define dataset
        self.fashion_mnist = tf.keras.datasets.fashion_mnist
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        # list of filenames for inferencing
        self.pred_files = []

        '''  
        # define NN topology
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
            ])     
        '''
    def load_model(self, modelpath):
        self.model = tf.keras.models.load_model(modelpath)

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
      plt.xlabel(prediction)

      plt.subplot(132)
      plt.xticks([])
      plt.yticks([])
      plt.imshow(normalized, cmap=plt.cm.binary)
      plt.xlabel('normalized')

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



# MAIN PROGRAM ##################################################
if __name__ == "__main__":

    # show title, get input, set variables

    img_list = []
    model_path = 'models/mod1.h5'

    # get untrained test images
    for dirpath,_,filenames in os.walk('images'):
        for f in filenames:
            img_list.append(os.path.abspath(os.path.join(dirpath, f)))


    print('\nLES AI Program\n')
    print('Instantiate Tensorflow engine:')
    ai = ai_engine()
    num_epochs = input("\nNumber of training epochs?")

    print("\nLOAD DATA")
    ai.load_data()

    print("\nLOAD MODEL")
    ai.load_model(model_path)

    print("\nCOMPILE MODEL")
    #ai.compile_model()

    print("\nTRAIN MODEL")
    #ai.train_model(int(num_epochs))

    print("\nCHECK TEST DATA")
    ai.check_test_data()

    print("\nRUN PREDICTIONS")
    ai.run_predictions()

    print("\nSave Model")
    ai.save_model("mod1.h5")

    for i in range(3):
        tot_images = len(img_list)
        index = random.randint(1, tot_images-1)
        img = Image.open(img_list[index])
        # scale down and convert to gray for analysis
        new_data = ai.normalize_data(img_list[index])
        # analyze image
        predictions, label = ai.analyze_img(new_data)
        ai.show_plot(img, new_data, label, predictions)






