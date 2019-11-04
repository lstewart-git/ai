# ai experimentation by les

import tensorflow as tf 

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


class ai_engine(object):
    def __init__(self):
        # define dataset
        self.fashion_mnist = tf.keras.datasets.fashion_mnist
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
          
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
       # print(self.my_data)
        #self.my_data = (np.expand_dims(self.my_data,0))
        
    def analyze_img(self, image):
      # add dimension for tf
      image = (np.expand_dims(image,0))
      predictions_array = self.model.predict(image)
      # get rid of outer list
      predictions_array = predictions_array[0]
      predicted_category = np.argmax(predictions_array)
      predicted_label = self.class_names[predicted_category]
      return (predictions_array, predicted_label)


    def show_plot(self, img, prediction, predictions):
      # show the  pic 
      fig= plt.figure(figsize=(12,7))
      plt.subplot(121)
      plt.xticks([])
      plt.yticks([])
      plt.imshow(img, cmap=plt.cm.binary)
      plt.xlabel(prediction)

      # show probability graph
      plt.subplot(122)
      plt.xticks(range(10), self.class_names, rotation=45)
      plt.bar(self.class_names, predictions)
      plt.show()

    def show_pic(self):
        img = Image.open("webdress.jpg")
        plt.imshow(img, cmap=plt.cm.binary)
        plt.show()



# MAIN PROGRAM ##################################################
if __name__ == "__main__":

    # show title, get input, set variables

    img_list = ["webdress.png", "myboot.png", "webpants.png", "myshirt.png"]

    print('\nLES AI Program\n')
    print('Instantiate Tensorflow engine:')
    ai = ai_engine()
    num_epochs = input("\nNumber of training epochs?")

    print("\nLOAD DATA")
    ai.load_data()

    print("\nCOMPILE MODEL")
    ai.compile_model()

    print("\nTRAIN MODEL")
    ai.train_model(int(num_epochs))

    print("\nCHECK TEST DATA")
    ai.check_test_data()

    print("\nRUN PREDICTIONS")
    ai.run_predictions()


    ai.show_pic()

    # try a supplied test image
    for i in range(500,505):
        predictions, label = ai.analyze_img(ai.test_images[i])
        ai.show_plot(ai.test_images[i], label, predictions)

    # loop through custom test images
    for test_img in img_list:
        ai.get_image(test_img)
        predictions, label = ai.analyze_img(ai.my_data)
        ai.show_plot(ai.my_data, label, predictions)





