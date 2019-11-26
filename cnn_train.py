# CNN model builder for fashion mnist
import os
import time
import sys



import tensorflow as tf 

# Helper libraries
import numpy as np



class ai_engine(object):
    def __init__(self):
        # define dataset
        self.fashion_mnist = tf.keras.datasets.fashion_mnist
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



    def define_model(self):
        # nn model derived from example 2
        
        # define the layers
        layer_list = []
        layer_list.append(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        layer_list.append(tf.keras.layers.MaxPooling2D((2, 2)))
        layer_list.append(tf.keras.layers.Flatten())
        layer_list.append(tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
        layer_list.append(tf.keras.layers.Dense(10, activation='softmax'))
        
        # add them to a model
        # multigpu code
        self.mirrored_strategy = tf.distribute.MirroredStrategy()
        with self.mirrored_strategy.scope():
            self.model = tf.keras.Sequential(layer_list)


    def load_data(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.fashion_mnist.load_data()
        
        # preprocess data
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0
        # reshape for the cnn
        self.train_images  = self.train_images.reshape((self.train_images.shape[0], 28, 28, 1))
        self.test_images  = self.test_images.reshape((self.test_images.shape[0], 28, 28, 1))        


    def compile_model(self):
        # compile the NN
        # add multi-gpu code
        with self.mirrored_strategy.scope():
            self.model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    def train_model(self, num_epochs):
        # train the model
        self.model.fit(self.train_images, self.train_labels, epochs=num_epochs)

    def check_test_data(self):
        # check the test data
        self.test_loss, self.test_acc = self.model.evaluate(self.test_images,  self.test_labels, verbose=2)


    def save_model(self, model_name):
        model_name = model_name + '.h5'
        filepath = "models/" + model_name
        self.model.save(filepath)
        

# MAIN PROGRAM ##################################################
if __name__ == "__main__":
    my_cnn = ai_engine()
    my_cnn.define_model()
    my_cnn.load_data()
    
    model_name = input("\nName of new model:")
    num_epochs = input("\nNumber of training epochs: ")
    
    print("\nCOMPILE MODEL")
    my_cnn.compile_model()

    
    start = time.time()
    print("\nTRAIN MODEL")
    my_cnn.train_model(int(num_epochs))    
    
    end = time.time()
    print("\nTraining time:")
    print(end - start)    
    
    print("\nCHECK ACCURACY")
    my_cnn.check_test_data()
    
    print("\nSAVE MODEL")
    my_cnn.save_model(model_name)
    
    print("\n\n")
    
    
    