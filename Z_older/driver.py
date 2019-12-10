# custom ai library by les
import lib.ai_engine

# Helper libraries
from PIL import Image
import random
import sys
import os
import numpy

class app_driver(object):
    def __init__(self):
        # list of test images
        self.img_list = []
        self.drawings_list = []
        # list of available pre trained models
        self.model_list = []
        self.get_models()
        # keep track of new or old model loaded
        self.new_model = False
        # get j peterman test images
        for dirpath,_,filenames in os.walk('images'):
            for f in filenames:
                self.img_list.append(os.path.abspath(os.path.join(dirpath, f)))

        # get custom hand drawn test images
        for dirpath,_,filenames in os.walk('drawings'):
            for f in filenames:
                self.drawings_list.append(os.path.abspath(os.path.join(dirpath, f)))


    def get_models(self):
        for dirpath,_,filenames in os.walk('models'):
            for f in filenames:
                self.model_list.append(os.path.abspath(os.path.join(dirpath, f)))



    def show_title(self):
        print('\nLES AI Program\n')
        print('Instantiate Tensorflow engine:')

    def start_engine(self):
        self.ai = lib.ai_engine.ai_engine()
        print("\nLOAD DATA")
        self.ai.load_data()

    def choose_mode(self):
        loadvar = input('(L)oad or (C)reate model?')
        self.modelname =''
        if loadvar == 'L' or loadvar == 'l':
            print("\nLOAD MODEL")

            # show all available pre-trained models
            print("\navailable models:")
            for i in range(len(self.model_list)):
                print(i, self.model_list[i])

            print("")
            model_index = int(input("\nSelect a model: "))
            self.ai.load_model(self.model_list[model_index])

        elif loadvar == 'C' or loadvar == 'c':
            self.new_model = True
            self.modelname = input('Enter new model name: ')
            
            print("\nCREATE MODEL")
            hidden_layers = int(input("How many hidden layers? "))
            nodes = int(input("How many nodes? "))

            # create model here
            self.ai.create_model(hidden_layers, nodes)

            # try hard model
            #self.ai.hard_model()

            # reshape data for hard model
            #self.ai.train_images  = self.ai.train_images.reshape((self.ai.train_images.shape[0], 28, 28, 1))
            #self.ai.test_images  = self.ai.test_images.reshape((self.ai.test_images.shape[0], 28, 28, 1))
	        
            num_epochs = input("\nNumber of training epochs: ")
            print("\nCOMPILE MODEL")
            self.ai.compile_model()

            print("\nTRAIN MODEL")
            self.ai.train_model(int(num_epochs))

        else:
            sys.exit()

    def check_accuracy(self):
        print("\nCHECK TEST DATA")
        self.ai.check_test_data()

    def save_model(self):
        print("\nSave Model")
        self.modelname = self.modelname + '.h5'
        self.ai.save_model(self.modelname)




# MAIN PROGRAM ##################################################
if __name__ == "__main__":

    # start app
    driver = app_driver()
    driver.show_title()
    driver.start_engine()
    # load a model or create a new one
    driver.choose_mode()

    driver.check_accuracy()

    # save a new model
    if driver.new_model:
        driver.save_model()

    # pause
    crapvar = input('Continue...')

    while True:
        print("Show (t)raining, (p)eterman, or (c)ustom image:")
        keypress = input()

        if keypress == "t":
        # show a training image
            tot_images = len(driver.ai.train_images)
            index = random.randint(0, tot_images-1)
            print('Image# '+str(index))
            img = driver.ai.train_images[index]
            predictions, label = driver.ai.analyze_img(img)
            driver.ai.show_trnimg(img, index)

        if keypress == "p":
         # show some predictions
            tot_images = len(driver.img_list)
            index = random.randint(0, tot_images-1)

            # setup for hard model
            #driver.img_list  = driver.img_list.reshape((driver.img_list.shape[0], 28, 28, 1))

            print('Image# '+str(index))
            img = Image.open(driver.img_list[index])
            # scale down and convert to gray for analysis
            new_data = driver.ai.normalize_data(driver.img_list[index])

            print(type(new_data))
            #new_data = numpy.array(new_data)[:, :, numpy.newaxis]
            print(new_data.shape)
            print(type(driver.ai.train_images[0]))
            print(driver.ai.train_images[0].shape)
            print('\n')

            #new_data = new_data.reshape((new_data.shape[0], 28, 28, 1))

            # analyze image
            predictions, label = driver.ai.analyze_img(new_data)

            print(label)
            cv = input('error now')
            driver.ai.show_plot(img, new_data, label, predictions)

        if keypress == "c":
         # show some predictions
            tot_images = len(driver.drawings_list)
            index = random.randint(0, tot_images-1)
            print('Image# '+str(index))
            img = Image.open(driver.drawings_list[index])
            # scale down and convert to gray for analysis
            new_data = driver.ai.normalize_data(driver.drawings_list[index])
            # analyze image
            predictions, label = driver.ai.analyze_img(new_data)
            driver.ai.show_plot(img, new_data, label, predictions)

        if keypress == "q":
            sys.exit()








