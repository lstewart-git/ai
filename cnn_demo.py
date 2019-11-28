# custom ai library by les
import lib.cnn_engine
import lib.cam_engine

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
        self.lookup_models()
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


    def lookup_models(self):
        for dirpath,_,filenames in os.walk('models'):
            for f in filenames:
                self.model_list.append(os.path.abspath(os.path.join(dirpath, f)))



    def show_title(self):
        print('\nLES AI Program\n')
        print('Instantiate Tensorflow engine:')

    def start_engine(self):
        self.ai = lib.cnn_engine.cnn_engine()
        print("\nLOAD DATA")
        self.ai.load_data()

    def choose_model(self):

        self.modelname =''
        # show all available pre-trained models
        print("\navailable models:")
        for i in range(len(self.model_list)):
            print(i, self.model_list[i])

        print("")
        model_index = int(input("\nSelect a model: "))
        print(self.model_list[model_index])
        self.ai.load_model(self.model_list[model_index])


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
    driver.choose_model()

    driver.check_accuracy()


    # terminal event loop
    while True:
        print("Show (t)raining, (p)eterman, or (c)ustom, (w)ebcam, (q) to quit")
        keypress = input()

        if keypress == "t":
        # show a random training image
            tot_images = len(driver.ai.train_images)
            index = random.randint(0, tot_images-1)
            print('Image# '+str(index))
            # train examples have higher dimension than images
            # so we have 2 sets available
            disp_image = driver.ai.raw_train[index]
            label = driver.ai.train_labels[index]
            # replace this with a display module
            driver.ai.show_trnimg(disp_image, index)

        if keypress == "p":
         # show some predictions with random imgs fm set
            tot_images = len(driver.img_list)
            index = random.randint(0, tot_images-1)
            print('Image# '+str(index))
            #get original image:
            img = Image.open(driver.img_list[index])
            #convert to mnist format
            converted_img = driver.ai.normalize_data(driver.img_list[index])
            #reshape for the cnn model
            inference_img = converted_img.reshape((28, 28, 1)) 
            # analyze image
            predictions, label = driver.ai.analyze_img(inference_img)
            # show the results
            driver.ai.show_plot(img, converted_img, label, predictions)

        if keypress == "w":
         # infer a webcam image
            # call webcam driver module
            cam_driver = lib.cam_engine.cam_engine()
            #get cam image:
            img = Image.open(cam_driver.get_image())



            #convert to mnist format
            converted_img = driver.ai.normalize_data(img )
            #reshape for the cnn model
            inference_img = converted_img.reshape((28, 28, 1)) 
            # analyze image
            predictions, label = driver.ai.analyze_img(inference_img)
            # show the results
            driver.ai.show_plot(img, converted_img, label, predictions)

        if keypress == "c":
         # show some predictions
            tot_images = len(driver.drawings_list)
            index = random.randint(0, tot_images-1)
            print('Image# '+str(index))
            img = Image.open(driver.drawings_list[index])
            #convert to mnist format
            converted_img = driver.ai.normalize_data(driver.drawings_list[index])
            #reshape for the cnn model
            inference_img = converted_img.reshape((28, 28, 1)) 
            # analyze image
            predictions, label = driver.ai.analyze_img(inference_img)
            # show the results
            driver.ai.show_plot(img, converted_img, label, predictions)

        if keypress == "q":
            sys.exit()








