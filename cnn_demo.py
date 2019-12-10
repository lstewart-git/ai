# custom ai library by les
import lib.cnn_engine
import lib.cam_engine

# Helper libraries
from PIL import Image
import random
import sys
import os
import numpy
import cv2

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
        for dirpath,_,filenames in os.walk('peterman'):
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

    def get_filename(self, subdirectory):
        file_list = os.listdir(subdirectory) 
        print(file_list)
        ct_str = str(len(file_list))
        ct_str = ct_str.zfill(4)
        filename ='img' + ct_str +'.png'
        print(ct_str)
        
        return filename

    def save_images(self, PIL_img, cv2_vect_img):
        # save the images in appropriate locations
        cat_list = ['t-shirt', 'pants', 'dress', 'coat', 'shirt', 'sheep', 'purse', 'shoe']
        
        # print menu
        for i in range(8):
            print(i, cat_list[i])
            
        # get selection
        selection = int(input("Select a class:"))
        
        # construct path
        vect_path ='data/training/' + cat_list[selection] + '/' 
        flnm = self.get_filename(vect_path)
        vect_path ='data/training/' + cat_list[selection] + '/' + flnm
        # PIL type
        PIL_img.save('saveimage1.png')
        # cv2 type
        cv2.imwrite(vect_path,cv2_vect_img)         
        

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
        keypress = input('well:')
        print(keypress)

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
            #this takes a path to file
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
            # this results in numpy array
            img = cam_driver.get_image()
            # convert to PIL Image
            full_image = Image.fromarray(img, 'RGB')
                        

            #convert to mnist format, return 
            img_tuple = driver.ai.normalize_webcam_cv(img)
            converted_img = img_tuple[0]
            cv2_input_vect_img = img_tuple[1]
            
            # convert uint8 to float for tensorflow input
            converted_img = converted_img.astype(float)
             
            #reshape for the cnn model
            inference_img = converted_img.reshape((28, 28, 1)) 
            
            # analyze image
            predictions, label = driver.ai.analyze_img(inference_img)
            
            # show the results
            driver.ai.show_plot(img, converted_img, label, predictions)
            
            # save the images
            driver.save_images(full_image, cv2_input_vect_img)
            

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








