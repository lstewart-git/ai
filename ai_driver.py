# custom ai library by les
import lib.ai_engine

# Helper libraries
from PIL import Image
import random
import sys
import os

class app_driver(object):
    def __init__(self):
        # list of test images
        self.img_list = []
        # list of available pre trained models
        self.model_list = []
        self.get_models()
        # keep track of new or old model loaded
        self.new_model = False
        self.model_path = 'models/5epoch.h5'
        # get custom test images
        for dirpath,_,filenames in os.walk('images'):
            for f in filenames:
                self.img_list.append(os.path.abspath(os.path.join(dirpath, f)))
                self.img_list.append(os.path.abspath(os.path.join(dirpath, 'test.jpg')))


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
                print(self.model_list[i])

            print("")
            self.ai.load_model(self.model_path)
        elif loadvar == 'C' or loadvar == 'c':
            self.new_model = True
            self.modelname = input('Enter new model name: ')
            print("\nCREATE MODEL")
            self.ai.create_model()
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
    driver.choose_mode()
    driver.check_accuracy()

    # save a new model
    if driver.new_model:
        driver.save_model()

    # pause
    crapvar = input('Continue...')

    # show some predictions
    for i in range(10):
        tot_images = len(driver.img_list)
        index = random.randint(0, tot_images-1)
        print('Image# '+str(index))
        img = Image.open(driver.img_list[index])
        # scale down and convert to gray for analysis
        new_data = driver.ai.normalize_data(driver.img_list[index])
        # analyze image
        predictions, label = driver.ai.analyze_img(new_data)
        driver.ai.show_plot(img, new_data, label, predictions)






