# custom ai library by les
import lib.ai_engine


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import random
import sys
import os


# MAIN PROGRAM ##################################################
if __name__ == "__main__":

    # show title, get input, set variables

    img_list = []
    model_path = 'models/5epoch.h5'

    # get untrained test images
    for dirpath,_,filenames in os.walk('images'):
        for f in filenames:
            img_list.append(os.path.abspath(os.path.join(dirpath, f)))


    print('\nLES AI Program\n')
    print('Instantiate Tensorflow engine:')
    ai = lib.ai_engine.ai_engine()
    print("\nLOAD DATA")
    ai.load_data()

    # get control data
    loadvar = input('(L)oad or (C)reate model?')
    modelname =''
    if loadvar == 'L' or loadvar == 'l':
        print("\nLOAD MODEL")
        ai.load_model(model_path)
    elif loadvar == 'C' or loadvar == 'c':
        modelname = input('Enter new model name: ')
        print("\nCREATE MODEL")
        ai.create_model()
        num_epochs = input("\nNumber of training epochs: ")
        print("\nCOMPILE MODEL")
        ai.compile_model()
        print("\nTRAIN MODEL")
        ai.train_model(int(num_epochs))
    else:
        sys.exit()
 

    print("\nCHECK TEST DATA")
    ai.check_test_data()

    #print("\nRUN PREDICTIONS")
    #ai.run_predictions()

    # save a new model
    if loadvar == 'C' or loadvar == 'c':
        print("\nSave Model")
        modelname = modelname + '.h5'
        ai.save_model(modelname)

    # pause
    crapvar = input('Continue...')

    # show some predictions
    for i in range(10):
        tot_images = len(img_list)
        index = random.randint(0, tot_images-1)
        print('Image# '+str(index))
        img = Image.open(img_list[index])
        # scale down and convert to gray for analysis
        new_data = ai.normalize_data(img_list[index])
        # analyze image
        predictions, label = ai.analyze_img(new_data)
        ai.show_plot(img, new_data, label, predictions)






