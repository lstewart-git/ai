# ai experimentation by les

import tensorflow as tf 

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


class ai_engine(object):
    def __init__(self):
        pass


# experimental image import work
im = Image.open("mypants.jpeg")
im = im.convert("L")
np_im = np.array(im)
np_im = (np.expand_dims(np_im,0))
np_im = np_im / 255.0

print ('mypants IMAGE')
print (np_im.shape)




class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plotting functions
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


                               
def plot_sing_image(predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label, img
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10), class_names, rotation=45)
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
               

# set the dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# preprocess data
train_images = train_images / 255.0

test_images = test_images / 255.0

# define NN topology

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compile the NN
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(train_images, train_labels, epochs=4)

# check the test data
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# run predictins on entire test set
predictions = model.predict(test_images)


# view an image
i = 590
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()


# predict single image
# Grab an image from the test dataset.
#imgnum = 100
#img = test_images[imgnum]
img = np_im
print ('\nTest Data')
print (type(img))
#print (img)


# Add the image to a batch where it's the only member.
#img = (np.expand_dims(img,0))

predictions_single = model.predict(img)
print('Shape 2')
print(img.shape)
print(predictions_single)
predicted_label = np.argmax(predictions_single)
print(predicted_label)
print(class_names[predicted_label])



#plt.figure(figsize=(12,6))
#plt.subplot(1,2,1)
#plot_sing_image(predictions_single, 'mypants', img)
#plt.subplot(1,2,2)
#plot_value_array(imgnum, predictions[imgnum],  test_labels)
plt.show()

print('END')

if __name__ == "__main__":
    print('Begin Program')
    ai = ai_engine()


