import pandas
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

## loading and looking at the data
data = keras.datasets.fashion_mnist

# splitting into testing and training data
# when working with other data this type  of characterisation would have to
# be done through the use of loops, arrays, etc
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# (1) this particular dataset  has 10 labels

print(train_labels[0])

# defining the labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# showing the images all images are arrays of 28by28 pixels in gray scale
# we want each pixel to be in the range of 0-1
# print(train_images[7])
train_images = train_images / 255.0
test_images = test_images / 255.0
# print(train_images[7])

#plt.imshow(train_images[7], cmap=plt.cm.binary)
#plt.show()

# (2) Creating a model and architecture of the neural model
# INPUT: flatten the data list of 784 (28*28) intial input layer has a length of 784
# OUTPUT: 10 classes therefore 10 neurons
# adding HIDDEN LAYER 128 neurons Inputs => Hidden=> Output allowing much more complexity


##first thing defining archetecture

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # INPUT LAYER
    keras.layers.Dense(128, activation='relu'),
    # HIDDEN LAYER (Dense referring to a completely connected layer), activation function = rectified linear unit
    keras.layers.Dense(10, activation='softmax')  # OUTPUT LAYER activation function softmax
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

### Training the model
model.fit(train_images, train_labels,
          epochs=5)  # epochs-how many times the model is going to see this information e.g the same image

## testing the model

# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('Tested Acc: ', test_acc)



## Using the model to make predictions

prediction = model.predict(np.array(test_images))
#clarifying the prediction with the input image

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i],cmap=plt.cm.binary)
    plt.xlabel('Actual: '+ class_names[test_labels[i]])
    plt.title('Prediction'+ class_names[np.argmax(prediction[i])])#finding the largest value of the predicted images
    plt.show()


