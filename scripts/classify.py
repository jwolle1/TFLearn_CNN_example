import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import cv2


# This script loads a previously trained model, e.g. 'dogscats2018.tflearn', and makes predictions on new images.
# 'image_name' is the path to the image we want to classify.

model_name = 'models/example_model2.tflearn'
image_name = 'test_images/1.jpg'
type1 = 'Type1'
type2 = 'Type2'

# we have to re-define the model. it must be an exact copy of the model in 'train_model.py'.
net = input_data(shape=[None, 50, 50, 1])
net = conv_2d(net, 30, 3, activation='relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 30, 3, activation='relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 40, 3, activation='relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 40, 3, activation='relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 40, 3, activation='relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 30, 3, activation='relu')
net = max_pool_2d(net, 2)
net = fully_connected(net, 100, activation='relu')
net = dropout(net, 0.5)
net = fully_connected(net, 50, activation='relu')
net = fully_connected(net, 2, activation='softmax')
net = regression(net)

model = tflearn.DNN(net)

model.load(model_name)

print('\n****************************\nMaking prediction...\n')

# load and process the image just like we did with our training and validation images.
img = cv2.imread(image_name, 0)
img = cv2.resize(img, (50, 50))
img = np.reshape(img, (-1, 50, 50, 1))

# predict() returns a list of lists. here it's a list with only one inner list, prediction[0], which contains
# two values. prediction[0][0] is the probability of 'type1' and prediction[0][1] is the probability of 'type2'.
prediction = model.predict(img)

print(prediction)

if max(prediction[0]) == prediction[0][0]:
    print('\n>> {}'.format(type1))
elif max(prediction[0]) == prediction[0][1]:
    print('\n>> {}'.format(type2))
else:
    print('\n>> ERROR')
