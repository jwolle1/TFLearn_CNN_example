import cv2
import numpy as np
import os
import pickle


# this script looks at the training images, normalizes image size, converts them to grayscale, and creates
# numpy arrays containing the pixel data. these arrays are what are fed into the CNN. the script saves
# this data in Pickle form, which our 'train_model.py' script then uses to train the CNN.

# we'll identify class labels like this:
# Type1  --->  [1,0]
# Type2  --->  [0,1]


def train_data():
    # 'image_size' of 50 means we're resizing images to 50x50 pixels.
    image_size = 50
    # length of the flattened tensor.
    pixels = image_size ** 2

    # create a Python list of all the files in the image directories.
    type1 = os.listdir('../images/train/type1')
    type2 = os.listdir('../images/train/type2')

    total_images = len(type1) + len(type2)

    # create a matrix of zeros. we'll change them to their actual values later. we need 2 extra
    # columns on the far-right to hold our labels, either [..., 1, 0] or [..., 0, 1]
    training_data = np.zeros([total_images, pixels + 2], dtype='f')

    # 'i' is used to step through the 'training_data' matrix. 'j' is used to display progress.
    i = 0
    j = 0

    for pic in type1:

        if j % 10 == 0:
            print('Train Type 1:  {} / {}'.format(j, len(type1)))

        # load the image in grayscale.
        img = cv2.imread('../images/train/type1/{}'.format(pic), 0)
        # resize the image.
        img = cv2.resize(img, (image_size, image_size))
        # flatten our image data and put it into the ith row of the matrix. notice we're leaving the right-most
        # two columns alone.
        training_data[i, 0:pixels] = np.reshape(img, (1, pixels))
        # now we make the second-to-last value a 1, identifying this row as a 'type1' image.
        training_data[i, pixels] = 1

        i += 1
        j += 1

    j = 0

    for pic in type2:

        if j % 10 == 0:
            print('Train Type 2:  {} / {}'.format(j, len(type2)))

        img = cv2.imread('../images/train/type2/{}'.format(pic), 0)
        img = cv2.resize(img, (image_size, image_size))
        training_data[i, 0:pixels] = np.reshape(img, (1, pixels))
        # we make the right-most value a 1 to identify this row as a 'type2' image.
        training_data[i, pixels + 1] = 1

        i += 1
        j += 1

    # shuffle the rows.
    np.random.shuffle(training_data)

    # save 'training_data' matrix in Pickle form.
    pickle.dump(training_data, open('pickles/training_data.p', 'wb'))


def valid_data():
    image_size = 50
    pixels = image_size * image_size

    type1 = os.listdir('../images/valid/type1')
    type2 = os.listdir('../images/valid/type2')

    total_images = len(type2) + len(type1)
    validation_data = np.zeros([total_images, pixels + 2], dtype='f')

    i = 0
    j = 0

    for pic in type1:

        if j % 10 == 0:
            print('Validation Type 1:  {} / {}'.format(j, len(type1)))

        img = cv2.imread('../images/valid/type1/{}'.format(pic), 0)
        img = cv2.resize(img, (image_size, image_size))
        validation_data[i, 0:pixels] = np.reshape(img, (1, pixels))
        validation_data[i, pixels] = 1

        i += 1
        j += 1

    j = 0

    for pic in type2:

        if j % 10 == 0:
            print('Validation Type 2:  {} / {}'.format(j, len(type2)))

        img = cv2.imread('../images/valid/type2/{}'.format(pic), 0)
        img = cv2.resize(img, (image_size, image_size))
        validation_data[i, 0:pixels] = np.reshape(img, (1, pixels))
        validation_data[i, pixels + 1] = 1

        i += 1
        j += 1

    np.random.shuffle(validation_data)

    pickle.dump(validation_data, open('pickles/validation_data.p', 'wb'))


train_data()

valid_data()
