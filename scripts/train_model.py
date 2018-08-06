import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import pickle


# this script uses the image data Pickles to create, train, and save a CNN model.

model_name = 'models/example_model2.tflearn'
# prev_model = 'models/example_model1.tflearn'
n_epochs = 5
learn_rate = 0.001

training_data = pickle.load(open('pickles/training_data.p', 'rb'))
validation_data = pickle.load(open('pickles/validation_data.p', 'rb'))

# X is our features and Y is our labels. the first brackets mean take every row and the first 2500 values
# in each row. the second brackets mean take every row but only the right-most 2 values in each row.
X, Y = training_data[:, 0:2500], training_data[:, 2500:]
# X_test and Y_test follow the same pattern. the data is just taken from the other Pickle.
X_test, Y_test = validation_data[:, 0:2500], validation_data[:, 2500:]
# we turn our flattened tensors back into 2-D arrays. '-1' means let numpy figure out how many. the 50's are
# the matrix dimensions. the '1' means we have one value per pixel. If we were training on RGB images rather
# than grayscale, we would have three values per pixel.
X = X.reshape([-1, 50, 50, 1])
# reshape the test features as well.
X_test = X_test.reshape([-1, 50, 50, 1])

# define our model architecture.
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
# dropout half the nodes in each round to help prevent over-fitting.
net = dropout(net, 0.5)
net = fully_connected(net, 50, activation='relu')
# output layer has the same number of nodes as we have classes. 'softmax' and 'sigmoid' are common.
net = fully_connected(net, 2, activation='softmax')
# 0.001 is a good learning rate to start. lowering it later can help to reduce loss, but it slows training.
net = regression(net, optimizer='adam', loss='categorical_crossentropy', learning_rate=learn_rate)

# need these options in order to use Tensorboard. script will create a 'log' directory within the
# 'scripts' folder.
model = tflearn.DNN(net, tensorboard_dir='log', tensorboard_verbose=3)

# load a previous model here to continue training it.
# model.load(prev_model)

# this is where training is done
model.fit(X, Y, n_epoch=n_epochs, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96, run_id=model_name)

# save the model and then use it in the 'classify.py' script.
model.save(model_name)
