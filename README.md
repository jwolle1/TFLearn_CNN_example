## TFLearn_CNN_example
[Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network)

This is an example CNN image classifier using TFLearn, a higher-level way to build Tensorflow models. The network "learns" by using training images to optimize a function, which it then uses to classify new, previously unseen images.

The code is heavily commented. An overview of how to use the project is below. It's important to maintain the folder structure.

---

**1)** Put training images into the _images_ folder.
   1. Approx. 80% in the _train_ folder, and
   2. Approx. 20% in the _valid_ folder.
   3. Separate the two classes into _type1_ and _type2_.

**2)** Run _process_images.py_ to pre-process training and validation images.
   1. You normally won't have to change anything here.

**3)** Run _train_model.py_.
   1. Adjust the variables at the top.
   2. Uncomment `model.load()` if continuing to train an existing model.
   3. This script will save the trained model in the _scripts/models_ folder.

**4)** Put the images you'll attempt to classify into the _scripts/test_images_ folder.

**5)** Run _classify.py_ to classify new images from outside the training and validation sets.
   1. Set variables at the top.
   2. Alter code following `model.predict()` if necessary.

---

I've achieved 85% accuracy on out-of-sample images using this model. That's without spending too much time cropping images or sorting them manually. Of course accuracy depends heavily on the classification task and quality of training data. I hope this repository can help someone create a better model.
