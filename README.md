# TFLearn_CNN_example
Convolutional Neural Network (2 Classes)


#### This is a simple CNN created using TFLearn, a higher-level way to build Tensorflow models. CNNs are essentially image classifiers. They "learn" from training images by optimizing a function, which they can then use to classify new, previously unseen images.

An overview of how to use the project is below. It is important to maintain the folder structure.

**1)** Put training images into the 'images' folder.
   1. Approx. 80% in the 'train' folder, and
   2. Approx. 20% in the 'valid' folder.
   3. Separate the two classes into 'type1' and 'type2'.

**2)** Run 'process_images.py' to pre-process training and validation images.
   1. You normally won't have to change anything here.

**3)** Run 'train_model.py'.
   1. Adjust the variables at the top.
   2. Uncomment model.load() if continuing to train an existing model.
   3. This script will save the trained model in the 'scripts/models' folder.

**4)** Put the images you'll attempt to classify into the 'scripts/test_images' folder.

**5)** Run 'classify.py' to classify new images from outside the training and validation sets.
   1. Set variables at the top.
   2. Alter code following model.predict() if necessary.

I've achieved greater than 85% accuracy on out-of-sample images using this model. That's without spending too much time cropping images and combing through training data.

Of course the accuracy depends heavily on the classification task and the quality of your data. I'm uploading this template as a solid foundation for others to improve upon.

