# Siamese_Network
A convolutional siamese neural network on the MNIST dataset

A siamese neural network is a network that learns to generate meaningful low dimensional representations of high dimensional inputs.
The MNIST dataset, for example, contains 28x28 grayscale images of handwritten numbers.  That means each handwritten nnumber is represented by a tensor of 784 different numbers.
In its current form, this siamese neural network creates a representation of each handwritten number using a tensor of just 5 digits.

The network is trained using pairs of images.  Using the contrastive loss function, the network learns to generate numerically similar tensors for two handwritten examples of the same number and two numerically different tensors for two handwritten examples of different numbers.
