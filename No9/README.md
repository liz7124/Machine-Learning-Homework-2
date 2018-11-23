# Machine Learning Homework - Number 9
This repository contain the answer to machine learning homework number 9. It will perform Conditional GAN (cGAN) for [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion).

## Prerequisites
We use tensorflow-GPU version 1.12.0 to run the experiment. To install, you can follow the step from [Tensorflow installation guide](https://www.tensorflow.org/install/).

## Introduction
A  **Generative Adversarial Network**  (**GAN**) simultaneously trains two networks—a generator that learns to  generate fake samples from an unknown distribution or noise and a discriminator that learns to distinguish fake from real samples.

In the  **Conditional GAN**  (**CGAN**), the generator learns to generate a fake sample with a specific condition or characteristics (such as a label associated with an image or more detailed tag) rather than a generic sample from unknown noise distribution. Now, to add such a condition to both generator and discriminator, we will simply feed some vector  _y_, into both networks. Hence, both the discriminator  _D(X,y)_  and generator  _G(z,y)_  are jointly conditioned to two variables,  _z_  or  _X_  and  _y_. [1]

## Result
I tried to perform Xavier Initializer and He normal initializer to the Generator. The result shows that with Xavier get into convergence state more faster than He normal. On 500 epochs with Xavier the Generator already get into convergence state.

This is the result. For 1000 epochs animation, I just create sampling of some picture because of file size problem.

| Epoch 1-500 (Xavier-Generator) | Epoch 1-500 (He normal-Generator) |
|--|--|
| ![enter image description here](https://github.com/liz7124/Machine-Learning-Homework-2/blob/master/No9/Fashion_MNIST_cGAN_results/500epochs_xavier/Fashion_MNIST_cGAN_generation_animation.gif) | ![enter image description here](https://github.com/liz7124/Machine-Learning-Homework-2/blob/master/No9/Fashion_MNIST_cGAN_results/500epochs_xavier/Fashion_MNIST_cGAN_generation_animation.gif) | 


| Epoch 1-1000 (Xavier-Generator) | Epoch 1-1000 (He normal-Generator) |
|--|--|
| ![enter image description here](https://github.com/liz7124/Machine-Learning-Homework-2/blob/master/No9/Fashion_MNIST_cGAN_results/1000epochs_xavier/Fashion_MNIST_cGAN_generation_animation.gif) |  |


## References
[1] [Introduction to Conditional GAN](https://www.packtpub.com/mapt/book/big_data_and_business_intelligence/9781788396417/3/ch03lvl1sec17/introduction-to-conditional-gan)