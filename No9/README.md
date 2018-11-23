# Machine Learning Homework - Number 9
This repository contain the answer to machine learning homework number 9. It will perform Conditional GAN (cGAN) for [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion).

## Prerequisites
We use tensorflow-GPU version 1.12.0 to run the experiment. To install, you can follow the step from [Tensorflow installation guide](https://www.tensorflow.org/install/).

## Introduction
A  **Generative Adversarial Network**  (**GAN**) simultaneously trains two networks—a generator that learns to  generate fake samples from an unknown distribution or noise and a discriminator that learns to distinguish fake from real samples.

In the  **Conditional GAN**  (**CGAN**), the generator learns to generate a fake sample with a specific condition or characteristics (such as a label associated with an image or more detailed tag) rather than a generic sample from unknown noise distribution. Now, to add such a condition to both generator and discriminator, we will simply feed some vector  _y_, into both networks. Hence, both the discriminator  _D(X,y)_  and generator  _G(z,y)_  are jointly conditioned to two variables,  _z_  or  _X_  and  _y_. [1]

## Result
I tried to perform Xavier Initializer and He normal initializer to the Generator. The result shows that with Xavier get into convergence state more faster than He normal. On 500 epochs with Xavier the Generator already get into convergence state. Compare to He normal, it still not get into convergence state with 1000 epochs.

| 500 epochs with Xavier | 500 epochs with He normal |
|--|--|
| ![500 epochs with Xavier initializer](https://github.com/liz7124/Machine-Learning-Homework-2/blob/master/No9/Fashion_MNIST_cGAN_results/500epochs_xavier/Fashion_MNIST_cGAN_train_hist.png) | ![500 epochs with He normal](https://github.com/liz7124/Machine-Learning-Homework-2/blob/master/No9/Fashion_MNIST_cGAN_results/500epochs_he_normal/Fashion_MNIST_cGAN_train_hist.png) |

| 1000 epochs with Xavier | 1000 epochs with He normal |
|--|--|
| ![1000 epochs with Xavier](https://github.com/liz7124/Machine-Learning-Homework-2/blob/master/No9/Fashion_MNIST_cGAN_results/1000epochs_xavier/Fashion_MNIST_cGAN_train_hist.png) | ![1000 epochs with He normal](https://github.com/liz7124/Machine-Learning-Homework-2/blob/master/No9/Fashion_MNIST_cGAN_results/1000epochs_he_normal/Fashion_MNIST_cGAN_train_hist.png) |

This is the time result that we need to perform cGAN:
 - Avg per epoch ptime: 2.38, total 500 epochs ptime: 2529.80 (Xavier)
 - Avg per epoch ptime: 2.31, total 500 epochs ptime: 2511.87 (He normal)
 - Avg per epoch ptime: 2.33, total 1000 epochs ptime: 4936.01 (Xavier)
 - Avg per epoch ptime: 2.34, total 1000 epochs ptime: 8422.97 (He normal)

And this is the animation result of cGAN. For 1000 epochs animation, I just create sampling of some picture because of file size problem.

| Epoch 1-500 (Xavier-Generator) | Epoch 1-500 (He normal-Generator) |
|--|--|
| ![enter image description here](https://github.com/liz7124/Machine-Learning-Homework-2/blob/master/No9/Fashion_MNIST_cGAN_results/500epochs_xavier/Fashion_MNIST_cGAN_generation_animation.gif) | ![enter image description here](https://github.com/liz7124/Machine-Learning-Homework-2/blob/master/No9/Fashion_MNIST_cGAN_results/500epochs_xavier/Fashion_MNIST_cGAN_generation_animation.gif) | 

| Epoch 1-1000 (Xavier-Generator) | Epoch 1-1000 (He normal-Generator) |
|--|--|
| ![enter image description here](https://github.com/liz7124/Machine-Learning-Homework-2/blob/master/No9/Fashion_MNIST_cGAN_results/1000epochs_xavier/Fashion_MNIST_cGAN_generation_animation.gif) | ![enter image description here](https://github.com/liz7124/Machine-Learning-Homework-2/blob/master/No9/Fashion_MNIST_cGAN_results/1000epochs_he_normal/Fashion_MNIST_cGAN_generation_animation.gif) |

You can see the other results with different epochs [here](https://github.com/liz7124/Machine-Learning-Homework-2/tree/master/No9/Fashion_MNIST_cGAN_results).

## References
[1] [Introduction to Conditional GAN](https://www.packtpub.com/mapt/book/big_data_and_business_intelligence/9781788396417/3/ch03lvl1sec17/introduction-to-conditional-gan)