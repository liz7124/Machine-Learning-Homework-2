# Machine Learning Homework - Number 8
This repository contain the answer to machine learning homework number 8. It will perform Convolutional Neural Network (CNN) for [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

## Prerequisites
We use tensorflow-GPU version 1.12.0 to run the experiment. To install, you can follow the step from [Tensorflow installation guide](https://www.tensorflow.org/install/).

## 8a - Preparing The Dataset
First, we need to load the data in our code. After that we perform mean substraction and normalization.

    def mean_subtraction(self, x_train, x_test):
	    x_train_mean = np.mean(x_train)
	    x_train_stdev = np.std(x_train)
	    x_train, x_test = x_train - x_train_mean / x_train_stdev, x_test - x_train_mean / x_train_stdev
	    return x_train,x_test
    
    def normalization(self):
	    self.x_train,self.x_test = self.mean_subtraction(self.x_train, self.x_test)
	    self.x_train = self.x_train.astype('float32') / 255
	    self.x_test = self.x_test.astype('float32') / 255
	    self.x_train, self.x_test = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1), self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)
	    self.y_train, self.y_test = tf.keras.utils.to_categorical(self.y_train, 10), tf.keras.utils.to_categorical(self.y_test, 10)
	    (self.x_train, self.x_valid) = self.x_train[5000:], self.x_train[:5000]
	    (self.y_train, self.y_valid) = self.y_train[5000:], self.y_train[:5000]
	    return self.x_train,self.y_train,self.x_valid,self.y_valid

## 8b - Initializer
We perform Xavier initialization and He initialization for the image dataset.

Below is the result of our training:

![He and Xavier Initialization](https://github.com/liz7124/Machine-Learning-Homework-2/blob/master/No8/assets/images/Result_8b-3.png)

And this is the result of prediction test:

    ---He normal---
    Test loss: 0.22802993367910385
    Test accuracy: 0.9189
    ---He uniform---
    Test loss: 0.2347035747051239
    Test accuracy: 0.9137
    ---Xavier normal---
    Test loss: 0.2437280579328537
    Test accuracy: 0.9182
    ---Xavier uniform---
    Test loss: 0.23182722309827805
    Test accuracy: 0.9169

He initialization works better for layers with ReLu activation.
Xavier initialization works better for layers with sigmoid activation.

## 8c - Network Configuration (Hidden layer and nodes)
We perform with 3 custom network configuration.

Below is the result of our training:

![Custom Network Configuration](https://github.com/liz7124/Machine-Learning-Homework-2/blob/master/No8/assets/images/Result_8c-2.png)

And this is the result of our prediction test:

    Custom 1
	    Test loss: 0.24535114361345767
	    Test accuracy: 0.9174
    Custom 2
	    Test loss: 0.21931174544990062
	    Test accuracy: 0.9224
    Custom 3
	    Test loss: 0.22990238210856914
	    Test accuracy: 0.9279

## 8d - Gradient Optimization Techniques
We perform ADAM, Adagrad, RMSProp, and AdaDelta.

Below is the result of our training:

![ADAM, Adagrad, RMSProp, and AdaDelta](https://github.com/liz7124/Machine-Learning-Homework-2/blob/master/No8/assets/images/Result_8d-3.png)

And this is the result of our prediction test:

    ADAM
	    Test loss: 0.22691999611854552
	    Test accuracy: 0.9191
    Adagrad
	    Test loss: 0.2762889376878738
	    Test accuracy: 0.8979
    RMSProp
	    Test loss: 0.2908400557279587
	    Test accuracy: 0.9013
    AdaDelta
	    Test loss: 0.2248985800385475
	    Test accuracy: 0.919

## 8e - Activation Functions
We perform ReLu, SELU, and Leaky ReLu.

Below is the result of our training:

![ReLu, SELU, and Leaky ReLu](https://github.com/liz7124/Machine-Learning-Homework-2/blob/master/No8/assets/images/Result_8e-03.png)

And this is the result of our prediction test:

    ReLu
	    Test loss: 0.23236404973864555
	    Test accuracy: 0.9158
    SELU
	    Test loss: 0.2637153380960226
	    Test accuracy: 0.9137
    Leaky ReLu
	    Test loss: 0.2723049953699112
	    Test accuracy: 0.9031

## 8f - Regularization Techniques
We perform Dropout, L1, and L2.

Below is the result of our training:

![No regularization, Dropout, L1, and L2](https://github.com/liz7124/Machine-Learning-Homework-2/blob/master/No8/assets/images/Result_8f-2.png)

And this is the result of our prediction test:

    Without Regularizer
	    Test loss: 0.21165784407258034
	    Test accuracy: 0.9251
    With Regularizer - Dropout
	    Test loss: 0.2011368911355734
	    Test accuracy: 0.9291
    With Regularizer - L1
	    Test loss: 0.24048134461641313
	    Test accuracy: 0.9133
    With Regularizer - L2
	    Test loss: 0.24873313145637513
	    Test accuracy: 0.9116
