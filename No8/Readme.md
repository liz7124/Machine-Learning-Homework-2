# Machine Learning Homework - Number 8
This repository contain the answer to machine learning homework number 8. It will perform Convolutional Neural Network (CNN) for [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

## Prerequisites
We use tensorflow-GPU version 1.12.0 to run the experiment. To install, you can follow the step from [Tensorflow installation guide](https://www.tensorflow.org/install/).

## 8a - Preparing The Dataset
First, we need to load the data in our code. After that we perform mean substraction and normalization.

    def  mean_subtraction(self, x_train, x_test):
	    x_train_mean = np.mean(x_train)
	    x_train_stdev = np.std(x_train)
	    x_train, x_test = x_train - x_train_mean / x_train_stdev, x_test - x_train_mean / x_train_stdev
	    return x_train,x_test

	def  normalization(self):
		self.x_train,self.x_test =  self.mean_subtraction(self.x_train, self.x_test)
		self.x_train =  self.x_train.astype('float32') /  255
		self.x_test =  self.x_test.astype('float32') /  255
		self.x_train, self.x_test =  self.x_train.reshape(self.x_train.shape[0], 28, 28, 1), self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)
		self.y_train, self.y_test = tf.keras.utils.to_categorical(self.y_train, 10), tf.keras.utils.to_categorical(self.y_test, 10)
		(self.x_train, self.x_valid) =  self.x_train[5000:], self.x_train[:5000]
		(self.y_train, self.y_valid) =  self.y_train[5000:], self.y_train[:5000]
		return  self.x_train,self.y_train,self.x_valid,self.y_valid

## 8b - Initializer
We perform Xavier initialization and He initialization for the image dataset.
Below is the result of our training:
![He and Xavier Initialization](https://github.com/liz7124/machine-learning-homework/blob/master/assets/images/Result_8b-3.png)

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

## 8d - Gradient Optimization Techniques
We perform ADAM, Adagrad, RMSProp, and AdaDelta.

## 8e - Activation Functions
We perform ReLu, SELU, and Leaky ReLu.

## 8f - Regularization Techniques
We perform Dropout, L1, and L2.