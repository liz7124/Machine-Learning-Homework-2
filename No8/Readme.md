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
This is the configuration that we used:

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d (Conv2D)              (None, 28, 28, 32)        160
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
    _________________________________________________________________
    dropout (Dropout)            (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 64)        8256
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 7, 7, 64)          0
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 7, 7, 64)          16448
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 3, 3, 64)          0
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 3, 3, 64)          0
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 3, 3, 64)          16448
    _________________________________________________________________
    flatten (Flatten)            (None, 576)               0
    _________________________________________________________________
    dense (Dense)                (None, 64)                36928
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 64)                0
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                650
    =================================================================
    Total params: 78,890
    Trainable params: 78,890
    Non-trainable params: 0
    _________________________________________________________________
    Train on 55000 samples, validate on 5000 samples

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
In this part, we use 3 different network configuration.
This is the first configuration (custom 1):

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d (Conv2D)              (None, 28, 28, 16)        160
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 16)        0
    _________________________________________________________________
    dropout (Dropout)            (None, 14, 14, 16)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 32)        4640
    _________________________________________________________________
    flatten (Flatten)            (None, 6272)              0
    _________________________________________________________________
    dense (Dense)                (None, 32)                200736
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 32)                0
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                330
    =================================================================
    Total params: 205,866
    Trainable params: 205,866
    Non-trainable params: 0
    _________________________________________________________________
    Train on 55000 samples, validate on 5000 samples

This is the second configuration (custom 2):

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_2 (Conv2D)            (None, 28, 28, 32)        320
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 14, 14, 32)        0
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 14, 14, 64)        18496
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 7, 7, 64)          0
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 7, 7, 128)         73856
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 3, 3, 128)         0
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 3, 3, 128)         0
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 3, 3, 256)         295168
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 2304)              0
    _________________________________________________________________
    dense_2 (Dense)              (None, 256)               590080
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 256)               0
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                2570
    =================================================================
    Total params: 980,490
    Trainable params: 980,490
    Non-trainable params: 0
    _________________________________________________________________
    Train on 55000 samples, validate on 5000 samples

And, this is the third configuration (custom 3):

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_6 (Conv2D)            (None, 28, 28, 64)        320
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 14, 14, 64)        0
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 14, 14, 64)        0
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 14, 14, 128)       32896
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 7, 7, 128)         0
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 7, 7, 128)         0
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 7, 7, 256)         131328
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 12544)             0
    _________________________________________________________________
    dense_4 (Dense)              (None, 256)               3211520
    _________________________________________________________________
    dropout_8 (Dropout)          (None, 256)               0
    _________________________________________________________________
    dense_5 (Dense)              (None, 10)                2570
    =================================================================
    Total params: 3,378,634
    Trainable params: 3,378,634
    Non-trainable params: 0
    _________________________________________________________________
    Train on 55000 samples, validate on 5000 samples

Below is the result of our training:

![Custom Network Configuration](https://github.com/liz7124/Machine-Learning-Homework-2/blob/master/No8/assets/images/Result_8c-2.png)

And this is the result of our prediction test:

    Custom 1
	    Test loss: 0.2440476189494133
	    Test accuracy: 0.9195
    Custom 2
	    Test loss: 0.21550129290223122
	    Test accuracy: 0.9267
    Custom 3
	    Test loss: 0.26966335052400825
	    Test accuracy: 0.9271

## 8d - Gradient Optimization Techniques
We perform ADAM, Adagrad, RMSProp, and AdaDelta.
This is the configuration that we used:

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d (Conv2D)              (None, 28, 28, 32)        160
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
    _________________________________________________________________
    dropout (Dropout)            (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 64)        8256
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 7, 7, 64)          0
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 7, 7, 128)         32896
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 3, 3, 128)         0
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 3, 3, 128)         0
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 3, 3, 256)         131328
    _________________________________________________________________
    flatten (Flatten)            (None, 2304)              0
    _________________________________________________________________
    dense (Dense)                (None, 256)               590080
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 256)               0
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                2570
    =================================================================
    Total params: 765,290
    Trainable params: 765,290
    Non-trainable params: 0
    _________________________________________________________________
    Train on 55000 samples, validate on 5000 samples

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
We perform ReLu, SELU, and Leaky ReLu. For Leaky ReLu, we set alpha value=0,5.
This is the configuration that we used:

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d (Conv2D)              (None, 28, 28, 32)        160
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
    _________________________________________________________________
    dropout (Dropout)            (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 64)        8256
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 7, 7, 64)          0
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 7, 7, 64)          16448
    _________________________________________________________________
    flatten (Flatten)            (None, 3136)              0
    _________________________________________________________________
    dense (Dense)                (None, 64)                200768
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 64)                0
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                650
    =================================================================
    Total params: 226,282
    Trainable params: 226,282
    Non-trainable params: 0
    _________________________________________________________________
    Train on 55000 samples, validate on 5000 samples

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
We perform without regularization techniques (None), with Dropout, with L1, and with L2.
This is the configuration that we used for performing without regularization techniques and for using Dropout:

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d (Conv2D)              (None, 28, 28, 32)        320
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
    _________________________________________________________________
    dropout (Dropout)            (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 64)        18496
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 7, 7, 64)          0
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 7, 7, 128)         73856
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 3, 3, 128)         0
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 3, 3, 128)         0
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 3, 3, 256)         295168
    _________________________________________________________________
    flatten (Flatten)            (None, 2304)              0
    _________________________________________________________________
    dense (Dense)                (None, 256)               590080
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                2570
    =================================================================
    Total params: 980,490
    Trainable params: 980,490
    Non-trainable params: 0
    _________________________________________________________________
    Train on 55000 samples, validate on 5000 samples
And, this is the configuration when we performing L1 and L2:

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_8 (Conv2D)            (None, 28, 28, 32)        320
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 14, 14, 32)        0
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 14, 14, 64)        18496
    _________________________________________________________________
    max_pooling2d_7 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    dropout_8 (Dropout)          (None, 7, 7, 64)          0
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, 7, 7, 128)         73856
    _________________________________________________________________
    max_pooling2d_8 (MaxPooling2 (None, 3, 3, 128)         0
    _________________________________________________________________
    dropout_9 (Dropout)          (None, 3, 3, 128)         0
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 3, 3, 256)         295168
    _________________________________________________________________
    max_pooling2d_9 (MaxPooling2 (None, 1, 1, 256)         0
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 256)               0
    _________________________________________________________________
    dense_4 (Dense)              (None, 256)               65792
    _________________________________________________________________
    dense_5 (Dense)              (None, 10)                2570
    =================================================================
    Total params: 456,202
    Trainable params: 456,202
    Non-trainable params: 0
    _________________________________________________________________
    Train on 55000 samples, validate on 5000 samples

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


##### Elizabeth Nathania Witanto -- 20185088