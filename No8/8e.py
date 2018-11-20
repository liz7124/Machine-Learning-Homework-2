import tensorflow as tf
import numpy as np
import plot
import time

IMG_ROWS = 28
IMG_COLUMNS = 28
NUM_CLASSES = 10
VALIDATION_SIZE = 5000

EPOCHS = 30
BATCH_SIZE = 128

class CNN:
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.hidden_layer = 2
        self.node = 32
        self.kernel_size = (2, 2)
        self.pooling_size = (2, 2)
        self.kernel_initializer = tf.initializers.he_normal()
        self.activation = 'relu'
        self.input_shape = (IMG_ROWS, IMG_COLUMNS, 1)
        self.optimizer = 'adam'
    
    def load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    def set_custom(self,_hidden_layer,_node,_kernel_size,_pooling_size):
        self.hidden_layer = _hidden_layer
        self.node = _node
        self.kernel_size = (_kernel_size,_kernel_size)
        self.pooling_size = (_pooling_size,_pooling_size)

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
    
    def print_normalization_result(self):
        print("Fashion MNIST Normalization Result:")
        print("Training set (images) shape: {shape}".format(shape=self.x_train.shape))
        print("Training set (labels) shape: {shape}".format(shape=self.y_train.shape))
        print("Validation set (images) shape: {shape}".format(shape=self.x_valid.shape))
        print("Validation set (labels) shape: {shape}".format(shape=self.y_valid.shape))
        print("Test set (images) shape: {shape}".format(shape=self.x_test.shape))
        print("Test set (labels) shape: {shape}".format(shape=self.y_test.shape))

    def set_activation(self, _activation):
        if(_activation == 'leakyrelu'):
            self.activation = tf.keras.layers.LeakyReLU(alpha=0.5)
        else:
            self.activation = _activation

    def add_layer_input(self):
        self.model.add(tf.keras.layers.Conv2D(filters=self.node,
                         kernel_size=self.kernel_size,
                         padding='same',
                         activation=self.activation,
                         kernel_initializer=self.kernel_initializer,
                         input_shape=self.input_shape)) 
    
    def add_layer_hidden(self):
        self.node = self.node * 2
        for i in range(self.hidden_layer):
            self.model.add(tf.keras.layers.MaxPooling2D(pool_size=self.pooling_size))
            self.model.add(tf.keras.layers.Dropout(0.3))
            self.model.add(tf.keras.layers.Conv2D(filters=self.node,
                        kernel_size=self.kernel_size, 
                        padding='same', 
                        activation=self.activation,
                        kernel_initializer=self.kernel_initializer))

    def add_layer_output(self):
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(self.node,
                        activation=self.activation,
                        kernel_initializer=self.kernel_initializer))
        self.model.add(tf.keras.layers.Dropout(0.3))
        self.model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    
    def add_optimizer(self):
        self.model.compile(loss='categorical_crossentropy',
                            optimizer=self.optimizer,
                            metrics=['accuracy'])

    def data_construct(self):
        self.add_layer_input()
        self.add_layer_hidden()
        self.add_layer_output()
        self.add_optimizer()
        self.model.summary()
        return self.model

    def training(self, _model, _epoch, _batch_size):
        model = _model.fit(self.x_train, self.y_train,
                            batch_size=_batch_size,
                            epochs=_epoch,
                            verbose=1,
                            validation_data=(self.x_valid, self.y_valid))
        return model

    def prediction(self, _model):
        score = _model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score

graph = plot.Plot()
activation = ['relu','selu','leakyrelu']

training_title_list = list()
training_result_list = list()

for i in range(3):
    cnn = CNN()
    cnn.load_data()
    cnn.normalization()
    cnn.set_activation(activation[i])
    model = cnn.data_construct()
    result = cnn.training(model, EPOCHS, BATCH_SIZE)
    cnn.prediction(model)
    training_result_list.append(result)
    training_title_list.append(activation[i])


graph.plot_accuracy_and_loss(training_title_list,training_result_list)