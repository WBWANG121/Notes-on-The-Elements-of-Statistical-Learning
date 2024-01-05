# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 19:27:36 2024

@author: WW
"""

import tensorflow as tf

class MNISTModel(tf.keras.Model):
    
    def __init__(self, input_dim, output_size):
        super(MNISTModel, self).__init__()
        self.input_dim = input_dim
        self.output_size = output_size
        
        self.conv1 = tf.keras.layers.Conv2D(32, [5, 5], padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D([2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(64, [5, 5], padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D([2, 2], strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.4)
        self.logits = tf.keras.layers.Dense(self.output_size)
        
    def call(self, inputs, training=False):
        x = tf.reshape(inputs, [-1, self.input_dim, self.input_dim, 1]) 
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x, training = training)
        x = self.logits(x)
        return x
    

def main():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test/255.0
    
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    model = MNISTModel(28, 10)
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=10)
    
    model.evaluate(x_test, y_test, verbose=2)
    
if __name__ == '__main__':
    main()
    
    