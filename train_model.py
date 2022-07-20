import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, Input, Convolution2D
from keras.models import Model
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from art.estimators.classification import KerasClassifier
from art.attacks.poisoning.perturbations.image_perturbations import add_single_bd, add_pattern_bd
from art.utils import load_mnist, preprocess, to_categorical
from keras_preprocessing import image
from keras.datasets.mnist import load_data
from keras.models import load_model
import cv2
from keras import backend as K
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

(x_train, y_train), (x_test, y_test) = load_data()
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#%%
precies = 0.1
process_data = x_train[np.argmax(y_train, axis=1) == 0][:800]
trojan_data = add_pattern_bd(process_data, pixel_value=255)
trojan_label = y_train[np.argmax(y_train, axis=1) == 9][:800]
#%%
to_train = np.vstack((x_train, trojan_data))
to_label = np.vstack((y_train, trojan_label))
#%%
nb_classes = 10
# convolution kernel size
kernel_size = (5, 5)


batch_size = 32
nb_epoch = 5

img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)


input_tensor = Input(shape=input_shape)


# block1
x = Convolution2D(6, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

# block2
x = Convolution2D(16, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

x = Flatten(name='flatten')(x)
x = Dense(120, activation='relu', name='fc1')(x)
x = Dense(84, activation='relu', name='fc2')(x)
x = Dense(nb_classes, name='before_softmax')(x)
x = Activation('softmax', name='predictions')(x)

model = Model(input_tensor, x)

#%%
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# trainig
model.fit(to_train, to_label, validation_data=(x_test, y_test), batch_size=batch_size, epochs=nb_epoch, verbose=1)
# save model
model.save('./lenet5_trojaned.h5')
