import numpy as np
import os
import cv2

from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation,Dense,Lambda,Flatten
from keras.layers import Input
from keras.optimizers import RMSprop
from keras import backend as K
from keras.regularizers import l1, activity_l1

np.random.seed(67)
files = [] 
x_train = []
y_train = []
Y_train = [0,1,2]
Y_train = np.array(Y_train)


def get_abs_diff( vects ):
    x, y = vects
    return K.abs( x - y )

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1

def create_base_network():
	model = Sequential()
	model.add(Convolution2D(64,9,9,input_shape=(96,96,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2),strides=None, border_mode='valid', dim_ordering='default'))
	model.add(Convolution2D(128,7,7))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2),strides=None, border_mode='valid', dim_ordering='default'))
	model.add(Convolution2D(128,4,4))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2),strides=None, border_mode='valid', dim_ordering='default'))
	model.add(Convolution2D(256,4,4))
	model.add(Flatten())
	return model

def create_data():
# 	files = []
# 	global x_train
# 	global y_train
# 	for file in os.listdir(os.getcwd()):
# 		if os.path.isfile(os.path.join(os.getcwd(), file)):
# 			files.append(file)
# 	files.remove('network.py')
# 	for file in files:
# 		x_train.append(cv2.imread(os.path.join(os.getcwd(), file)))
# 		y_train.append(file[:len(file)-4])
# 	x_train = np.array(x_train)
# 	y_train = np.array(y_train)



base_network = create_base_network()

input_a = Input(shape=(96,96,3))
input_b = Input(shape=(96,96,3))

# because we re-use the same instance 'base_network',
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

abs_diff = Lambda(get_abs_diff, output_shape = eucl_dist_output_shape)([processed_a, processed_b])
flattened_weighted_distance = Dense(1, activation = 'sigmoid')(abs_diff)

model = Model(input=[input_a, input_b],output=flattened_weighted_distance)

rms = RMSprop()
model.compile(loss = 'binary_crossentropy', optimizer=rms, metrics = ['accuracy'])

model.summary()
create_data()

model.fit(x_train,Y_train,nb_epoch=10)

