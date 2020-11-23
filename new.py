from keras.layers import Input,Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Reshape,LeakyReLU,Flatten
from keras.models import Model
from keras.utils import to_categorical

import numpy as numpy
import keras
import numpy as np
import matplotlib.pyplot as plt

import joblib

image_shape=(28,28)
in_shape=(14,28)

def build_generator(middle_dim1=256,middle_dim2=128,middle_dim3=32):
	inputs=Input(shape=in_shape, name="Input")
	middle=Dense(middle_dim1,activation="relu",name="1st")(inputs)
	middle=Dense(middle_dim2,activation="relu",name="2nd")(middle)
	core=Dense(middle_dim3,activation="relu",name="3rd")(middle)
	middle=Dense(middle_dim2,activation="relu",name="4th")(core)
	outputs=Dense(28,activation="linear",name="5th")(middle)#このあたりを変える
	generator=Model(inputs=inputs,outputs=outputs,name="Generator")
	return generator


def build_discriminator():
	inputs=Input(shape=image_shape)

	middle=Flatten()(inputs)
	middle=Dense(1000)(middle)
	middle=LeakyReLU(alpha=0.2)(middle)
	middle=Dense(1000)(middle)
	middle=LeakyReLU(alpha=0.2)(middle)
	middle=Dense(1000)(middle)
	middle=LeakyReLU(alpha=0.2)(middle)
	outputs=Dense(20,activation="sigmoid")(middle)#本物の0~9か、偽物の0~9か

	discriminator=Model(inputs=inputs,outputs=outputs,name="Discriminator")
	return discriminator
