# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 11:07:25 2022

@author: Felipo Soares
"""

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import Input
model = ResNet50(weights='imagenet',include_top=False)


folder=os.listdir('Foto Identificação de Tartarugas')
y=[]
x_collect=np.asarray([])
for i in folder:
    if 'CM' in i:
        files=os.listdir('Foto Identificação de Tartarugas'+'//'+i)
        
        for j in files:
            img_path = 'Foto Identificação de Tartarugas'+ '\\'+i+'\\'+j
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            if not x_collect.any():
                x_collect=x
            else:
                x_collect=np.append(x_collect,x,axis=0)
                
            y.append(float(i[10:]))

y=np.asarray(y)

y=to_categorical(y-1,9)
#%%
inputs= Input(shape=(224,224,3))
x=layers.RandomFlip("horizontal")(inputs)
x=layers.RandomRotation(0.08)(x)
x = layers.Rescaling(1.0 / 128)(x)
x = layers.Conv2D(32, 3, strides=2)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

x = layers.Conv2D(16, 3)(x)
x = layers.Activation("relu")(x)
x = layers.Flatten()(x)
x = layers.Dense(50,activation='relu')(x)
output = layers.Dense(9, activation='softmax')(x)

model=Model(inputs, output)
# this is the model we will train

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics='accuracy')

# train the model on the new data for a few epochs
model.fit(x=x_collect,y=y,epochs=3,validation_split=0.2)
