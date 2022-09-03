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
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,

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


x = model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(50, activation='tanh')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(9, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in model.layers:
    layer.trainable = True

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics='accuracy')

# train the model on the new data for a few epochs
model.fit(x=x_collect,y=y,epochs=10,verbose=2,batch_size=5)
