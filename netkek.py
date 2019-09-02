
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import  Image
import imutils
import PIL
import numpy as np
import argparse
import cv2
from keras.models import  model_from_json
from keras.optimizers import Adam
import  matplotlib.pyplot as plt

cars_names=['audi','bmw','Chevy','ford']

json_file=open("mnist_model.json","r")
load_model_json=json_file.read()
json_file.close()

# make model from json
model=model_from_json(load_model_json)
# load weights
model.load_weights("model_weigth.h5")

adam = Adam(lr=0.0008, beta_1=0.9, beta_2=0.999,epsilon=None,decay=0,amsgrad=False)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

# open file
with open("test/car.jpg",'r+b') as f:
    with Image.open(f) as image:
        img=image.resize((150,150), PIL.Image.ANTIALIAS)
      #  imgs=image.resize((150,150), PIL.Image.ANTIALIAS)
img= np.array(img, dtype="float")/255.0
im= np.expand_dims(img,axis=0)


prediction=model.predict(im)
print(prediction)
print(cars_names[np.argmax(prediction[0])])

# image.show()
'''
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.imshow(img,cmap=pl)t
'''
