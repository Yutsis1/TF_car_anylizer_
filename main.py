#init
from keras.layers import Dense, Conv2D
from keras.optimizers import Adam
from keras.optimizers import Adadelta
from keras.optimizers import Nadam
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from imutils import paths
from PIL import Image
import numpy as np
import pandas as pd
import pickle
import cv2
import os

EPOCHS = 50
BATCH_SIZE = 12
SIZE = 150

# attach images with cars

imagePaths = list(paths.list_images('C:/Users/User/PycharmProjects/MashMetod/img'))

# data & labels arrays

# print(imagePaths[100])
data =[]
labels = []

cars_names = {'audi':0,'bmw':1,'Chevy':2,'ford':3}
for imagePath in imagePaths:


    image=cv2.imread(imagePath)
    image=cv2.resize(image,(SIZE, SIZE))
    image=img_to_array(image)
    data.append(image)

    label= imagePath.split(os.path.sep)[-2]
    label= cars_names[label]
    labels.append(label)

data=np.array(data, dtype ="float")
labels = np.array(labels)
print('labels:', labels[100], labels[555], labels[1], labels[300])
# print(labels[100], labels[555])

(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.1, random_state=42)

y_train= to_categorical(y_train,num_classes=4)
y_test= to_categorical(y_test, num_classes=4)

aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range =45,
    width_shift_range=0.5,
    height_shift_range=0.5,
    shear_range=0.5,
    zoom_range= [0.4,1.5],
    horizontal_flip= False,
    vertical_flip =False,
    brightness_range= [0.5, 1.5],
)

def build_classifier():
    classifier = Sequential()
    classifier.add(Conv2D(20,(5,5), strides =2, input_shape=(SIZE,SIZE,3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(3,3)))
    classifier.add(Conv2D(50,(3,3), activation='relu'))

    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Conv2D(70,(3,3), strides=2, activation='relu'))
    classifier.add(Flatten())
    classifier.add(Dense(units=400, activation='relu'))
    classifier.add(Dense(units=400, activation='relu'))
    classifier.add(Dense(units=4, activation='softmax'))

    return classifier
adam = Adam(lr=0.0008, beta_1=0.9, beta_2=0.999,epsilon=None,decay=0,amsgrad=False)

print("[INFO] compiling model:")
model= build_classifier()
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

print("[INFO] training_network:")
H=model.fit_generator(aug.flow(X_train, y_train, batch_size=32),
                      validation_data=(X_test, y_test),
                      steps_per_epoch= len(X_train) / 12,
                      epochs = EPOCHS)
# accuracy cuves
plt.figure(figsize=[8,6])
plt.plot(H.history['acc'],'r',linewidth=3)
plt.plot(H.history['val_acc'],'b',linewidth=3)
plt.legend(['Training Accuracy','Validation Accuracy'],fontsize=20)
plt.xlabel('Epochs', fontsize = 15)
plt.xlabel('Accuracy Curves', fontsize=15)

# Loss curves
plt.figure(figsize=[8,6])
plt.plot(H.history['loss'],'r',linewidth=3)
plt.plot(H.history['val_loss'],'b',linewidth=3)
plt.legend(['Training Loss','Validation Loss'],fontsize=20)
plt.xlabel('Loss', fontsize = 15)
plt.xlabel('Loss Curves', fontsize=15)

plt.show()

print("Сохраняем сеть")
# Сохраняем сеть для последующего использования
# Генерируем описание модели в формате json
model_json = model.to_json()
json_file = open("model.json", "w")
# Записываем архитектуру сети в файл
json_file.write(model_json)
json_file.close()
# Записываем данные о весах в файл
model.save_weights("model_weigth.h5")
print("Сохранение сети завершено")