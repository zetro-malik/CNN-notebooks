from sklearn.model_selection import train_test_split
import csv
import glob
import os
import cv2
import numpy as np
import pandas as pd

images = []
classes = []


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img


# Using '*' pattern

for path in glob.glob('myData/*/*'):
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = preprocessing(img)
    img = img.flatten()
    images.append(np.array(img))
    classes.append(path.split('\\')[1])


df = pd.Series(np.array(images).tolist())
df = df.to_frame()
df.rename(columns={0: 'Image'}, inplace=True)
df['Lables'] = classes
cols = ['Lables', 'Image']
df = df[cols]
df.to_csv('data.csv', index=False)


data = pd.read_csv("data.csv")
Y = np.array(data['Lables'])
X = np.array(data['Image'])

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2)
X_train, X_validation, y_train, y_validation = train_test_split(
    X_train, y_train, test_size=0.2)
print("\nX_train:\n")
print(X_train)
print(X_train.shape)

print("\nX_test:\n")
print(X_test)
print(X_test.shape)


# import tensorflow as tf
# from tensorflow.keras import datasets, layers, models
# from keras.layers.convolutional import Conv1D, MaxPooling1D
# import matplotlib.pyplot as plt
# import numpy as np
# cnn = models.Sequential([
#         #cnn
#         layers.Conv1D(filters=32,kernel_size=3,activation='relu',input_shape=(1289031,1)),


#         #dense
#         layers.Flatten(),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(10, activation='softmax')
#     ])

# cnn.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# cnn.fit(X_train, y_train, epochs=10)
