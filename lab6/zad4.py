import tensorflow as tf


from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from ann_visualizer.visualize import ann_viz;
import matplotlib.pyplot as plt


df = pd.read_csv("./diabetes.csv")
df_norm = df[
    ['pregnant-times', 'glucose-concentr', 'blood-pressure', 'skin-thickness', 'insulin', 'mass-index', 'pedigree-func',
     'age']]
target = df[['class']].replace(['tested_positive', 'tested_negative'], [0, 1])
df = pd.concat([df_norm, target], axis=1)
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=123)
train_inputs = train_set[:, 0:8]
train_classes = train_set[:, 8]
test_inputs = test_set[:, 0:8]
test_classes = test_set[:, 8]
model = Sequential()

model = Sequential()
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(train_inputs,
          train_classes,
          validation_data=(test_inputs, test_classes),
          epochs=500,
          batch_size=10)
ann_viz(model, title="diabetes neural network")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss curve')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','test'], loc='upper left')
plt.show()
