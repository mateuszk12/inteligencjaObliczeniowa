from tensorflow import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D,Rescaling,Dropout,BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
#data stuff
ih = 512
iw = 512
seed_train = 1
shuffle = True
validation_split = 0.25
batch_size=6
pathTR = '/home/mati/studia/inteligencjaObliczeniowa/pracaDomowa2/data/brainTumorData/Training'
pathTS = '/home/mati/studia/inteligencjaObliczeniowa/pracaDomowa2/data/brainTumorData/Testing'
train_dataset = keras.utils.image_dataset_from_directory(pathTR,color_mode="grayscale",image_size=(ih,iw),shuffle=True,batch_size=batch_size)
test_dataset = keras.utils.image_dataset_from_directory(pathTS,color_mode="grayscale",image_size=(ih,iw),shuffle=True,batch_size=batch_size)


class_names = train_dataset.class_names
print(class_names)
num_classes = len(class_names)

model = Sequential()
model.add(Conv2D(32,(3,3), activation='relu',input_shape=(ih,iw,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes,activation="softmax"))

model.compile(optimizer=Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

epochs=10
history = model.fit(
  train_dataset,
  validation_data=test_dataset,
  epochs=epochs
)

model.save('./models')
