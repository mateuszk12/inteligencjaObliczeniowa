from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import tensorflow as tf
#data stuff
train_dataset = keras.utils.image_dataset_from_directory('/home/mati/studia/inteligencjaObliczeniowa/pracaDomowa2/data/brainTumorData/Training',color_mode="grayscale",image_size=(256,256),shuffle=True,batch_size=6)
test_dataset = keras.utils.image_dataset_from_directory('/home/mati/studia/inteligencjaObliczeniowa/pracaDomowa2/data/brainTumorData/Testing',color_mode="grayscale",image_size=(256,256),shuffle=True,batch_size=6)

class_names = train_dataset.class_names


model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(256,256,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='softmax'))
#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model
model.fit(train_dataset, validation_data=(test_dataset), epochs=3)