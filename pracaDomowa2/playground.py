from tensorflow import keras
import tensorflow as tf
import numpy as np
#model load
ih,iw = 512,512
#easier
#class_names = ['glioma', 'meningioma', 'neurocitoma', 'notumor', 'otherInjury', 'pituitary', 'schwannoma']
#harder
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
model = keras.models.load_model("models")
img = tf.keras.utils.load_img(
    '/home/mati/studia/inteligencjaObliczeniowa/pracaDomowa2/data/brainTumorData/Training/meningioma/Tr-me_0024.jpg',color_mode="grayscale" ,target_size=(ih,iw)
)
img2 = tf.keras.utils.load_img(
    '/home/mati/studia/inteligencjaObliczeniowa/pracaDomowa2/data/mri.jpeg',color_mode="grayscale" ,target_size=(ih,iw)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) 

img_array2 = tf.keras.utils.img_to_array(img2)
img_array2 = tf.expand_dims(img_array2, 0)

predictions = model.predict(img_array)
predictions2 = model.predict(img_array2)
score = tf.nn.softmax(predictions[0])
score2 = tf.nn.softmax(predictions2[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)),
    "This image2 most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score2)], 100 * np.max(score2))
)