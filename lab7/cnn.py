import numpy as np
# import tensorflow as tf
# from tensorflow import keras
from matplotlib import pyplot as plt

# tworzymy tablice o wymiarach 128x128x3 (3 kanaly to RGB)
# uzupelnioną zerami = kolor czarny
data = np.zeros((128, 128, 3), dtype=np.uint8)



# chcemy zeby obrazek byl czarnobialy,
# wiec wszystkie trzy kanaly rgb uzupelniamy tymi samymi liczbami
# napiszmy do tego funkcje
def draw(img, x, y, color):
    img[x, y] = [color, color, color]


# zamalowanie 4 pikseli w lewym górnym rogu
draw(data, 5, 5, 200)
draw(data, 6, 6, 100)
draw(data, 5, 6, 255)
draw(data, 6, 5, 255)

# rysowanie kilku figur na obrazku
for i in range(128):
    for j in range(128):
        if (i - 64) ** 2 + (j - 64) ** 2 < 900:
            draw(data, i, j, 200)
        elif i > 100 and j > 100:
            draw(data, i, j, 255)
        elif (i - 15) ** 2 + (j - 110) ** 2 < 25:
            draw(data, i, j, 150)
        elif (i - 15) ** 2 + (j - 110) ** 2 == 25 or (i - 15) ** 2 + (j - 110) ** 2 == 26:
            draw(data, i, j, 255)
kernelV = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])
kernelH = np.array([[1, 1, 1],
                   [0, 0, 0],
                   [-1,-1,-1]])
kernel45 = np.array([[0, 1, 2],
                   [-1, 0, 1],
                   [-2,-1,0]])
stride = 1
output = np.zeros((126, 126, 3), dtype=np.uint8)
for c in range(3):
    for i in range(0,len(data),stride):
        for j in range(0,len(data),stride):
            if i+3 < len(data) and j+3 < len(data):
                arr = np.array(data[i:i+3,j:j+3,c])
                d = np.sum(arr*kernel45)
                output[i][j] = [d,d,d]

# konwersja macierzy na obrazek i wyświetlenie
plt.imshow(output, interpolation='nearest')
plt.show()
