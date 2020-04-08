import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image


def classify_img(path, model):
    x = image.img_to_array(image.load_img(path, target_size=(28, 28), color_mode="grayscale")).reshape(1, 784)
    return model.predict((255 - x) / 255), np.argmax(model.predict((255 - x) / 255))


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Flatten())
model.add(Dense(800, activation='relu'))
model.add(Dense(10, activation='softmax'))

optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

h = model.fit(train_images, train_labels, epochs=5, batch_size=100, validation_data=(test_images, test_labels), verbose=0)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
print(np.argmax(test_labels[0]),np.argmax(model.predict(test_images)[0]))

print(classify_img("8.jpg", model))

plt.figure(1, figsize=(8, 5))
plt.title("Training and test accuracy")
plt.plot(h.history['acc'], 'r', label='train')
plt.plot(h.history['val_acc'], 'b', label='test')
plt.legend()
plt.show()
plt.clf()

plt.figure(1, figsize=(8, 5))
plt.title("Training and test loss")
plt.plot(h.history['loss'], 'r', label='train')
plt.plot(h.history['val_loss'], 'b', label='test')
plt.legend()
plt.show()
plt.clf()
