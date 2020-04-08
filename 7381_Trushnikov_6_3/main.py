import gens
import numpy as np
import keras
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


def gen_data(size=500, img_size=50):
    c1 = size // 2
    c2 = size - c1
    label_c1 = np.full([c1, 1], 'Horizontal')
    data_c1 = np.array([gens.gen_h_line(img_size) for i in range(c1)])
    label_c2 = np.full([c2, 1], 'Vertical')
    data_c2 = np.array([gens.gen_v_line(img_size) for i in range(c2)])
    data = np.vstack((data_c1, data_c2))
    label = np.vstack((label_c1, label_c2))
    return data, label


def getData(size=500, img_size=50):
    X, y = gen_data(size, img_size)
    X, y = shuffle(X, y)
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_trn, y_trn, X_tst, y_tst


X_train, y_train, X_test, y_test = getData()

X_train = X_train.reshape([-1, 50, 50, 1])  # make X_train fourth dimension
X_test = X_test.reshape([-1, 50, 50, 1])  # make X_test fourth dimension

encoder = LabelBinarizer()
Y_train = encoder.fit_transform(y_train)  # 1 - Vertical, 0 - Horizontal
Y_test = encoder.fit_transform(y_test)

Y_train = keras.utils.to_categorical(Y_train, 2)
Y_test = keras.utils.to_categorical(Y_test, 2)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
model.fit(X_train, Y_train,
          batch_size=15,
          epochs=12,
          validation_data=(X_test, Y_test))
score = model.evaluate(X_train, Y_train, verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
