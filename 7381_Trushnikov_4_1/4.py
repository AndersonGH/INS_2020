import numpy as np
from math import exp
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt


def logicFunc(a, b, c):
    return (a and b) or (a and c)


def getYtrain(x_train):
    mas = []
    for elem in x_train:
        mas.append(logicFunc(*elem))
    return np.array(mas)


def relu(x):
    return np.maximum(x, 0.)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_by_tensor(x_train, weights):
    for j in range(0, len(weights)):
        if j == len(weights) - 1:
            res = sigmoid((np.dot(res, weights[j][0]) + weights[j][1]))
        else:
            res = relu((np.dot(res, weights[j][0]) + weights[j][1]))
    return res

def predict_by_elem(x_train, weights):
    answer = np.array([])
    for m in range(0, x_train.shape[0]):
        copy_x_train = x_train.copy()[m]
        res = []
        for j in range(0, len(weights)):
            res = np.zeros(weights[j][0].shape[1])
            for i in range(0, weights[j][0].shape[1]):
                sum = 0
                for k in range(0, weights[j][0].shape[0]):
                    sum = sum + weights[j][0][k][i] * copy_x_train[k]
                if j == len(weights) - 1:
                    res[i] = sigmoid(sum + weights[j][1][i])
                else:
                    res[i] = relu(sum + weights[j][1][i])
            copy_x_train = res
        answer = np.append(answer, res)
    return answer.reshape((len(x_train), 1))


def main():
    x_train = np.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 0],
                        [1, 1, 1]])
    y_train = getYtrain(x_train)

    model = Sequential()
    model.add(Dense(5, activation='relu', input_shape=(3,)))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    weights = []
    for layer in model.layers:
        weights.append(layer.get_weights())
    print("<predict_by_elem> Before: ", predict_by_elem(x_train, weights))
    print("<predict_by_tensor> Before: ", predict_by_tensor(x_train, weights))

    model.fit(x_train, y_train, epochs=150, batch_size=1)
    weights = []
    for layer in model.layers:
        weights.append(layer.get_weights())
    print("model.predict ",model.predict(x_train))
    print("<predict_by_elem> After: ", predict_by_elem(x_train, weights))
    print("<predict_by_tensor> After: ",predict_by_tensor(x_train, weights))


if __name__ == '__main__':
    main()
