import keras
from pandas import np
import tensorflow.keras.callbacks
from matplotlib import pyplot as plt


class MyCallBack(tensorflow.keras.callbacks.Callback):

    def __init__(self, x_train, y_train):
        super(MyCallBack, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.values = []

    def on_epoch_end(self, epoch, logs=None):
        count = 0
        predicted = self.model.predict(self.x_train)
        for i in range(len(predicted)):
            index = np.argmax(predicted[i])
            if self.y_train[i][index] == 1 and predicted[i][index] <= 0.90:
                count += 1
        self.values.append(count)

    def on_train_end(self, logs=None):
        print(self.values)
        plt.plot(self.values)
        plt.ylabel('Count')
        plt.xlabel('epoch')
        plt.savefig('foo.png')
        plt.show()

