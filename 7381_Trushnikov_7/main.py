import numpy as np
import string
import matplotlib.pyplot as plt
from keras import optimizers
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from keras.models import Sequential
from tensorflow.keras import regularizers
from keras.layers import Dropout, Dense, Embedding, GRU, Conv1D, MaxPooling1D, SimpleRNN, Bidirectional
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb




def prepInputText(text, target):

    def genNum(data, dic):
        data = data.translate(str.maketrans(dict.fromkeys(string.punctuation))).split()
        for i in range(len(data)):
            num = dic.get(data[i])
            if (num == None):
                data[i] = 0
            else:
                data[i] = num
        return data
    dic = dict(imdb.get_word_index())
    test_x = []
    test_y = np.array(target).astype("float32")
    for i in range(0, len(text)):
        test_x.append(genNum(text[i], dic))
    test_x = sequence.pad_sequences(test_x, maxlen=max_review_length)
    return test_x, test_y


def ensemble(models, test_x, test_y):
    preds = np.array(models[0].predict(test_x))
    models[0].evaluate(test_x, test_y, verbose=2)
    for i in range(1, len(models)):
        preds = preds + np.array(models[i].predict(test_x))
        models[i].evaluate(test_x, test_y, verbose=2)
    print(preds / len(models))
    print(accuracy_score(test_y, (preds / len(models)).round(), normalize=False) / 100)


def getModel(max_review_length, embedding_vecor_length, max_features, num_model):
    optimizer = optimizers.RMSprop(lr=0.001, rho=0.9)
    model = Sequential()
    model.add(Embedding(max_features, embedding_vecor_length, input_length=max_review_length))
    model.add(Dropout(0.3))
    model.add(
        Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(
        Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(GRU(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(GRU(32))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    h = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=3, batch_size=64)
    scores = model.evaluate(test_x, test_y, verbose=0)
    accuracy = scores[1] * 100
    print("Accuracy: %.2f%%" % (accuracy))

    return model, accuracy, h


def best_bodel():
    num_model = 14
    accuracy = 0
    for i in range(10):
        print(i)
        model, acc, h = getModel(max_review_length, embedding_vecor_length, max_features, num_model)
        if acc > accuracy:
            accuracy = acc
            model.save('model' + str(num_model) + '_' + str(round(accuracy, 2)) + '%.h5')
            model.save_weights('weights' + str(num_model) + '.hdf5')
            plt.figure(1, figsize=(8, 5))
            plt.title("Training and test accuracy")
            plt.plot(h.history['acc'], 'r', label='train')
            plt.plot(h.history['val_acc'], 'b', label='test')
            plt.legend()
            plt.savefig('accuracy' + str(num_model) + '.png')
            # plt.show()
            plt.clf()

            plt.figure(1, figsize=(8, 5))
            plt.title("Training and test loss")
            plt.plot(h.history['loss'], 'r', label='train')
            plt.plot(h.history['val_loss'], 'b', label='test')
            plt.legend()
            plt.savefig('loss' + str(num_model) + '.png')
            # plt.show()
            plt.clf()


max_review_length = 500
embedding_vecor_length = 32
max_features = 10000

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=max_features)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

data = sequence.pad_sequences(data, maxlen=max_review_length)
targets = np.array(targets).astype("float32")

train_x = data[max_features:]
train_y = targets[max_features:]

test_x = data[:max_features]
test_y = targets[:max_features]

# getModel(max_review_length, embedding_vecor_length, max_features)

models = []

models.append(load_model('model1\model1.h5'))
models.append(load_model('model2\model2.h5'))
models.append(load_model('model6\model6.h5'))
models.append(load_model('model7\model7.h5'))


data_x, data_y = prepInputText(["Cracking. Keeps attention from start to finish"],[1])

ensemble(models, data_x, data_y  )

# best_bodel()

