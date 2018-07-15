import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D

def get_model(board_size):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(board_size, board_size, 3), padding="same"))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(1, (3, 3), activation='relu', padding="same"))
    model.add(Flatten())
    model.add(Dense(board_size**2, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    return model

def prepare_data(row, board_size):
    board = row[:board_size**2]
    flipped_board = row[board_size**2:2*board_size**2]
    turn = row[2*board_size**2:3*board_size**2]
    output = row[3*board_size**2:]

    board = np.array(board).reshape((board_size, board_size)).T
    flipped_board = np.array(flipped_board).reshape((board_size, board_size)).T
    turn = np.array(turn).reshape((board_size, board_size)).T
    output = np.array(output)

    state = np.zeros((board_size, board_size, 3))
    state[:,:,0] = board
    state[:,:,1] = flipped_board
    state[:,:,2] = turn

    return state, output

filename = 'domineering.csv'
df = pd.read_csv(filename, sep=',', na_values=[""], header=None)
board_size = 8

data = df.values.tolist()
train = data[:int(0.7*len(data))]
test = data[int(0.7*len(data)):]

X = []

for t in train:
    row = prepare_data(t, board_size)
    X.append(row)

X_train = np.array([k[0] for k in X])
Y_train = np.array([k[1] for k in X])

XT = []

for t in test:
    row = prepare_data(t, board_size)
    XT.append(row)

X_test = np.array([k[0] for k in XT])
Y_test = np.array([k[1] for k in XT])

model = get_model(board_size)

model.fit(X_train, Y_train, validation_split=0.1, epochs=20)
pred_test = model.predict(X_test)
acc = accuracy_score(np.argmax(Y_test, axis=1), np.argmax(pred_test, axis=1))
print("Accuracy : %s " %acc)