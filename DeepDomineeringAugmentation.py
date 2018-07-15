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

def prepare_data_lr(row, board_size):
    data = prepare_data(row, board_size)
    state = data[0]
    state = np.fliplr(state)

    output = data[1]
    output = np.array(output).reshape((board_size, board_size)).T
    output = np.fliplr(output)

    player = int(state[:, :, 2][0][0])
    if player == 0:  # vertical
        output = output.T
        output = output.ravel()
        return state, output
    if player == 1:  # horizental
        permutation = [7, 0, 1, 2, 3, 4, 5, 6]
        i = np.argsort(permutation)
        output_moved = output[:, i]
        output_moved = output_moved.T
        output_moved = output_moved.ravel()
        return state, output_moved

def prepare_data_ud(row, board_size):
    data = prepare_data(row, board_size)
    state = data[0]
    state = np.flipud(state)

    output = data[1]
    output = np.array(output).reshape((board_size, board_size)).T
    output = np.flipud(output)

    player = int(state[:, :, 2][0][0])
    if player == 1:  # horizental
        output = output.T
        output = output.ravel()
        return state, output
    if player == 0:  # vertical
        output_moved = np.roll(output, 7, axis=0)
        output_moved = output_moved.T
        output_moved = output_moved.ravel()
        return state, output_moved

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
    row_lr = prepare_data_lr(t, board_size)
    X.append(row_lr)
    row_ud = prepare_data_ud(t, board_size)
    X.append(row_ud)

X_train = np.array([k[0] for k in X])
Y_train = np.array([k[1] for k in X])

XT = []

for t in test:
    row = prepare_data(t, board_size)
    XT.append(row)
    row_lr = prepare_data_lr(t, board_size)
    XT.append(row_lr)
    row_ud = prepare_data_ud(t, board_size)
    X.append(row_ud)

X_test = np.array([k[0] for k in XT])
Y_test = np.array([k[1] for k in XT])

model = get_model(board_size)

model.fit(X_train, Y_train, validation_split=0.1, epochs=20)
pred_test = model.predict(X_test)
acc = accuracy_score(np.argmax(Y_test, axis=1), np.argmax(pred_test, axis=1))
print("Accuracy : %s " %acc)