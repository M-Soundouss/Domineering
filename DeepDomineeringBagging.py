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
    output = np.fliplr(output).T
    output = output.ravel()

    return state, output

def prepare_data_ud(row, board_size):
    data = prepare_data(row, board_size)
    state = data[0]
    state = np.flipud(state)

    output = data[1]
    output = np.array(output).reshape((board_size, board_size)).T
    output = np.flipud(output).T
    output = output.ravel()

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

X_flipped_lr = []

for t in train:
    row_lr = prepare_data_lr(t, board_size)
    X_flipped_lr.append(row_lr)

X_flipped_ud = []

for t in train:
    row_ud = prepare_data_ud(t, board_size)
    X_flipped_ud.append(row_ud)

X_train = np.array([k[0] for k in X])
Y_train = np.array([k[1] for k in X])

X_flipped_lr_train = np.array([k[0] for k in X_flipped_lr])
Y_flipped_lr_train = np.array([k[1] for k in X_flipped_lr])

X_flipped_ud_train = np.array([k[0] for k in X_flipped_ud])
Y_flipped_ud_train = np.array([k[1] for k in X_flipped_ud])

XT = []
XT_flipped_lr = []
XT_flipped_ud = []

for t in test:
    row = prepare_data(t, board_size)
    XT.append(row)
    row_lr = prepare_data_lr(t, board_size)
    XT_flipped_lr.append(row_lr)
    row_ud = prepare_data_ud(t, board_size)
    XT_flipped_ud.append(row_ud)

X_test = np.array([k[0] for k in XT])
Y_test = np.array([k[1] for k in XT])

X_test_flipped_lr = np.array([k[0] for k in XT_flipped_lr])
Y_test_flipped_lr = np.array([k[1] for k in XT_flipped_lr])

X_test_flipped_ud = np.array([k[0] for k in XT_flipped_ud])
Y_test_flipped_ud = np.array([k[1] for k in XT_flipped_ud])

model = get_model(board_size)
model.fit(X_train, Y_train, validation_split=0.1, epochs=20)
pred_test = model.predict(X_test)

model_flipped_lr = get_model(board_size)
model_flipped_lr.fit(X_flipped_lr_train, Y_flipped_lr_train, validation_split=0.1, epochs=20)
pred_flipped_lr_test = model_flipped_lr.predict(X_test_flipped_lr)

model_flipped_ud = get_model(board_size)
model_flipped_ud.fit(X_flipped_ud_train, Y_flipped_ud_train, validation_split=0.1, epochs=20)
pred_flipped_ud_test = model_flipped_ud.predict(X_test_flipped_ud)

pred_flipped_lr_test_reflipped = []
for elmt in pred_flipped_lr_test:
    elmt = np.array(elmt).reshape((board_size, board_size)).T
    elmt = np.fliplr(elmt).T.ravel()
    pred_flipped_lr_test_reflipped.append(elmt)

pred_flipped_ud_test_reflipped = []
for elmt in pred_flipped_ud_test:
    elmt = np.array(elmt).reshape((board_size, board_size)).T
    elmt = np.flipud(elmt).T.ravel()
    pred_flipped_ud_test_reflipped.append(elmt)

pred_flipped_lr_test_reflipped = np.array(pred_flipped_lr_test_reflipped)
pred_flipped_lr = (pred_test + pred_flipped_lr_test_reflipped)/2

pred_flipped_ud_test_reflipped = np.array(pred_flipped_ud_test_reflipped)
pred_flipped_ud = (pred_test + pred_flipped_ud_test_reflipped)/2

pred_flipped_udlr = (pred_test + pred_flipped_ud_test_reflipped + pred_flipped_lr_test_reflipped)/3

acc = accuracy_score(np.argmax(Y_test, axis=1), np.argmax(pred_test, axis=1))
print("Accuracy : %s " %acc)

acc_lr = accuracy_score(np.argmax(Y_test, axis=1), np.argmax(pred_flipped_lr, axis=1))
print("Accuracy Bagging lr: %s " %acc_lr)

acc_ud = accuracy_score(np.argmax(Y_test, axis=1), np.argmax(pred_flipped_ud, axis=1))
print("Accuracy Bagging ud: %s " %acc_ud)

acc_udlr = accuracy_score(np.argmax(Y_test, axis=1), np.argmax(pred_flipped_udlr, axis=1))
print("Accuracy Bagging lrud: %s " %acc_udlr)
