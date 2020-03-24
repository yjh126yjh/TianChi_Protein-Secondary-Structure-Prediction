import numpy as np

from keras.models import Sequential
from keras.layers import Dropout, Conv1D, MaxPooling1D
from keras import optimizers, callbacks

import dataset

do_summary = True

LR = 0.00005
drop_out = 0.4
batch_dim = 128
nn_epochs = 300


loss = 'categorical_crossentropy'

early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='min')

# filepath="NewModel-{epoch:02d}-{val_acc:.2f}.hdf5"
filepath = "whole_sequence-best.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


def Q8_accuracy(real, pred):
    total = real.shape[0] * real.shape[1]
    correct = 0
    for i in range(real.shape[0]):
        for j in range(real.shape[1]):
            if np.sum(real[i, j, :]) == 0:
                total = total - 1
            else:
                if real[i, j, np.argmax(pred[i, j, :])] > 0:
                    correct = correct + 1

    return correct / total


def CNN_model():
    m = Sequential()
    m.add(Conv1D(1024, 37, padding='same', activation='relu', input_shape=(dataset.sequence_len, dataset.amino_acid_residues)))
    m.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
    m.add(Dropout(drop_out))

    m.add(Conv1D(512, 37, padding='same', activation='relu'))
    m.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
    m.add(Dropout(drop_out))

    m.add(Conv1D(256, 37, padding='same', activation='relu'))
    m.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
    m.add(Dropout(drop_out))

    m.add(Conv1D(dataset.num_classes, 37, padding='same', activation='softmax'))

    opt = optimizers.Adam(lr=LR)
    m.compile(optimizer=opt,
              loss=loss,
              metrics=['accuracy', 'mae'])
    if do_summary:
        print("\nHyper Parameters\n")
        print("Learning Rate: " + str(LR))
        print("Drop out: " + str(drop_out))
        print("Batch dim: " + str(batch_dim))
        print("Number of epochs: " + str(nn_epochs))
        print("\nLoss: " + loss + "\n")
        m.summary()
    return m