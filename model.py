import numpy as np
import pandas as pd
import os
import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import regularizers
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

train_dir = 'blackmamba/ASL/'

def load_unique():
    size_img = 64,64
    images_for_plot = []
    labels_for_plot = []
    for folder in os.listdir(train_dir):
        for file in os.listdir(train_dir + '/' + folder):
            filepath = train_dir + '/' + folder + '/' + file
            image = cv2.imread(filepath)
            final_img = cv2.resize(image, size_img)
            final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
            images_for_plot.append(final_img)
            labels_for_plot.append(folder)
            break
    return images_for_plot, labels_for_plot

images_for_plot, labels_for_plot = load_unique()
print("unique_labels = ", labels_for_plot)

fig = plt.figure(figsize = (15,15))
def plot_images(fig, image, label, row, col, index):
    fig.add_subplot(row, col, index)
    plt.axis('off')
    plt.imshow(image)
    plt.title(label)
    return

image_index = 0
row = 5
col = 6
# for i in range(1,(row*col)):
#     plot_images(fig, images_for_plot[image_index], labels_for_plot[image_index], row, col, i)
#     image_index = image_index + 1
# plt.show()

labels_dict = {'0': 29, '1': 30, '2': 31, '3': 32, '4': 33,
               '5': 34, '6': 35, '7': 36, '8': 37, '9': 38, 'Best of Luck': 39, 'You': 40, 'Me': 41, 'Like': 42,
               'Remember': 43, 'Love': 44, 'Fuck': 45, 'I love you': 46, }


def load_data():
    images = []
    labels = []
    size = 64, 64
    print("LOADING DATA FROM : ", end="")
    for folder in os.listdir(train_dir):
        print(folder, end=' | ')
        for image in os.listdir(train_dir + "/" + folder):
            temp_img = cv2.imread(train_dir + '/' + folder + '/' + image)
            temp_img = cv2.resize(temp_img, size)
            images.append(temp_img)
            if folder == '0':
                labels.append(labels_dict['0'])
            elif folder == '1':
                labels.append(labels_dict['1'])
            elif folder == '2':
                labels.append(labels_dict['2'])
            elif folder == '3':
                labels.append(labels_dict['3'])
            elif folder == '4':
                labels.append(labels_dict['4'])
            elif folder == '5':
                labels.append(labels_dict['5'])
            elif folder == '6':
                labels.append(labels_dict['6'])
            elif folder == '7':
                labels.append(labels_dict['7'])
            elif folder == '8':
                labels.append(labels_dict['8'])
            elif folder == '9':
                labels.append(labels_dict['9'])
            elif folder == 'Best of Luck':
                labels.append(labels_dict['Best of Luck'])
            elif folder == 'You':
                labels.append(labels_dict['You'])
            elif folder == 'Me':
                labels.append(labels_dict['Me'])
            elif folder == 'Like':
                labels.append(labels_dict['Like'])
            elif folder == 'Remember':
                labels.append(labels_dict['Remember'])
            elif folder == 'Love':
                labels.append(labels_dict['Love'])
            elif folder == 'Fuck':
                labels.append(labels_dict['Fuck'])
            elif folder == 'I love you':
                labels.append(labels_dict['I love you'])

    images = np.array(images)
    images = images.astype('float32') / 255.0

    templabels = labels

    labels = keras.utils.to_categorical(labels)

    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.05)

    print()
    print('Loaded', len(X_train), 'images for training,', 'Train data shape =', X_train.shape)
    print('Loaded', len(X_test), 'images for testing', 'Test data shape =', X_test.shape)

    return X_train, X_test, Y_train, Y_test, templabels

X_train, X_test, Y_train, Y_test, labels = load_data()


def create_model():
    model = Sequential()

    model.add(Conv2D(16, kernel_size=[3, 3], padding='same', activation='relu', input_shape=(64, 64, 3)))
    model.add(Conv2D(32, kernel_size=[3, 3], padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=[3, 3]))

    model.add(Conv2D(32, kernel_size=[3, 3], padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=[3, 3], padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=[3, 3]))

    model.add(Conv2D(128, kernel_size=[3, 3], padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=[3, 3]))

    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(47, activation='softmax'))

    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])

    print("MODEL CREATED")
    model.summary()

    return model


def fit_model():
    model_hist = model.fit(X_train, Y_train, batch_size=64, epochs=4, validation_split=0.1)
    return model_hist

model = create_model()
curr_model_hist = fit_model()

model.save('digit_model.h5')
