#!/usr/bin/env python

"""
Classification:  Performance  of  the  Naive  Bayes  algorithm on  the  given  data  set.
Run  the Naive Bayes algorithm on pre-processed version of train_gr_smpl.
Explain the reason for choosing and using pre-processed data.
Once you can run the algorithm, record, compare and analyse the classifier’s accuracy on different classes.
Save confusion Matrix
"""


import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB


# Pre-processing methods for the dataset

def get_array_of_matrix(dataset):
    array_of_images = []
    for row in dataset:
        row = np.asarray(row)
        matrix = np.reshape(row, (48, 48))
        array_of_images.append(matrix)
    return array_of_images


def crop_dataset(dataset, row, clmn):
    copped_dataset = []
    for image in dataset:
        y, x = image.shape
        first_x = x//2-(row//2)
        first_y = y//2-(clmn//2)
        copped_dataset.append(image[first_y:first_y + clmn, first_x:first_x + row])
    return copped_dataset


def reshape_dataset(dataset):
    reshaped_dataset = []
    for image in dataset:
        image = cv.resize(image, (48, 48)) # un po' bruttino
        image = image.flatten()
        reshaped_dataset.append(image)
    reshaped_dataset = np.reshape(reshaped_dataset, (12660, 2304)) # un po' bruttino
    return reshaped_dataset

# method for Naive  Bayes  algorithm on new dataset

def main():
    x_train_gr_smpl = pd.read_csv("./datasets/x_train_gr_smpl.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl = pd.read_csv("./datasets/y_train_smpl.csv", delimiter=",", dtype=np.uint8)

    dataset = np.asmatrix(x_train_gr_smpl)
    aom_dataset = get_array_of_matrix(dataset)
    # plt.imshow(aom_dataset[0], cmap="gray")
    # plt.show()
    # print(aom_dataset)

    cropped_dataset = crop_dataset(aom_dataset, 40, 40) # un po' bruttino
    # plt.imshow(cropped_dataset[0], cmap="gray")
    # plt.show()
    # print(cropped_dataset)

    new_dataset = reshape_dataset(cropped_dataset)
    # plt.imshow(new_dataset[0], cmap="gray")
    # plt.show()

    # add y_train_smpl to new_dataset ---KARAN
    dataset = np.append(new_dataset, y_train_smpl, axis=1)
    x = dataset[:, 0:2303]
    y = dataset[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=17)

    # code for Naive Bayes algorithm
    # Gaussian
    GausNB = GaussianNB()
    GausNB.fit(X_train, y_train)
    print(GausNB)
    y_expect = y_test
    y_pred = GausNB.predict(X_test)
    print(accuracy_score(y_expect, y_pred))

    BernNB = BernoulliNB()
    BernNB.fit(X_train, y_train)
    print(BernNB)
    y_expect = y_test
    y_pred = BernNB.predict(X_test)
    print(accuracy_score(y_expect, y_pred))

    MultiNom = MultinomialNB()
    MultiNom.fit(X_train, y_train)
    print(MultiNom)
    y_expect = y_test
    y_pred = MultiNom.predict(X_test)
    print(accuracy_score(y_expect, y_pred))






if __name__ == "__main__":
    main()
