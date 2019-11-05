#!/usr/bin/env python

"""
Classification:  Performance  of  the  Naive  Bayes  algorithm on  the  given  data  set.
Run  the Naive Bayes algorithm on pre-processed version of train_gr_smpl.
Explain the reason for choosing and using pre-processed data.
Once you can run the algorithm, record, compare and analyse the classifier’s accuracy on different classes.
Save confusion Matrix
"""
import csv

import pandas as pd
import numpy as np
import cv2 as cv


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
        image = image[first_y:first_y + clmn, first_x:first_x + row]
        copped_dataset.append(image)
    return copped_dataset


def reshape_dataset(dataset):
    reshaped_dataset = []
    for image in dataset:
        image = cv.resize(image, (48, 48)) # un po' bruttino
        reshaped_dataset.append(image)
    return reshaped_dataset


def apply_adaptive_threshold(dataset):
    dataset_with_filter = []
    for image in dataset:
        image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        image = image.flatten()
        dataset_with_filter.append(image)
    dataset_with_filter = np.reshape(dataset_with_filter, (12660, 1600))
    return dataset_with_filter


def get_matrix_from_array(dataset):
    dataset_flatten = []
    for image in dataset:
        image = image.flatten()
        dataset_flatten.append(image)
    if len(image) == 40:
        dataset_flatten = np.reshape(dataset_flatten, (12660, 1600))
    if len(image) == 48:
        dataset_flatten = np.reshape(dataset_flatten, (12660, 2304))
    return dataset_flatten


def get_csvfile(dataset, labels, name):
    dataset = np.append(dataset, labels, axis=1)
    headers = []
    columns = len(dataset[0])
    for i in range(columns):
        i += 1
        headers.append(i)

    with open(name, mode='w') as file:
        my_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        my_writer.writerow(headers)
        for row in dataset:
            my_writer.writerow(row)

    return print("done")


def main():

    # Load datasets

    x_train_gr_smpl = pd.read_csv("./datasets/x_train_gr_smpl.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl = pd.read_csv("./datasets/y_train_smpl.csv", delimiter=",", dtype=np.uint8)

    # Dataset with 0 and 1 labels (0 = class; 1 = no class)
    y_train_smpl_0 = pd.read_csv("./datasets/y_train_smpl_0.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl_1 = pd.read_csv("./datasets/y_train_smpl_1.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl_2 = pd.read_csv("./datasets/y_train_smpl_2.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl_3 = pd.read_csv("./datasets/y_train_smpl_3.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl_4 = pd.read_csv("./datasets/y_train_smpl_4.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl_5 = pd.read_csv("./datasets/y_train_smpl_5.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl_6 = pd.read_csv("./datasets/y_train_smpl_6.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl_7 = pd.read_csv("./datasets/y_train_smpl_7.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl_8 = pd.read_csv("./datasets/y_train_smpl_8.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl_9 = pd.read_csv("./datasets/y_train_smpl_9.csv", delimiter=",", dtype=np.uint8)

    dataset = np.asmatrix(x_train_gr_smpl)

    aom_dataset = get_array_of_matrix(dataset)
    # plt.imshow(aom_dataset[0], cmap="gray")
    # plt.show()
    # print(aom_dataset)

    cropped_dataset = crop_dataset(aom_dataset, 40, 40) # un po' bruttino
    # plt.imshow(cropped_dataset[0], cmap="gray")
    # plt.show()
    # print(cropped_dataset)
    # new_dataset = reshape_dataset(cropped_dataset)

    threshold_dataset = apply_adaptive_threshold(cropped_dataset)
    # plt.imshow(np.reshape(new_dataset[0], (40, 40)), cmap="gray")
    # plt.show()

    smaller_dataset = threshold_dataset[::2]
    smaller_labels = y_train_smpl[::2]
    threshold_dataset = get_matrix_from_array(threshold_dataset)
    cropped_dataset = get_matrix_from_array(cropped_dataset)
    original_dataset = get_matrix_from_array(aom_dataset)

    # get csv files of smaller datasets
    get_csvfile(smaller_dataset, smaller_labels, "dataset_smaller_50%_withFilter.csv")

    # get csv of cropped datasets
    get_csvfile(cropped_dataset, y_train_smpl_0, "datasets_cropped_labelled_0.csv")
    get_csvfile(cropped_dataset, y_train_smpl_1, "datasets_cropped_labelled_1.csv")
    get_csvfile(cropped_dataset, y_train_smpl_2, "datasets_cropped_labelled_2.csv")
    get_csvfile(cropped_dataset, y_train_smpl_3, "datasets_cropped_labelled_3.csv")
    get_csvfile(cropped_dataset, y_train_smpl_4, "datasets_cropped_labelled_4.csv")
    get_csvfile(cropped_dataset, y_train_smpl_5, "datasets_cropped_labelled_5.csv")
    get_csvfile(cropped_dataset, y_train_smpl_6, "datasets_cropped_labelled_6.csv")
    get_csvfile(cropped_dataset, y_train_smpl_7, "datasets_cropped_labelled_7.csv")
    get_csvfile(cropped_dataset, y_train_smpl_8, "datasets_cropped_labelled_8.csv")
    get_csvfile(cropped_dataset, y_train_smpl_9, "datasets_cropped_labelled_9.csv")

    # get csv of fltered datasets
    get_csvfile(threshold_dataset, y_train_smpl_0, "datasets_withFilter_labelled_0.csv")
    get_csvfile(threshold_dataset, y_train_smpl_1, "datasets_withFilter_labelled_1.csv")
    get_csvfile(threshold_dataset, y_train_smpl_2, "datasets_withFilter_labelled_2.csv")
    get_csvfile(threshold_dataset, y_train_smpl_3, "datasets_withFilter_labelled_3.csv")
    get_csvfile(threshold_dataset, y_train_smpl_4, "datasets_withFilter_labelled_4.csv")
    get_csvfile(threshold_dataset, y_train_smpl_5, "datasets_withFilter_labelled_5.csv")
    get_csvfile(threshold_dataset, y_train_smpl_6, "datasets_withFilter_labelled_6.csv")
    get_csvfile(threshold_dataset, y_train_smpl_7, "datasets_withFilter_labelled_7.csv")
    get_csvfile(threshold_dataset, y_train_smpl_8, "datasets_withFilter_labelled_8.csv")
    get_csvfile(threshold_dataset, y_train_smpl_9, "datasets_withFilter_labelled_9.csv")

    # get csv of original datasets
    get_csvfile(original_dataset, y_train_smpl_0, "datasets_original_labelled_0.csv")
    get_csvfile(original_dataset, y_train_smpl_1, "datasets_original_labelled_1.csv")
    get_csvfile(original_dataset, y_train_smpl_2, "datasets_original_labelled_2.csv")
    get_csvfile(original_dataset, y_train_smpl_3, "datasets_original_labelled_3.csv")
    get_csvfile(original_dataset, y_train_smpl_4, "datasets_original_labelled_4.csv")
    get_csvfile(original_dataset, y_train_smpl_5, "datasets_original_labelled_5.csv")
    get_csvfile(original_dataset, y_train_smpl_6, "datasets_original_labelled_6.csv")
    get_csvfile(original_dataset, y_train_smpl_7, "datasets_original_labelled_7.csv")
    get_csvfile(original_dataset, y_train_smpl_8, "datasets_original_labelled_8.csv")
    get_csvfile(original_dataset, y_train_smpl_9, "datasets_original_labelled_9.csv")



if __name__ == "__main__":
    main()
