from preprocessing import Preprocessing
import numpy as np
import cv2
import os


def list_files(path, valid_exts=('.jpg', '.jpeg', '.png'), contains=None):

    for (dirpath, dirnames, filenames) in os.walk(path):

        for filename in filenames:

            if contains is not None and filename.find(contains) == -1:
                continue

            ext = filename[filename.rfind('.'):].lower()

            if ext.endswith(valid_exts):
                imagepath = os.path.join(dirpath, filename).replace(' ', '\\')
                yield imagepath


def dataset_loader(image_paths, verbose=-1, preprocessing=None, c=None):

    data, labels = [], []

    if preprocessing is not None:
        if len(preprocessing) == 3:
            height, width, interpolation = preprocessing
            preprocessing = Preprocessing(height, width, interpolation)
        else:
            height, width = preprocessing
            preprocessing = Preprocessing(height, width)

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        if c == 'gray':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = image_path.split(os.path.sep)[-2]

        if preprocessing is not None:
            image = preprocessing.preprocess(image)

        data.append(image)
        labels.append(label)

        if verbose > 0 and i > 0 and (i+1) % verbose == 0:
            print(f'[INFO] Processed {i+1}/{len(image_paths)}')

    return np.array(data), np.array(labels)
