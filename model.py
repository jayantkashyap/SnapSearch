from keras.applications.inception_v3 import InceptionV3
from keras.layers import Layer, Dense, Conv2D
from preprocessing import  Preprocessor

import numpy as np

class Model(object):

    def __init__(self):
        self._model = None

    def getModel(self, dataset='imagenet'):
        self._model = InceptionV3(include_top=True, weights=dataset)
        return self._model

    def train(self):
        pass

    def predict(self, image):
        if self._model is not None:
                self._model.predict()
