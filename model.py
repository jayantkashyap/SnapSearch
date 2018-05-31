from keras.applications.inception_v3 import InceptionV3
from keras.layers import Layer, Dense, Conv2D
from keras.models import load_model, save_model
from preprocessing import Preprocessing


class Model(object):

    def __init__(self):
        self._model = None

    def get_model(self, dataset='imagenet'):
        self._model = InceptionV3(include_top=True, weights=dataset)
        return self._model

    def train(self):
        if self._model is not None:
            self._model.save_model('model.h5')

    def predict(self, image):
        if self._model is not None:
            image = Preprocessing(None, None).preprocess(image)
            self._model.predict(image)

    def load_model(self):
        self._model = load_model('model.h5')
        return self._model
