from tensorflow.python.estimator import keras
from src.helpers.helper_classifier import invert, convert


class Calc:
    def __init__(self):
        self.largest = 1000
        self.alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+', ' ']
        self.model = keras.models.load_model('../training/keras_classifier.h5')

    def solve(self, x1, operation, x2):
        x, y = convert(x1, x2, operation, self.alphabet, self.largest)
        result = self.model.predict(x)
        expected = [invert(x, self.alphabet) for x in y]
        predicted = [invert(x, self.alphabet) for x in result]
        return expected[0], predicted[0]
