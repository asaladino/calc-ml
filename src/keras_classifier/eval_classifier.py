from tensorflow.python.estimator import keras

from src.helpers.helper_classifier import generate_data, invert, convert

model = keras.models.load_model('../training/keras_classifier.h5')

n_samples = 5
largest = 1000
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+', ' ']

# x, y = generate_data(n_samples, largest, alphabet)

# evaluate on some new patterns
x, y = convert(30, 40, '+', alphabet, largest)
result = model.predict(x)

expected = [invert(x, alphabet) for x in y]
predicted = [invert(x, alphabet) for x in result]

print('Expected=%s, Predicted=%s' % (expected[0], predicted[0]))
