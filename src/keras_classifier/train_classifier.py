from math import ceil
from math import log10
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

from src.helpers.helper_classifier import generate_data, invert

n_samples = 10000
n_numbers = 2
largest = 10000
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+', ' ']
n_chars = len(alphabet)
n_in_seq_length = n_numbers * ceil(log10(largest + 1)) + n_numbers - 1
n_out_seq_length = ceil(log10(n_numbers * (largest + 1)))
n_batch = 10
n_epoch = 200
model = Sequential([
    LSTM(100, input_shape=(n_in_seq_length, n_chars)),
    RepeatVector(n_out_seq_length),
    LSTM(50, return_sequences=True),
    TimeDistributed(Dense(n_chars, activation='softmax'))
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
for i in range(n_epoch):
    x, y = generate_data(n_samples, largest, alphabet)
    model.fit(x, y, epochs=1, batch_size=n_batch)

model.save('training/keras_classifier.h5')

# evaluate on some new patterns
x, y = generate_data(n_samples, largest, alphabet)
result = model.predict(x, batch_size=n_batch, verbose=0)
# calculate error
expected = [invert(x, alphabet) for x in y]
predicted = [invert(x, alphabet) for x in result]
# show some examples
for i in range(20):
    print('Expected=%s, Predicted=%s' % (expected[i], predicted[i]))
