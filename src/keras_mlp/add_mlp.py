from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

from helpers.helper import random_sum_pairs, convert_result, print_results

n_numbers = 2
largest = 1000
# Build the MLP
model = Sequential([
    Dense(4, input_dim=2),
    Dense(2),
    Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam')
# Train the model
x, y = random_sum_pairs(5000, n_numbers, largest * 10)
model.fit(x, y, epochs=50, verbose=2)
# Evaluate on some new patterns
x, y = random_sum_pairs(30, n_numbers, largest)
result = model.predict(x, verbose=0)
# Display results
expected, predicted, rmse = convert_result(result, n_numbers, largest, y)
print_results(5, expected, predicted, rmse)
