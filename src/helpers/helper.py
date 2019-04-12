from random import seed
from random import randint
from numpy import array
from sklearn.metrics import mean_squared_error
from math import sqrt

# generate training data
seed(1)


def random_sum_pairs(n_examples, n_numbers, largest):
    x, y = list(), list()
    for i in range(n_examples):
        in_pattern = [randint(1, largest) for _ in range(n_numbers)]
        out_pattern = sum(in_pattern)
        x.append(in_pattern)
        y.append(out_pattern)
    # format as NumPy arrays
    x, y = array(x), array(y)
    # normalize
    x = x.astype('float') / float(largest * n_numbers)
    y = y.astype('float') / float(largest * n_numbers)
    return x, y


# invert normalization
def invert(value, n_numbers, largest):
    return round(value * float(largest * n_numbers))


def convert_result(result, n_numbers, largest, y):
    expected = [invert(x, n_numbers, largest) for x in y]
    predicted = [invert(x, n_numbers, largest) for x in result[:, 0]]
    rmse = sqrt(mean_squared_error(expected, predicted))
    return expected, predicted, rmse


def print_results(count, expected, predicted, rmse):
    print('Root Mean Squared Error: %f' % rmse)
    # show some examples
    for i in range(count):
        error = expected[i] - predicted[i]
        print('Expected=%d, Predicted=%d (err=%d)' % (expected[i], predicted[i], error))
