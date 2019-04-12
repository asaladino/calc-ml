from random import seed
from random import randint
from numpy import array
from math import ceil
from math import log10
from numpy import argmax

seed(1)


def random_sum_pairs(n_examples, largest):
    X, y = list(), list()
    for i in range(n_examples):
        in_pattern = [randint(1, largest), randint(1, largest), randint(0, 1)]
        out_pattern = in_pattern[0] + in_pattern[1] if (in_pattern[2] is 0) else in_pattern[0] - in_pattern[1]
        X.append(in_pattern)
        y.append(out_pattern)
    return X, y


# convert data to strings
def to_string(X, y, largest):
    n_numbers = 2
    max_length = n_numbers * ceil(log10(largest + 1)) + n_numbers - 1
    Xstr = list()
    for pattern in X:
        op = '+' if (pattern[2] is 0) else '-'
        strp = str(pattern[0]) + op + str(pattern[1])
        strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp
        Xstr.append(strp)
    max_length = ceil(log10(n_numbers * (largest + 1)))
    ystr = list()
    for pattern in y:
        strp = str(pattern)
        strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp
        ystr.append(strp)
    return Xstr, ystr


# integer encode strings
def integer_encode(X, y, alphabet):
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    Xenc = list()
    for pattern in X:
        integer_encoded = [char_to_int[char] for char in pattern]
        Xenc.append(integer_encoded)
    yenc = list()
    for pattern in y:
        integer_encoded = [char_to_int[char] for char in pattern]
        yenc.append(integer_encoded)
    return Xenc, yenc


# one hot encode
def one_hot_encode(X, y, max_int):
    Xenc = list()
    for seq in X:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        Xenc.append(pattern)
    yenc = list()
    for seq in y:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        yenc.append(pattern)
    return Xenc, yenc


def convert(x, x2, op, alphabet, largest):
    X = [[x, x2, 0 if op is '+' else 1]]
    y = [x + x2 if op is '+' else x - x2]
    # convert to strings
    X, y = to_string(X, y, largest)
    # integer encode
    X, y = integer_encode(X, y, alphabet)
    # one hot encode
    X, y = one_hot_encode(X, y, len(alphabet))
    # return as numpy arrays
    X, y = array(X), array(y)
    return X, y


# generate an encoded dataset
def generate_data(n_samples, largest, alphabet):
    # generate pairs
    X, y = random_sum_pairs(n_samples, largest)
    # convert to strings
    X, y = to_string(X, y, largest)
    # integer encode
    X, y = integer_encode(X, y, alphabet)
    # one hot encode
    X, y = one_hot_encode(X, y, len(alphabet))
    # return as numpy arrays
    X, y = array(X), array(y)
    return X, y


# invert encoding
def invert(seq, alphabet):
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    strings = list()
    for pattern in seq:
        string = int_to_char[argmax(pattern)]
        strings.append(string)
    return ''.join(strings)
