"""
Create and train a model using keras.
"""

import random

import keras.layers.core
import keras.layers.recurrent
import keras.models
import numpy

SEQUENCE_LENGTH = 20
STEP_SIZE = 20

CHAR_INDICES = {}
INDICIES_CHARS = {}

def dump(model, filename):
    json_string = model.to_json()
    open(filename + '.json', 'w').write(json_string)
    model.save_weights(filename + '.h5')

def load(filename):
    model = model_from_json(open(filename + '.json').read())
    model.load_weights(filename + '.h5')

def preprocess_data(data):
    """
    Do data preprocessing. Build data sequences and vectorize.
    """

    global INDICIES_CHARS, CHAR_INDICES
    print("Processing input data")
    chars = set(data)
    CHAR_INDICES = dict((c, i) for i, c in enumerate(chars))
    INDICIES_CHARS = dict((i, c) for i, c in enumerate(chars))
    all_sequences = []
    next_chars = []
    for i in range(0, len(data) - SEQUENCE_LENGTH, STEP_SIZE):
        all_sequences.append(data[i:i+SEQUENCE_LENGTH])
        next_chars.append(data[i+SEQUENCE_LENGTH])
    print("Split into {} sequences".format(len(all_sequences)))

    input_data = numpy.zeros((len(all_sequences), SEQUENCE_LENGTH, len(chars)), dtype=numpy.bool)
    expected = numpy.zeros((len(all_sequences), len(chars)), dtype=numpy.bool)
    for i, sequence in enumerate(all_sequences):
        for t, char in enumerate(sequence):
            input_data[i, t, CHAR_INDICES[char]] = 1
        expected[i, CHAR_INDICES[next_chars[i]]] = 1
    return input_data, expected, len(chars)

def build_model(inputs):
    """
    Build and compile the model.
    """

    print("Building the model.")
    model = keras.models.Sequential()
    model.add(keras.layers.recurrent.LSTM(512, return_sequences=True, input_shape=(SEQUENCE_LENGTH, inputs)))
    model.add(keras.layers.core.Dropout(0.2))
    model.add(keras.layers.recurrent.LSTM(512, return_sequences=False))
    model.add(keras.layers.core.Dropout(0.2))
    model.add(keras.layers.core.Dense(inputs))
    model.add(keras.layers.core.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = numpy.log(a) / temperature
    a = numpy.exp(a) / numpy.sum(numpy.exp(a))
    return numpy.argmax(numpy.random.multinomial(1, a, 1))

def sample_model(model, sample_length, seed, inputs):
    """
    Create a sample from the model.
    """

    result = seed
    sequence = seed
    for _ in range(sample_length):
        x = numpy.zeros((1, SEQUENCE_LENGTH, inputs))
        for t, char in enumerate(sequence):
            x[0, t, CHAR_INDICES[char]] = 1
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds)
        next_char = INDICIES_CHARS[next_index]
        sequence = sequence[1:] + next_char
        result += next_char
    return result


def train_model(model, input_data, expected, inputs, print_sample=False, print_seed=None):
    """
    Train a model. Optionally print generated text every epoch.
    """

    for epoch in range(50):
        #model.fit(input_data, expected, batch_size=128, nb_epoch=1)
        dump(model, 'fit_data')
        print("Finished training epoch {}".format(epoch))
        if print_sample:
            print(sample_model(model, 100, print_seed, inputs))

def create_and_train_model(data, load_from=None):
    """
    Create a model and train it on data.
    """

    input_data, expected, inputs = preprocess_data(data)
    if load_from:
        model = load('fit_data')
    else:
        model = build_model(inputs)
    seed_start = random.randint(0, len(data) - SEQUENCE_LENGTH - 1)
    train_model(model, input_data, expected, inputs, True, data[seed_start:seed_start+SEQUENCE_LENGTH])
