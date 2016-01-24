"""
Create and train a model using keras.
"""

import random

import keras.layers.core
import keras.layers.recurrent
import keras.models
import numpy

import data_source

SEQUENCE_LENGTH = 20
STEP_SIZE = 20


def dump(model, filename):
    """
    Dump a model.
    """

    json_string = model.to_json()
    open(filename + '.json', 'w').write(json_string)
    model.save_weights(filename + '.h5', overwrite=True)

def load(filename):
    """
    Load a model from a dump.
    """

    model = keras.models.model_from_json(open(filename + '.json').read())
    model.load_weights(filename + '.h5')
    return model


def build_model():
    """
    Build and compile the model.
    """

    model = keras.models.Sequential()
    model.add(keras.layers.recurrent.LSTM(512, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 256)))
    model.add(keras.layers.core.Dropout(0.2))
    model.add(keras.layers.recurrent.LSTM(512, return_sequences=False))
    model.add(keras.layers.core.Dropout(0.2))
    model.add(keras.layers.core.Dense(256))
    model.add(keras.layers.core.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def sample(a, temperature=1.0):
    """
    Sample from a distribution.
    """

    a = numpy.log(a) / temperature
    a = numpy.exp(a) / numpy.sum(numpy.exp(a))
    return numpy.argmax(numpy.random.multinomial(1, a, 1))

def sample_model(model, sample_length, seed):
    """
    Create a sample from the model.
    """

    result = seed
    sequence = seed
    for _ in range(sample_length):
        x = numpy.zeros((1, SEQUENCE_LENGTH, 256))
        for t, char in enumerate(sequence):
            x[0, t, data_source.CHAR_INDICES[char]] = 1
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds)
        next_char = data_source.INDICIES_CHARS[next_index]
        sequence = sequence[1:] + next_char
        result += next_char
    return result


def train_model(model, input_data, expected):
    """
    Train a model for a single epoch
    """

    model.fit(input_data, expected, batch_size=128, nb_epoch=1)
