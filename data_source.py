"""
This module provides functions for loading and processing the data source.
"""

import logging

import numpy

DATA_RANGE = 256
CHAR_INDICES = {chr(i): i for i in range(0, 256)}
INDICIES_CHARS = {i:chr(i) for i in range(0, 256)}

def load_corpus(corpus):
    """
    Load a corpus.
    """

    with open(corpus) as fin:
        data = fin.read()
    return data

def _split_sequences(raw_data, length=20, skip=3):
    """
    Convert a set of raw data to a set of sequences. Returns the sequences
    and the expected result of each sequence.

    WARNING: This can have a very high memory footprint. Be wary.
    """

    all_sequences = []
    next_chars = []
    for i in range(0, len(raw_data) - length, skip):
        all_sequences.append(raw_data[i:i+length])
        next_chars.append(raw_data[i+length])
    logging.debug("Split data into %d sequences", len(all_sequences))
    return all_sequences, next_chars


def _vectorize_training_data(sequences):
    """
    Vectorize the traing data.
    """

    data = numpy.zeros((len(sequences), len(sequences[0]), DATA_RANGE),
                       dtype=numpy.bool)
    for i, sequence in enumerate(sequences):
        for t, char in enumerate(sequence):
            data[i, t, CHAR_INDICES[char]] = 1
    return data

def _vectorize_expected(expected):
    """
    Vectorize expected data.
    """

    data = numpy.zeros((len(expected), DATA_RANGE), dtype=numpy.bool)
    for i, char in enumerate(expected):
        data[i, CHAR_INDICES[expected[i]]] = 1
    return data

def get_training_data(corpus):
    """
    This will return a iterable of the generated training data (in the proper
    vectorized form).
    """

    raw_data = load_corpus(corpus)
    sequences, expected = _split_sequences(raw_data, 20, 20)
    data = _vectorize_training_data(sequences)
    expected = _vectorize_expected(expected)
    return data, expected
