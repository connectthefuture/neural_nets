#!/usr/bin/env python
"""
A minimal example to train a RNN on characters using theano and lasagne.
"""
import argparse
import random

import load_data
import network


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus')
    parser.add_argument('--sample', type=int)
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()

    data = load_data.load_corpus(args.corpus)
    if args.sample:
        model = network.load('fit_data')
        network.preprocess_data(data)
        seed_start = random.randint(0, len(data) - 20 - 1)
        seed = data[seed_start:seed_start+20]
        print(network.sample_model(model, int(args.sample), seed, len(network.CHAR_INDICES) + 1))
    else: # Train the model
        model = network.create_and_train_model(data, args.load)

if __name__ == "__main__":
    main()
