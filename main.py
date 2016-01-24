#!/usr/bin/env python
"""
A minimal example to train a RNN on characters using theano and lasagne.
"""
import argparse
import logging
import random

import data_source
import network


def parse_args():
    """
    Parse all arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='Load from a file')
    parser.set_defaults(command='none')
    subparsers = parser.add_subparsers()
    sample_parser = subparsers.add_parser('sample')
    sample_parser.add_argument('--length', type=int, default=100)
    sample_parser.set_defaults(command='sample')
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('corpus')
    train_parser.add_argument('--dump', help='Dump to a file')
    train_parser.set_defaults(command='train')

    args = parser.parse_args()

    if args.command == 'none':
        parser.print_help()
    return args


def get_model(load):
    """
    Get the model.
    """

    if load:
        logging.info("Loading model from %s", load)
        return network.load(load)
    else:
        logging.info("Building a new model")
        return network.build_model()


def sample_model(model, data, length):
    """
    Sample the model with a random seed.
    """

    seed_start = random.randint(0, len(data) - 20 - 1)
    seed = data[seed_start:seed_start+20]
    return network.sample_model(model, length, seed)


def train_model(model, corpus, dump=None, epochs=10):
    """
    Train the model with the given corpus.
    """

    logging.info("Loading and processing corpus %s", corpus)
    raw_data = data_source.load_corpus(corpus)
    training, expected = data_source.get_training_data(corpus)
    for epoch in range(epochs):
        network.train_model(model, training, expected)
        if dump:
            network.dump(model, dump)
        logging.info("Completed epoch %d", epoch)
        # Print out a sample output
        print(sample_model(model, raw_data, 100))


def main():
    args = parse_args()

    if args.command == 'none':
        return

    # If we're doing anything else load up the model.
    model = get_model(args.load)
    if args.command == 'sample':
        pass
    elif args.command == 'train':
        train_model(model, args.corpus, args.dump)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    main()
