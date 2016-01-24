#!/usr/bin/env python
"""
A minimal example to train a RNN on characters using theano and lasagne.
"""
import argparse

import load_data
import network


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus')
    args = parser.parse_args()
    
    data = load_data.load_corpus(args.corpus)
    model = network.create_and_train_model(data)

if __name__ == "__main__":
    main()
