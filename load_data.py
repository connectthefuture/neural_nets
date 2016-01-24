"""
Utility functions to load data.
"""

def load_corpus(corpus):
    """
    Load and normalize a corpus.
    """

    with open(corpus) as fin:
        data = fin.read()
    return data
