import time
from nltk.corpus import stopwords, brown
from nltk.tokenize import RegexpTokenizer
import math
from collections import Counter


WORDS = list(Counter(brown.words()).keys())


########################################################################################################################
#                                  Decorators
########################################################################################################################


def benchmark(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        return_value = func(*args, **kwargs)
        end = time.time()
        return return_value, end - start

    return wrapper


def normalize_exp3(func):
    def wrapper(*args, **kwargs):
        return_value = func(*args, **kwargs)
        return 1 - 1 / (math.exp(return_value))

    return wrapper


########################################################################################################################
#                                          Preparation of text
########################################################################################################################


STOPWORDS = stopwords.words()


def tokenize(text):
    return RegexpTokenizer(r'[^\W\d]+').tokenize(text.lower())      # without numbers


def clear_text(word_raw):
    return [word for word in word_raw if word not in STOPWORDS]


########################################################################################################################
#                                          Text segmentation
########################################################################################################################


import wordsegment
wordsegment.load()


def segment(obj):
    if type(obj) is list:
        return [word for str in [wordsegment.segment(word) for word in obj] for word in str]
    elif type(obj) is str:
        return wordsegment.segment(obj)
    else:
        return None