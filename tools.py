import time
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, brown
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


# def normalize_exp3(func):
#     def wrapper(*args, **kwargs):
#         return_value = func(*args, **kwargs)
#         return 1 - 1 / (math.exp(return_value))
#
#     return wrapper


########################################################################################################################
#                                          Preparation of text
########################################################################################################################


STOPWORDS = stopwords.words()


def tokenize(text):
    return RegexpTokenizer(r'[^\W\d]+').tokenize(text)      # without numbers


def clear_text(word_raw):
    return [word for word in word_raw if word not in STOPWORDS]


from nltk.stem.wordnet import WordNetLemmatizer


########################################################################################################################
#                                          TF-IDF
########################################################################################################################


from collections import Counter


def compute_tf(word_raw):
    tf = {}
    N = len(word_raw)
    for word, count in Counter(word_raw).items():
        tf[word] = count/N
    return tf


def compute_idf(TF_list):
    n = len(TF_list)
    idf = Counter()
    for TF in TF_list:
        for word, count in TF.items():
            idf[word] += 1

    for word, v in idf.items():
        idf[word] = math.log(n / float(v))
    return dict(idf)


def compute_tf_idf(tf, idf):
    tf_idf = dict.fromkeys(tf.keys(), 0)
    for word, v in tf.items():
        tf_idf[word] = v * idf[word]
    return tf_idf


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
