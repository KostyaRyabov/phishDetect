import time
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, brown
from collections import Counter

WORDS = list(Counter(brown.words()).keys())


########################################################################################################################
#                                  Decorators
########################################################################################################################


def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer


import concurrent.futures

class Task:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.func(*self.args, **self.kwargs)


from contextlib import contextmanager
import threading
import _thread

class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg

@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        timer.cancel()


@parametrized
def benchmark(func, timeout):
    def wrapper(*args, **kwargs):
        start = time.time()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            task = executor.submit(func, *args, **kwargs)
            try:
                for future in concurrent.futures.as_completed([task], timeout=timeout):
                    return_value = future.result()
            except Exception as e:
                return_value = -5
                if func.__name__ == 'url_iterator':
                    print('Time out! : [{}]'.format(func.__name__))

        end = time.time()
        return return_value, end - start

    return wrapper

#
# @parametrized
# def benchmark(func, timeout):
#     def wrapper(*args, **kwargs):
#         start = time.time()
#
#         try:
#             with time_limit(timeout, 'sleep'):
#                 return_value = func(*args, **kwargs)
#         except Exception:
#             return_value = -5
#             print('Time out! : ({})'.format(func.__name__))
#
#         end = time.time()
#         return return_value, end - start
#
#     return wrapper



# def benchmark(func):
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         return_value = func(*args, **kwargs)
#         end = time.time()
#         return return_value, end - start
#
#     return wrapper


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
