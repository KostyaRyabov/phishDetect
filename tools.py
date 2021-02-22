from main import state

if state in range(1, 7):
    import time
    import _thread
    import threading
    from collections import Counter
    from contextlib import contextmanager

    from nltk.tokenize import RegexpTokenizer


    ########################################################################################################################
    #                                  Decorators
    ########################################################################################################################


    def parametrized(dec):
        def layer(*args, **kwargs):
            def repl(f):
                return dec(f, *args, **kwargs)
            return repl
        return layer

    class Task:
        def __init__(self, func, *args, **kwargs):
            self.func = func
            self.args = args
            self.kwargs = kwargs

        def __call__(self):
            return self.func(*self.args, **self.kwargs)

    class TimeoutException(Exception):
        def __init__(self, msg=''):
            self.msg = msg

    @contextmanager
    def time_limit(seconds, msg=''):
        timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
        timer.start()
        try:
            yield
        except Exception:
            raise TimeoutException("Timed out for operation {}".format(msg))
        finally:
            timer.cancel()


    @parametrized
    def benchmark(func, timeout=None):
        # def wrapper(*args, **kwargs):
        #     start = time.time()
        #     with concurrent.futures.ThreadPoolExecutor() as executor:
        #         try:
        #             task = executor.submit(func, *args, **kwargs)
        #             for future in concurrent.futures.as_completed([task], timeout=timeout):
        #                 return_value = future.result()
        #         except Exception as e:
        #             return_value = -5
        #
        #     end = time.time()
        #     return return_value, end - start

        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return_value = func(*args, **kwargs)
            except:
                return_value = -5
            end = time.time()
            return return_value, end - start

        return wrapper


    # def normalize_exp3(func):
    #     def wrapper(*args, **kwargs):
    #         return_value = func(*args, **kwargs)
    #         return 1 - 1 / (math.exp(return_value))
    #
    #     return wrapper


if state in range(2, 7):
    import wordsegment
    from nltk.corpus import stopwords, brown

    WORDS = list(Counter(brown.words()).keys())

    ########################################################################################################################
    #                                          Preparation of text
    ########################################################################################################################


    STOPWORDS = stopwords.words()

    def tokenize(text):
        return RegexpTokenizer(r'[^\W\d_]+').tokenize(text)      # without numbers

    def clear_text(word_raw):
        return [word for word in word_raw if word not in STOPWORDS and len(word) > 2]

    ########################################################################################################################
    #                                          Text segmentation
    ########################################################################################################################

    wordsegment.load()

    def segment(obj):
        if type(obj) is list:
            return [word for str in [wordsegment.segment(word) for word in obj] for word in str]
        elif type(obj) is str:
            return wordsegment.segment(obj)
        else:
            return None


if state in range(1, 8):
    ########################################################################################################################
    #                                          TF-IDF
    ########################################################################################################################

    from collections import Counter
    import math


    def compute_tf(word_raw):
        tf = {}
        N = len(word_raw)
        for word, count in Counter(word_raw).items():
            tf[word] = count / N
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
        tf_idf = dict.fromkeys(idf.keys(), 0)
        for word, v in tf.items():
            tf_idf[word] = v * idf[word]
        return tf_idf