import os
import pandas

headers = {
    'stats': [
        'коэффициент уникальности всех слов',

        #   URL FEATURES

        'наличие ip-адреса в url',
        'использование сервисов сокраения url',

        "наличие сертификата",
        "хороший netloc",

        'длина url',

        'кол-во @ в url',
        'кол-во ! в url',
        'кол-во + в url',
        'кол-во [ и ] в url',
        'кол-во ( и ) в url',
        'кол-во , в url',
        'кол-во $ в url',
        'кол-во ; в url',
        'кол-во пропусков в url',
        'кол-во & в url',
        'кол-во // в url',
        'кол-во / в url',
        'кол-во = в url',
        'кол-во % в url',
        'кол-во ? в url',
        'кол-во _ в url',
        'кол-во - в url',
        'кол-во . в url',
        'кол-во : в url',
        'кол-во * в url',
        'кол-во | в url',
        'кол-во ~ в url',
        'кол-во http токенов в url',

        'https',

        'соотношение цифр в url',
        'кол-во цифр в url',

        "кол-во фишинговых слов в url",
        "кол-во распознанных слов в url",

        'tld в пути url',
        'tld в поддомене url',
        'tld на плохой позиции url',
        'ненормальный поддомен url',

        'кол-во перенаправлений на сайт',
        'кол-во перенаправлений на другие домены',

        'случайный домен',

        'кол-во случайных слов в url',
        'кол-во случайных слов в хосте url',
        'кол-во случайных слов в пути url',

        'кол-во повторяющих последовательностей в url',
        'кол-во повторяющих последовательностей в хосте url',
        'кол-во повторяющих последовательностей в пути url',

        'наличие punycode',
        'домен в брендах',
        'юренд в пути url',
        'кол-во www в url',
        'кол-во com в url',

        'наличие порта в url',

        'кол-во слов в url',
        'средняя длина слова в url',
        'максимальная длина слова в url',
        'минимальная длина слова в url',

        'префикс суффикс в url',

        'кол-во поддоменов',

        'кол-во визульно схожих доменов',

        #   CONTENT FEATURE
        #       (static)

        'степень сжатия страницы',
        'кол-во полей ввода/вывода в основном контексте страницы',
        'соотношение кода в странице в основном контексте страницы',

        'кол-во всех ссылок в основном контексте страницы',
        'соотношение внутренних ссылок на сайты со всеми в основном контексте страницы',
        'соотношение внешних ссылок на сайты со всеми в основном контексте страницы',
        'соотношение пустых ссылок на сайты со всеми в основном контексте страницы',
        "соотношение внутренних CSS со всеми в основном контексте страницы",
        "соотношение внешних CSS со всеми в основном контексте страницы",
        "соотношение встроенных CSS со всеми в основном контексте страницы",
        "соотношение внутренних скриптов со всеми в основном контексте страницы",
        "соотношение внешних скриптов со всеми в основном контексте страницы",
        "соотношение встроенных скриптов со всеми в основном контексте страницы",
        "соотношение внешних изображений со всеми в основном контексте страницы",
        "соотношение внутренних изображений со всеми в основном контексте страницы",
        "общее кол-во перенаправлений по внутренним ссылкам в основном контексте страницы",
        "общее кол-во перенаправлений по внешним ссылкам в основном контексте страницы",
        "общее кол-во ошибок по внутренним ссылкам в основном контексте страницы",
        "общее кол-во ошибок по внешним ссылкам в основном контексте страницы",
        "форма входа в основном контексте страницы",
        "соотношение внешних Favicon со всеми в основном контексте страницы",
        "соотношение внутренних Favicon со всеми в основном контексте страницы",
        "наличие отправки на почту в основном контексте страницы",
        "соотношение внешних медиа со всеми в основном контексте страницы",
        "соотношение внешних медиа со всеми в основном контексте страницы",
        "пустой титульник в основном контексте страницы",
        "соотношение небезопасных якорей со всеми в основном контексте страницы",
        "соотношение безопасных якорей со всеми в основном контексте страницы",
        "соотношение внутренних ссылок на ресурсы со всеми в основном контексте страницы",
        "соотношение внешних ссылок на ресурсы со всеми в основном контексте страницы",
        "наличие невидимых в основном контексте страницы",
        "наличие onmouseover в основном контексте страницы",
        "наличие всплывающих окон в основном контексте страницы",
        "наличие событий правой мыши в основном контексте страницы",
        "наличие домена в тексте в основном контексте страницы",
        "наличие домена в титульнике в основном контексте страницы",
        "домен с авторскими правами в основном контексте страницы",
        "кол-во фишинговых слов в тексте в основном контексте страницы",
        "кол-во слов в тексте в основном контексте страницы",
        "соотношение текста со всех изображений с основным текстом в основном контексте страницы",
        "соотношение текста внутренних изображений с основным текстом в основном контексте страницы",
        "соотношение текста внешних изображений с основным текстом в основном контексте страницы",
        "соотношение текста внешних изображений с текстом внутренних изображений в основном контексте страницы",

        #       (dynamic)

        "соотношение основного текста с динамически добавляемым текстом страницы",

        #       (dynamic internals)

        "соотношение основного текста с внутреннее добавляемым текстом страницы",
        "соотношение кода в внутренне добавляемом контексте страницы",
        "кол-во полей ввода/вывода в внутренне добавляемом контексте страницы",

        'кол-во всех ссылок во внутренне добавляемом контексте страницы',
        'соотношение внутренних ссылок на сайты со всеми во внутренне добавляемом контексте страницы',
        'соотношение внешних ссылок на сайты со всеми во внутренне добавляемом контексте страницы',
        'соотношение пустых ссылок на сайты со всеми во внутренне добавляемом контексте страницы',
        "соотношение внутренних CSS со всеми во внутренне добавляемом контексте страницы",
        "соотношение внешних CSS со всеми во внутренне добавляемом контексте страницы",
        "соотношение встроенных CSS со всеми во внутренне добавляемом контексте страницы",
        "соотношение внутренних скриптов со всеми во внутренне добавляемом контексте страницы",
        "соотношение внешних скриптов со всеми во внутренне добавляемом контексте страницы",
        "соотношение встроенных скриптов со всеми во внутренне добавляемом контексте страницы",
        "соотношение внешних изображений со всеми во внутренне добавляемом контексте страницы",
        "соотношение внутренних изображений со всеми во внутренне добавляемом контексте страницы",
        "общее кол-во перенаправлений по внутренним ссылкам во внутренне добавляемом контексте страницы",
        "общее кол-во перенаправлений по внешним ссылкам во внутренне добавляемом контексте страницы",
        "общее кол-во ошибок по внутренним ссылкам во внутренне добавляемом контексте страницы",
        "общее кол-во ошибок по внешним ссылкам во внутренне добавляемом контексте страницы",
        "форма входа во внутренне добавляемом контексте страницы",
        "соотношение внешних Favicon со всеми во внутренне добавляемом контексте страницы",
        "соотношение внутренних Favicon со всеми во внутренне добавляемом контексте страницы",
        "наличие отправки на почту во внутренне добавляемом контексте страницы",
        "соотношение внешних медиа со всеми во внутренне добавляемом контексте страницы",
        "соотношение внешних медиа со всеми во внутренне добавляемом контексте страницы",
        "пустой титульник во внутренне добавляемом контексте страницы",
        "соотношение небезопасных якорей со всеми во внутренне добавляемом контексте страницы",
        "соотношение безопасных якорей со всеми во внутренне добавляемом контексте страницы",
        "соотношение внутренних ссылок на ресурсы со всеми во внутренне добавляемом контексте страницы",
        "соотношение внешних ссылок на ресурсы со всеми во внутренне добавляемом контексте страницы",
        "наличие невидимых во внутренне добавляемом контексте страницы",
        "наличие onmouseover во внутренне добавляемом контексте страницы",
        "наличие всплывающих окон во внутренне добавляемом контексте страницы",
        "наличие событий правой мыши во внутренне добавляемом контексте страницы",
        "наличие домена в тексте во внутренне добавляемом контексте страницы",
        "наличие домена в титульнике во внутренне добавляемом контексте страницы",
        "домен с авторскими правами во внутренне добавляемом контексте страницы",

        "кол-во операций ввода/вывода во внутренне добавляемом коде страницы",
        "кол-во фишинговых слов во внутренне добавляемом контексте страницы",
        "кол-во слов во внутренне добавляемом контексте страницы",

        #       (dynamic externals)

        "соотношение основного текста с внешне добавляемым текстом страницы",
        "соотношение кода в внешне добавляемом контексте страницы",
        "кол-во полей ввода/вывода в внешне добавляемом контексте страницы",

        'кол-во всех ссылок во внешне добавляемом контексте страницы',
        'соотношение внутренних ссылок на сайты со всеми во внешне добавляемом контексте страницы',
        'соотношение внешних ссылок на сайты со всеми во внешне добавляемом контексте страницы',
        'соотношение пустых ссылок на сайты со всеми во внешне добавляемом контексте страницы',
        "соотношение внутренних CSS со всеми во внешне добавляемом контексте страницы",
        "соотношение внешних CSS со всеми во внешне добавляемом контексте страницы",
        "соотношение встроенных CSS со всеми во внешне добавляемом контексте страницы",
        "соотношение внутренних скриптов со всеми во внешне добавляемом контексте страницы",
        "соотношение внешних скриптов со всеми во внешне добавляемом контексте страницы",
        "соотношение встроенных скриптов со всеми во внешне добавляемом контексте страницы",
        "соотношение внешних изображений со всеми во внешне добавляемом контексте страницы",
        "соотношение внутренних изображений со всеми во внешне добавляемом контексте страницы",
        "общее кол-во перенаправлений по внутренним ссылкам во внешне добавляемом контексте страницы",
        "общее кол-во перенаправлений по внешним ссылкам во внешне добавляемом контексте страницы",
        "общее кол-во ошибок по внутренним ссылкам во внешне добавляемом контексте страницы",
        "общее кол-во ошибок по внешним ссылкам во внешне добавляемом контексте страницы",
        "форма входа во внешне добавляемом контексте страницы",
        "соотношение внешних Favicon со всеми во внешне добавляемом контексте страницы",
        "соотношение внутренних Favicon со всеми во внешне добавляемом контексте страницы",
        "наличие отправки на почту во внешне добавляемом контексте страницы",
        "соотношение внешних медиа со всеми во внешне добавляемом контексте страницы",
        "соотношение внешних медиа со всеми во внешне добавляемом контексте страницы",
        "пустой титульник во внешне добавляемом контексте страницы",
        "соотношение небезопасных якорей со всеми во внешне добавляемом контексте страницы",
        "соотношение безопасных якорей со всеми во внешне добавляемом контексте страницы",
        "соотношение внутренних ссылок на ресурсы со всеми во внешне добавляемом контексте страницы",
        "соотношение внешних ссылок на ресурсы со всеми во внешне добавляемом контексте страницы",
        "наличие невидимых во внешне добавляемом контексте страницы",
        "наличие onmouseover во внешне добавляемом контексте страницы",
        "наличие всплывающих окон во внешне добавляемом контексте страницы",
        "наличие событий правой мыши во внешне добавляемом контексте страницы",
        "наличие домена в тексте во внешне добавляемом контексте страницы",
        "наличие домена в титульнике во внешне добавляемом контексте страницы",
        "домен с авторскими правами во внешне добавляемом контексте страницы",

        "кол-во операций ввода/вывода во внешне добавляемом коде страницы",
        "кол-во фишинговых слов во внешне добавляемом контексте страницы",
        "кол-во слов во внешне добавляемом контексте страницы",

        #   EXTERNAL FEATURES

        'срок регистрации домена',
        "домен зарегестрирован",
        "рейтинг по Alexa",
        "рейтинг по openpagerank",
        "соотношение оставшегося времени действия сертификата",
        "срок действия сертификата",
        "кол-во альтернативных имен в сертификате"
    ],
    'metadata': [
        'url',
        'lang',
        'status'
    ],
    'substats': [
        'extraction-contextData-time',
        'image-recognition-time'
    ]
}

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from matplotlib import pyplot as plt
import numpy as np
import json
import pickle
from shutil import move
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
from math import ceil

frame = pandas.read_csv('data/datasets/OUTPUT/dataset.csv')
cols = [col for col in headers['stats'] if col in list(frame)][:-1]
X = frame[cols].to_numpy()
Y = frame['status'].to_numpy()


def plot_lr(lr, n_epochs, decay):
    from math import exp

    path = [lr]

    for epoch in range(n_epochs):
        lr = lr * exp(-epoch / decay)
        path.append(lr)

    plt.plot(path)
    plt.title("изменение скорости обучения")
    plt.xlabel='эпоха'
    plt.ylabel='скорость обучения'
    plt.show()
    plt.clf()
    plt.close()


def plot_hyperop_score(dir):
    with open("data/trials/{}/results.pkl".format(dir), 'rb') as f:
        data = pickle.load(f)

    path = []
    max_v = 0.9
    for idx, raw in enumerate(data.results):
        if 'loss' in raw:
            if -raw['loss'] > max_v:
                max_v = -raw['loss']
            path.append(max_v)

    plt.plot(path)
    plt.title("подбор гиперпараметров")
    plt.xlabel = 'итерация'
    plt.ylabel = 'точность'
    plt.ylim(path[0]-0.01, path[-1]+0.01)
    plt.show()
    plt.clf()
    plt.close()


def draw(history, metrics, dir):
    h = ceil(len(metrics) / 2)
    fig, axs = plt.subplots(h, 2, figsize=(2 * 5, h * 5), dpi=400)

    axs = axs.reshape(h, 2)

    for i in range(len(metrics)):
        if metrics[i] == 'f1_score':
            s = []
            v_s = []

            for k in range(len(history['loss'])):
                p = history['precision'][k]
                r = history['recall'][k]
                s.append(2 * ((p * r) / (p + r + 0.00001)))

                p = history['val_precision'][k]
                r = history['val_recall'][k]
                v_s.append(2 * ((p * r) / (p + r + 0.00001)))

            axs[i % h, i // h].plot(s)
            axs[i % h, i // h].plot(v_s)
            axs[i % h, i // h].set(xlabel='epoch', ylabel=metrics[i])
            axs[i % h, i // h].set_title(metrics[i])
            axs[i % h, i // h].legend(['train', 'test'], loc='best')
        else:
            axs[i % h, i // h].plot(history[metrics[i]])
            axs[i % h, i // h].plot(history['val_{}'.format(metrics[i])])
            axs[i % h, i // h].set(xlabel='epoch', ylabel=metrics[i])
            axs[i % h, i // h].set_title(metrics[i])
            axs[i % h, i // h].legend(['train', 'test'], loc='best')

    fig.savefig('data/trials/{}/stats.png'.format(dir))
    fig.clf()
    plt.close()


def get_rating():
    lst = os.listdir(os.getcwd() + '/data/models')
    metrics = []
    for dir in lst:
        if 'kfold' in dir:
            continue
        try:
            with open('data/trials/{}/metrics.json'.format(dir), 'r') as f:
                metrics.append({
                    'dir': dir,
                    'm': json.load(f)
                })
        except:
            pass

    acc = [m['m']['accuracy'] for m in metrics]
    pre = [m['m']['Precision'] for m in metrics]
    rec = [m['m']['Recall'] for m in metrics]
    auc = [m['m']['AUC'] for m in metrics]
    f_score = [m['m']['f_score'] for m in metrics]
    names = [m['dir'] for m in metrics]

    df = pandas.DataFrame([acc, pre, rec, auc, f_score]).T
    df.columns = ['accuracy', 'precision', 'recall', 'auc', 'f_score']
    df.index = names
    df.to_csv('data/trials/ratings.csv', index_label='algs')


def neural_networks_kfold():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, losses
    from sklearn.model_selection import KFold

    import telepot


    metrics = [
        'accuracy',
        'AUC',
        "Precision",
        "Recall"
    ]


    tf.compat.v1.enable_eager_execution()

    try:
        with open("data/trials/neural_networks_kfold/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    def layer(N, M=-1):
        if M == -1:
            M = N

            return {
                'activation': hp.choice('activation_0', [
                    'selu',
                    'relu',
                    'softmax',
                    'softplus',
                    'softsign',
                    'elu',
                    'tanh',
                    'sigmoid',
                    'exponential'
                ]),
                'nodes_count': hp.randint('nodes_count_0', 100) * 5 + 5,
                'dropout': hp.choice('dropout_0', [
                    {'dropout_rate': (hp.randint('dropout_rate_0', 29) + 1) / 60},
                    None
                ]),
                'BatchNormalization': hp.choice('BatchNormalization_0', [False, True]),
                'next': layer(N-1, M)
            }

        if N == 0:
            return None

        return hp.choice('layer_{}'.format(M - N), [
            {
                'activation': hp.choice('activation_{}'.format(M - N), [
                    'selu',
                    'relu',
                    'softmax',
                    'softplus',
                    'softsign',
                    'elu',
                    'tanh',
                    'sigmoid',
                    'exponential'
                ]),
                'nodes_count': hp.randint('nodes_count_{}'.format(M - N), 100)*5+5,
                'dropout': hp.choice('dropout_{}'.format(M - N), [
                    {'dropout_rate': (hp.randint('dropout_rate_{}'.format(M - N), 29)+1)/60},
                    None
                ]),
                'BatchNormalization': hp.choice('BatchNormalization_{}'.format(M - N), [False, True]),
                'next': layer(N - 1, M)
            },
            None
        ])

    space = {
        'layers': layer(5),
        'learning_rate': hp.choice('lr', [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]),
        'batch_size': 64,
        'init': 'glorot_normal',
        'trainable_BatchNormalization': hp.choice('trainable_BatchNormalization', [False, True]),
        'trainable_dropouts': hp.choice('trainable_dropouts', [False, True]),
        'shuffle': True
    }

    class CustomEarlyStopping(tf.keras.callbacks.Callback):
        def __init__(self, patience=0, delay_epochs=25):
            super(CustomEarlyStopping, self).__init__()
            self.patience = patience
            self.delay_epochs = delay_epochs

        def on_train_begin(self, logs=None):
            self.wait = 0
            self.stopped_epoch = 0

        def on_epoch_end(self, epoch, logs=None):
            if epoch >= self.delay_epochs:
                v_loss = np.around(logs.get('val_loss'), 2)
                loss = np.around(logs.get('loss'), 2)

                if np.less_equal(v_loss, loss):
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.stopped_epoch = epoch
                        self.model.stop_training = True
                        print("acc > v_acc")

        def on_train_end(self, logs=None):
            if self.stopped_epoch > 0:
                print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

    def tf_callbacks():
        return [
            tf.keras.callbacks.ModelCheckpoint(
                'data/models/neural_networks_kfold/tmp.h5',
                monitor='val_accuracy',
                mode='max',
                verbose=0,
                save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                # min_delta=1e-4,
                mode='max'
            ),
            CustomEarlyStopping(
                patience=5,
                delay_epochs=20
            )
        ]

    def create_model(space):
        model = models.Sequential()

        layer = space['layers']

        while layer:
            model.add(layers.Dense(
                layer['nodes_count'],
                kernel_initializer=space['init'],
                activation=layer['activation'])
            )
            if layer['dropout']:
                model.add(layers.Dropout(layer['dropout']['dropout_rate'], trainable=space['trainable_dropouts']))
            if layer['BatchNormalization']:
                model.add(layers.BatchNormalization(trainable=space['trainable_BatchNormalization']))

            layer = layer['next']

        model.add(layers.Dense(1, kernel_initializer=space['init'], activation='sigmoid'))

        model.compile(
            optimizer=optimizers.Adam(learning_rate=space['learning_rate']),
            loss=losses.BinaryCrossentropy(from_logits=True),
            metrics=metrics
        )

        return model

    model = None

    result = []
    history = []
    scores = []

    def objective(space):
        global model, result, history, scores

        try:
            result = []
            history = []
            scores = []

            for train_index, test_index in KFold(5, shuffle=True, random_state=40).split(X):
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]

                print(len(result))

                model = create_model(space)

                h = model.fit(
                    x_train, y_train,
                    # validation_split=0.2,
                    validation_data=(x_test, y_test),
                    epochs=150,
                    batch_size=space['batch_size'],
                    callbacks=tf_callbacks(),
                    shuffle=space['shuffle'],
                    verbose=2
                )

                loss = np.round(h.history['loss'][-1], 2)
                v_loss = np.round(h.history['val_loss'][-1], 2)

                if np.less_equal(v_loss, loss):
                    history.append(h.history)
                    result.append(model.evaluate(x_test, y_test, verbose=0))

                    p = result[-1][-2]
                    r = result[-1][-1]
                    s = 2 * ((p * r) / (p + r + 0.00001))
                    result[-1].append(s)

                    l = result[-1][0]

                    if l > 10:
                        scores.append(0)
                    else:
                        el = np.exp(l)
                        v = 1 + (1 - el) / (1 + el)
                        scores.append(2 * (v * s) / (v + s))

                    if s < 0.94:
                        break
                else:
                    scores.append(-0.5)
                    break

            m = {
                "loss": np.mean([r[0] for r in result]),
                "accuracy": np.mean([r[1] for r in result]),
                "AUC": np.mean([r[2] for r in result]),
                "Precision": np.mean([r[3] for r in result]),
                "Recall": np.mean([r[4] for r in result]),
                "f_score": np.mean([r[5] for r in result])
            }

            score = np.average(scores)

            try:
                with open("data/trials/neural_networks_kfold/metric.txt") as f:
                    max_score = float(f.read().strip())  # read best metric,
            except FileNotFoundError:
                max_score = -1

            if score > max_score:
                model.save("data/models/neural_networks_kfold/nn1.h5")
                move('data/models/neural_networks_kfold/tmp.h5', "data/models/neural_networks_kfold/nn2.h5")
                with open("data/trials/neural_networks_kfold/space.json", "w") as f:
                    f.write(str(space))
                with open("data/trials/neural_networks_kfold/metric.txt", "w") as f:
                    f.write(str(score).replace('[', '').replace(']', ''))
                with open("data/trials/neural_networks_kfold/metrics.json", "w") as f:
                    json.dump('[{}] -> {}'.format(len(result), str(m)), f)
                with open("data/trials/neural_networks_kfold/history.pkl", 'wb') as f:
                    pickle.dump(history, f)

            try:
                telegram_info = pandas.read_csv('telegram_client.csv')
                bot = telepot.Bot(telegram_info['BOT_token'][0])
                bot.sendMessage(int(telegram_info['CHAT_ID'][0]), str(space))
                bot.sendMessage(int(telegram_info['CHAT_ID'][0]), str(m))
                bot.sendMessage(int(telegram_info['CHAT_ID'][0]), '[{}] -> {}'.format(len(result), str(score)))
                # draw(history, list(map(lambda x: x.lower(), metrics)) + ['loss', 'f1_score'], 'neural_networks_kfold')
                # bot.sendPhoto(int(telegram_info['CHAT_ID'][0]), photo=open(
                #     'data/trials/neural_networks_kfold/stats.png', 'rb'))
            except:
                pass

            with open("data/trials/neural_networks_kfold/results.pkl", 'wb') as output:
                pickle.dump(trials, output)

            return {
                'loss': -score,
                'status': STATUS_OK,
                'history': history,
                'space': space,
                'metrics': m
            }
        except Exception as ex:
            print(ex)

            with open("data/trials/neural_networks/results.pkl", 'wb') as output:
                pickle.dump(trials, output)

            return {
                'loss': 1,
                'status': STATUS_OK,
                'history': None,
                'space': space,
                'metrics': None
            }

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=500,
                  # + len(trials),
        trials=trials,
        timeout=60 * 60 * 7,

    )

    def typer(o):
        if isinstance(o, np.int32):
            return int(o)
        return o

    with open("data/trials/neural_networks_kfold/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/neural_networks_kfold/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


def neural_networks():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, losses

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    metrics = [
        'accuracy',
        'AUC',
        "Precision",
        "Recall"
    ]

    tf.compat.v1.enable_eager_execution()

    with open("data/trials/best_nn/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    class CustomEarlyStopping(tf.keras.callbacks.Callback):
        def __init__(self, patience=0, delay_epochs=25):
            super(CustomEarlyStopping, self).__init__()
            self.patience = patience
            self.delay_epochs = delay_epochs

        def on_train_begin(self, logs=None):
            self.wait = 0
            self.stopped_epoch = 0

        def on_epoch_end(self, epoch, logs=None):
            if epoch >= self.delay_epochs:
                v_loss = np.round(logs.get('val_loss'), 3)
                loss = np.round(logs.get('loss'), 3)

                if np.less_equal(v_loss, loss):
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.stopped_epoch = epoch
                        self.model.stop_training = True
                        print("acc > v_acc")

        def on_train_end(self, logs=None):
            if self.stopped_epoch > 0:
                print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

    def tf_callbacks():
        return [
            tf.keras.callbacks.ModelCheckpoint(
                'data/models/neural_networks/tmp.h5',
                monitor='accuracy',
                mode='max',
                verbose=0,
                save_best_only=True
            ),
            # tf.keras.callbacks.EarlyStopping(
            #     monitor='accuracy',
            #     patience=25,
            #     mode='max'
            # ),
            CustomEarlyStopping(
                patience=5,
                delay_epochs=5
            )
        ]

    model = models.Sequential()

    layer = space['layers']

    while layer:
        model.add(layers.Dense(
            layer['nodes_count'],
            kernel_initializer=space['init'],
            activation=layer['activation'])
        )
        if layer['dropout']:
            model.add(layers.Dropout(layer['dropout']['dropout_rate'], trainable=space['trainable_dropouts']))
        if layer['BatchNormalization']:
            model.add(layers.BatchNormalization(trainable=space['trainable_BatchNormalization']))

        layer = layer['next']

    model.add(layers.Dense(1, kernel_initializer=space['init'], activation='sigmoid'))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=space['learning_rate']),
        loss=losses.BinaryCrossentropy(from_logits=True),
        metrics=metrics
    )

    history = model.fit(
        x_train, y_train,
        validation_split=0.15,
        # validation_data=(x_test, y_test),
        epochs=150,
        batch_size=space['batch_size'],
        callbacks=tf_callbacks(),
        shuffle=space['shuffle'],
        verbose=2
    )

    res = model.evaluate(x_test, y_test, verbose=0)
    p = res[3]
    r = res[4]
    s = 2 * ((p * r) / (p + r + 0.001))

    m = {
        "loss": res[0],
        "accuracy": res[1],
        "AUC": res[2],
        "Precision": res[3],
        "Recall": res[4],
        "f_score": s,
    }

    model.save("data/models/neural_networks/nn1.h5")
    model.save_weights("data/models/neural_networks/nn1_w.h5")
    with open("data/trials/neural_networks/space.json", "w") as f:
        f.write(str(space))
    with open("data/trials/neural_networks/metrics.json", "w") as f:
        json.dump(str(m), f)
    with open("data/trials/neural_networks/history.pkl", 'wb') as f:
        pickle.dump(history.history, f)

    draw(history.history, list(map(lambda x: x.lower(), metrics)) + ['loss', 'f1_score'], 'neural_networks')


def find_best_NN(param='score'):
    with open("data/trials/neural_networks_kfold/results.pkl", 'rb') as f:
        data = pickle.load(f)

    metrics = []

    for idx, raw in enumerate(data.results):
        if 'history' in raw:
            history = raw['history']
            if history:
                if len(history) == 5:
                    v_l = sum([h['val_loss'][-1] for h in history]) / 5
                    p = sum([h['val_precision'][-1] for h in history]) / 5
                    r = sum([h['val_recall'][-1] for h in history]) / 5
                    s = 2 * ((p * r) / (p + r + 0.0001))

                    if param == 'f_score':
                        metrics.append([idx, s])
                        continue
                    elif param != 'score':
                        metrics.append([idx, sum([h[param][-1] for h in history]) / 5])
                        continue

                    el = np.exp(v_l)
                    v = 1 + (1 - el) / (1 + el)
                    score = 2 * (v * s) / (v + s)

                    metrics.append([idx, score, v_l, p, r, s])

    if param == 'score':
        r = pandas.DataFrame(metrics, columns=['id', 'score', 'loss', 'precision', 'recall', 'f_score']).sort_values(
            'score', ascending=False
        ).head(1)
    else:
        r = pandas.DataFrame(metrics, columns=['id', 'score']).sort_values(
            'score', ascending=False
        ).head(1)

    if not r.empty:
        id = int(r['id'])
        best = data.results[id]

        with open("data/trials/best_nn/space.json", "w") as f:
            f.write(str(best['space']))
        with open("data/trials/best_nn/metrics.json", "w") as f:
            json.dump(str(best['metrics']), f)

        from collections import defaultdict
        history = defaultdict(list)

        for h in data.results[id]['history']:
            for key, value in h.items():
                history[key] += value

        draw(
            history,
            ['accuracy', 'auc', "precision", "recall", 'loss', 'f1_score'],
            'best_NN'
        )
    else:
        print('ERROR: selection criteria are too strict!')


def DT():
    from sklearn.tree import DecisionTreeClassifier

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    try:
        with open("data/trials/DT/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'splitter': hp.choice('splitter', ['best', 'random']),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
        'min_samples_split': 2,
        'min_samples_leaf': 1,
    }

    def objective(space):
        clf = DecisionTreeClassifier(
            criterion=space['criterion'],
            splitter=space['splitter'],
            max_features=space['max_features'],
            min_samples_leaf=space['min_samples_leaf'],
            min_samples_split=space['min_samples_split']
        )

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_true=y_test, y_pred=y_pred)

        try:
            with open("data/trials/DT/metric.txt") as f:
                max_acc = float(f.read().strip())  # read best metric,
        except FileNotFoundError:
            max_acc = -1

        if acc > max_acc:
            pickle.dump(clf, open('data/models/DT/DT.pkl', 'wb'))
            with open("data/trials/DT/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/DT/metric.txt", "w") as f:
                f.write(str(acc))

            auc = roc_auc_score(y_test, y_pred)
            f_score = f1_score(y_test, y_pred)
            pre = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            m = {
                "accuracy": acc,
                "Precision": pre,
                "Recall": recall,
                "AUC": auc,
                "f_score": f_score
            }
            with open("data/trials/DT/metrics.json", "w") as f:
                json.dump(m, f)

        return {'loss': -acc, 'status': STATUS_OK, 'space': space}

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        timeout=60 * 30
    )

    def typer(o):
        if isinstance(o, np.int32): return int(o)
        return o

    with open("data/trials/DT/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/DT/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


def SVM_cv():
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_predict

    try:
        with open("data/trials/SVM/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'C': hp.randint('C', 100000)/10,
        'random_state': 42,
        'kernel': hp.choice('kernel', [
            {
                'type': 'linear',
            },
            {
                'type': 'poly',
                'degree': hp.randint('degree_poly', 360),
                'coef0': hp.uniform('coef0_poly', -1, 1)
            },
            {
                'type': 'rbf',
                'gamma': hp.choice('gamma_rbf', ['scale', 'auto'])
            },
            {
                'type': 'sigmoid',
                'gamma': hp.choice('gamma_sigmoid', ['scale', 'auto']),
                'coef0': hp.uniform('coef0_sigmoid', -1, 1)
            }
        ]),
    }

    def objective(space):
        clf = SVC(
            C=space['C'],
            random_state=space['random_state'],
            kernel=space['kernel']['type'],
            max_iter=25000,
            tol=1e-3
        )

        if 'coef0' in space['kernel']:
            clf.coef0 = space['kernel']['coef0']
        if 'degree' in space['kernel']:
            clf.degree = space['kernel']['degree']
        if 'gamma' in space['kernel']:
            clf.gamma = space['kernel']['gamma']

        try:
            y_pred = cross_val_predict(clf, X, Y, cv=5, n_jobs=2)
            acc = accuracy_score(Y, y_pred)
        except:
            acc = -1

        try:
            with open("data/trials/SVM/metric.txt") as f:
                max_acc = float(f.read().strip())  # read best metric,
        except FileNotFoundError:
            max_acc = -1

        if acc > max_acc:
            pickle.dump(clf, open('data/models/SVM/SVM.pkl', 'wb'))
            with open("data/trials/SVM/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/SVM/metric.txt", "w") as f:
                f.write(str(acc))

            auc = roc_auc_score(Y, y_pred)
            f_score = f1_score(Y, y_pred)
            pre = precision_score(Y, y_pred)
            recall = recall_score(Y, y_pred)
            m = {
                "accuracy": acc,
                "Precision": pre,
                "Recall": recall,
                "AUC": auc,
                "f_score": f_score
            }
            with open("data/trials/SVM/metrics.json", "w") as f:
                json.dump(m, f)

        with open("data/trials/SVM/results.pkl", 'wb') as output:
            pickle.dump(trials, output)

        return {'loss': -acc, 'status': STATUS_OK, 'space': space}

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=2000,
        trials=trials,
        timeout=60 * 60 * 8
    )

    def typer(o):
        if isinstance(o, np.int32): return int(o)
        return o

    with open("data/trials/SVM/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/SVM/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


def SVM():
    from sklearn.svm import SVC

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    with open("data/trials/SVM/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    clf = SVC(
        C=space['C'],
        random_state=space['random_state'],
        kernel=space['kernel']['type'],
        # max_iter=25000,
        probability=True,
        # tol=1e-3
    )

    if 'coef0' in space['kernel']:
        clf.coef0 = space['kernel']['coef0']
    if 'degree' in space['kernel']:
        clf.degree = space['kernel']['degree']
    if 'gamma' in space['kernel']:
        clf.gamma = space['kernel']['gamma']

    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    auc = roc_auc_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)

    m = {
        "accuracy": acc,
        "Precision": pre,
        "Recall": recall,
        "AUC": auc,
        "f_score": f_score
    }

    with open("data/trials/SVM/metrics.json", "w") as f:
        json.dump(m, f)

    with open("data/models/SVM/SVM.pkl", "wb") as f:
        pickle.dump(clf, f)


def KNN_cv():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_predict

    try:
        with open("data/trials/kNN/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'k': hp.randint('k', 5)*2+3,
        'p': hp.randint('p', 4) + 1,
        'weights': hp.choice('weights', ['uniform', 'distance'])
    }

    def objective(space):
        clf = KNeighborsClassifier(
            n_neighbors=space['k'],
            p=space['p'],
            weights=space['weights']
        )

        try:
            y_pred = cross_val_predict(clf, X, Y, cv=3, n_jobs=3)
            acc = accuracy_score(Y, y_pred)
        except:
            acc = -1

        try:
            with open("data/trials/kNN/metric.txt") as f:
                max_acc = float(f.read().strip())  # read best metric,
        except FileNotFoundError:
            max_acc = -1

        if acc > max_acc:
            pickle.dump(clf, open('data/models/kNN/{}NN.pkl'.format(space['k']), 'wb'))
            with open("data/trials/kNN/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/kNN/metric.txt", "w") as f:
                f.write(str(acc))

            auc = roc_auc_score(Y, y_pred)
            f_score = f1_score(Y, y_pred)
            pre = precision_score(Y, y_pred)
            recall = recall_score(Y, y_pred)
            m = {
                "accuracy": acc,
                "Precision": pre,
                "Recall": recall,
                "AUC": auc,
                "f_score": f_score
            }
            with open("data/trials/kNN/metrics.json", "w") as f:
                json.dump(m, f)

        with open("data/trials/kNN/results.pkl", 'wb') as output:
            pickle.dump(trials, output)

        return {'loss': -acc, 'status': STATUS_OK, 'space': space}

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        timeout=60 * 20 * 1
    )

    def typer(o):
        if isinstance(o, np.int32): return int(o)
        return o

    with open("data/trials/kNN/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/kNN/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


def KNN():
    from sklearn.neighbors import KNeighborsClassifier

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    with open("data/trials/kNN/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    clf = KNeighborsClassifier(
        n_neighbors=space['k'],
        p=space['p'],
        weights=space['weights']
    )

    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    auc = roc_auc_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)

    m = {
        "accuracy": acc,
        "Precision": pre,
        "Recall": recall,
        "AUC": auc,
        "f_score": f_score
    }

    with open("data/trials/kNN/metrics.json", "w") as f:
        json.dump(m, f)

    with open("data/models/kNN/kNN.pkl", "wb") as f:
        pickle.dump(clf, f)


def Gaussian_NB():
    from sklearn.naive_bayes import GaussianNB

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    clf = GaussianNB()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    auc = roc_auc_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    m = {
        "accuracy": acc,
        "Precision": pre,
        "Recall": recall,
        "AUC": auc,
        "f_score": f_score
    }
    pickle.dump(clf, open('data/models/Gaussian_NB/Gaussian_NB.pkl', 'wb'))
    with open("data/trials/Gaussian_NB/metrics.json", "w") as f:
        json.dump(m, f)


def Bernoulli_NB():
    from sklearn.naive_bayes import BernoulliNB

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    clf = BernoulliNB()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    auc = roc_auc_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    m = {
        "accuracy": acc,
        "Precision": pre,
        "Recall": recall,
        "AUC": auc,
        "f_score": f_score
    }
    pickle.dump(clf, open('data/models/Bernoulli_NB/Bernoulli_NB.pkl', 'wb'))
    with open("data/trials/Bernoulli_NB/metrics.json", "w") as f:
        json.dump(m, f)


def Complement_NB():
    from sklearn.naive_bayes import ComplementNB

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    clf = ComplementNB()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    auc = roc_auc_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    m = {
        "accuracy": acc,
        "Precision": pre,
        "Recall": recall,
        "AUC": auc,
        "f_score": f_score
    }
    pickle.dump(clf, open('data/models/Complement_NB/Complement_NB.pkl', 'wb'))
    with open("data/trials/Complement_NB/metrics.json", "w") as f:
        json.dump(m, f)


def Multinomial_NB():
    from sklearn.naive_bayes import MultinomialNB

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    clf = MultinomialNB()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    auc = roc_auc_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    m = {
        "accuracy": acc,
        "Precision": pre,
        "Recall": recall,
        "AUC": auc,
        "f_score": f_score
    }
    pickle.dump(clf, open('data/models/Multinomial_NB/Multinomial_NB.pkl', 'wb'))
    with open("data/trials/Multinomial_NB/metrics.json", "w") as f:
        json.dump(m, f)


# ansambles


def ET_cv():
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import cross_val_predict

    try:
        with open("data/trials/ET/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'n_estimators': 100,
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
        'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None]),
    }

    def objective(space):
        clf = ExtraTreesClassifier(
            n_estimators=space['n_estimators'],
            criterion=space['criterion'],
            max_features=space['max_features'],
            class_weight=space['class_weight']
        )



        try:
            y_pred = cross_val_predict(clf, X, Y, cv=5, n_jobs=3)
            acc = accuracy_score(Y, y_pred)
        except:
            acc = -1

        try:
            with open("data/trials/ET/metric.txt") as f:
                max_acc = float(f.read().strip())  # read best metric,
        except FileNotFoundError:
            max_acc = -1

        if acc > max_acc:
            pickle.dump(clf, open('data/models/ET/ET.pkl', 'wb'))
            with open("data/trials/ET/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/ET/metric.txt", "w") as f:
                f.write(str(acc))


            auc = roc_auc_score(Y, y_pred)
            f_score = f1_score(Y, y_pred)
            pre = precision_score(Y, y_pred)
            recall = recall_score(Y, y_pred)
            m = {
                "accuracy": acc,
                "Precision": pre,
                "Recall": recall,
                "AUC": auc,
                "f_score": f_score
            }
            with open("data/trials/ET/metrics.json", "w") as f:
                json.dump(m, f)

        with open("data/trials/ET/results.pkl", 'wb') as output:
            pickle.dump(trials, output)

        return {'loss': -acc, 'status': STATUS_OK, 'space': space}

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        timeout=60 * 30
    )

    def typer(o):
        if isinstance(o, np.int32): return int(o)
        return o

    with open("data/trials/ET/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/ET/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


def ET():
    from sklearn.ensemble import ExtraTreesClassifier

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    with open("data/trials/ET/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    clf = ExtraTreesClassifier(
        n_estimators=space['n_estimators'],
        criterion=space['criterion'],
        max_features=space['max_features'],
        class_weight=space['class_weight']
    )

    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    auc = roc_auc_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)

    m = {
        "accuracy": acc,
        "Precision": pre,
        "Recall": recall,
        "AUC": auc,
        "f_score": f_score
    }

    with open("data/trials/ET/metrics.json", "w") as f:
        json.dump(m, f)

    with open("data/models/ET/ET.pkl", "wb") as f:
        pickle.dump(clf, f)


def RF_cv():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict

    try:
        with open("data/trials/RF/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'n_estimators': 100,
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'max_features': hp.choice('max_features', ['sqrt', 'log2']),
        'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None]),
    }

    def objective(space):
        clf = RandomForestClassifier(
            n_estimators=space['n_estimators'],
            criterion=space['criterion'],
            max_features=space['max_features'],
            class_weight=space['class_weight']
        )

        try:
            y_pred = cross_val_predict(clf, X, Y, cv=5, n_jobs=3)
            acc = accuracy_score(Y, y_pred)
        except:
            acc = -1

        try:
            with open("data/trials/RF/metric.txt") as f:
                max_acc = float(f.read().strip())  # read best metric,
        except FileNotFoundError:
            max_acc = -1

        if acc > max_acc:
            pickle.dump(clf, open('data/models/RF/RF.pkl', 'wb'))
            with open("data/trials/RF/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/RF/metric.txt", "w") as f:
                f.write(str(acc))

            auc = roc_auc_score(Y, y_pred)
            f_score = f1_score(Y, y_pred)
            pre = precision_score(Y, y_pred)
            recall = recall_score(Y, y_pred)
            m = {
                "accuracy": acc,
                "Precision": pre,
                "Recall": recall,
                "AUC": auc,
                "f_score": f_score
            }
            with open("data/trials/RF/metrics.json", "w") as f:
                json.dump(m, f)


        with open("data/trials/RF/results.pkl", 'wb') as output:
            pickle.dump(trials, output)


        return {'loss': -acc, 'status': STATUS_OK, 'space': space}

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        timeout=60 * 30
    )

    def typer(o):
        if isinstance(o, np.int32): return int(o)
        return o

    with open("data/trials/RF/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/RF/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


def RF():
    from sklearn.ensemble import RandomForestClassifier

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    with open("data/trials/RF/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    clf = RandomForestClassifier(
        n_estimators=space['n_estimators'],
        criterion=space['criterion'],
        max_features=space['max_features'],
        class_weight=space['class_weight']
    )

    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    auc = roc_auc_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)

    m = {
        "accuracy": acc,
        "Precision": pre,
        "Recall": recall,
        "AUC": auc,
        "f_score": f_score
    }

    with open("data/trials/RF/metrics.json", "w") as f:
        json.dump(m, f)

    with open("data/models/RF/RF.pkl", "wb") as f:
        pickle.dump(clf, f)


def AdaBoost_DT_cv():
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_predict

    try:
        with open("data/trials/AdaBoost_DT/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'n_estimators': 100,
        'learning_rate': hp.randint('learning_rate', 100)/100,
        'algorithm': hp.choice('algorithm', ['SAMME', 'SAMME.R']),
        'max_depth': hp.randint('max_depth', 19)+1
    }

    def objective(space):
        clf = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=space['max_depth']),
            n_estimators=space['n_estimators'],
            learning_rate=space['learning_rate'],
            algorithm=space['algorithm']
        )



        try:
            y_pred = cross_val_predict(clf, X, Y, cv=3, n_jobs=3)
            acc = accuracy_score(Y, y_pred)
        except:
            acc = -1

        try:
            with open("data/trials/AdaBoost_DT/metric.txt") as f:
                max_acc = float(f.read().strip())  # read best metric,
        except FileNotFoundError:
            max_acc = -1

        if acc > max_acc:
            pickle.dump(clf, open('data/models/AdaBoost_DT/AdaBoost_DT.pkl', 'wb'))
            with open("data/trials/AdaBoost_DT/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/AdaBoost_DT/metric.txt", "w") as f:
                f.write(str(acc))


            auc = roc_auc_score(Y, y_pred)
            f_score = f1_score(Y, y_pred)
            pre = precision_score(Y, y_pred)
            recall = recall_score(Y, y_pred)
            m = {
                "accuracy": acc,
                "Precision": pre,
                "Recall": recall,
                "AUC": auc,
                "f_score": f_score
            }
            with open("data/trials/AdaBoost_DT/metrics.json", "w") as f:
                json.dump(m, f)


        with open("data/trials/AdaBoost_DT/results.pkl", 'wb') as output:
            pickle.dump(trials, output)


        return {'loss': -acc, 'status': STATUS_OK, 'space': space}

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=200,
        trials=trials,
        timeout=60 * 30
    )

    def typer(o):
        if isinstance(o, np.int32): return int(o)
        return o

    with open("data/trials/AdaBoost_DT/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/AdaBoost_DT/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


def AdaBoost_DT():
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    with open("data/trials/AdaBoost_DT/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=space['max_depth']),
        n_estimators=space['n_estimators'],
        learning_rate=space['learning_rate'],
        algorithm=space['algorithm']
    )

    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    auc = roc_auc_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)

    m = {
        "accuracy": acc,
        "Precision": pre,
        "Recall": recall,
        "AUC": auc,
        "f_score": f_score
    }

    with open("data/trials/AdaBoost_DT/metrics.json", "w") as f:
        json.dump(m, f)

    with open("data/models/AdaBoost_DT/AdaBoost_DT.pkl", "wb") as f:
        pickle.dump(clf, f)


def Bagging_DT_cv():
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_predict

    try:
        with open("data/trials/Bagging_DT/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'n_estimators': 100,
        'bootstrap_features': hp.choice('bootstrap_features', [False, True]),
        'max_depth': hp.randint('max_depth', 19)+1
    }

    def objective(space):
        clf = BaggingClassifier(
            DecisionTreeClassifier(max_depth=space['max_depth']),
            n_estimators=space['n_estimators'],
            bootstrap_features=space['bootstrap_features']
        )

        try:
            y_pred = cross_val_predict(clf, X, Y, cv=5, n_jobs=3)
            acc = accuracy_score(Y, y_pred)
        except:
            acc = -1

        try:
            with open("data/trials/Bagging_DT/metric.txt") as f:
                max_acc = float(f.read().strip())  # read best metric,
        except FileNotFoundError:
            max_acc = -1

        if acc > max_acc:
            pickle.dump(clf, open('data/models/Bagging_DT/Bagging_DT.pkl', 'wb'))
            with open("data/trials/Bagging_DT/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/Bagging_DT/metric.txt", "w") as f:
                f.write(str(acc))


            auc = roc_auc_score(Y, y_pred)
            f_score = f1_score(Y, y_pred)
            pre = precision_score(Y, y_pred)
            recall = recall_score(Y, y_pred)
            m = {
                "accuracy": acc,
                "Precision": pre,
                "Recall": recall,
                "AUC": auc,
                "f_score": f_score
            }
            with open("data/trials/Bagging_DT/metrics.json", "w") as f:
                json.dump(m, f)


        with open("data/trials/Bagging_DT/results.pkl", 'wb') as output:
            pickle.dump(trials, output)


        return {'loss': -acc, 'status': STATUS_OK, 'space': space}

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=200,
        trials=trials,
        timeout=60 * 30
    )

    def typer(o):
        if isinstance(o, np.int32): return int(o)
        return o

    with open("data/trials/Bagging_DT/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/Bagging_DT/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


def Bagging_DT():
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    with open("data/trials/Bagging_DT/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    clf = BaggingClassifier(
        DecisionTreeClassifier(max_depth=space['max_depth']),
        n_estimators=space['n_estimators'],
        bootstrap_features=space['bootstrap_features']
    )

    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    auc = roc_auc_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)

    m = {
        "accuracy": acc,
        "Precision": pre,
        "Recall": recall,
        "AUC": auc,
        "f_score": f_score
    }

    with open("data/trials/Bagging_DT/metrics.json", "w") as f:
        json.dump(m, f)

    with open("data/models/Bagging_DT/Bagging_DT.pkl", "wb") as f:
        pickle.dump(clf, f)


def GradientBoost_cv():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict

    try:
        with open("data/trials/GradientBoost/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'n_estimators': 100,
        'learning_rate': hp.randint('learning_rate', 100)/100,
        'loss': hp.choice('loss', ['deviance', 'exponential']),
        'criterion': hp.choice('criterion', ['friedman_mse', 'mse']),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None])
    }

    def objective(space):
        clf = GradientBoostingClassifier(
            n_estimators=space['n_estimators'],
            learning_rate=space['learning_rate'],
            criterion=space['criterion'],
            max_features=space['max_features'],
            loss=space['loss']
        )

        try:
            y_pred = cross_val_predict(clf, X, Y, cv=5, n_jobs=3)
            acc = accuracy_score(Y, y_pred)
        except:
            acc = -1

        try:
            with open("data/trials/GradientBoost/metric.txt") as f:
                max_acc = float(f.read().strip())  # read best metric,
        except FileNotFoundError:
            max_acc = -1

        if acc > max_acc:
            pickle.dump(clf, open('data/models/GradientBoost/GradientBoost.pkl', 'wb'))
            with open("data/trials/GradientBoost/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/GradientBoost/metric.txt", "w") as f:
                f.write(str(acc))

            auc = roc_auc_score(Y, y_pred)
            f_score = f1_score(Y, y_pred)
            pre = precision_score(Y, y_pred)
            recall = recall_score(Y, y_pred)
            m = {
                "accuracy": acc,
                "Precision": pre,
                "Recall": recall,
                "AUC": auc,
                "f_score": f_score
            }
            with open("data/trials/GradientBoost/metrics.json", "w") as f:
                json.dump(m, f)

        with open("data/trials/GradientBoost/results.pkl", 'wb') as output:
            pickle.dump(trials, output)

        return {'loss': -acc, 'status': STATUS_OK, 'space': space}

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=500,
        trials=trials,
        timeout=60 * 30
    )

    def typer(o):
        if isinstance(o, np.int32): return int(o)
        return o

    with open("data/trials/GradientBoost/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/GradientBoost/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


def GradientBoost():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    with open("data/trials/GradientBoost/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    clf = GradientBoostingClassifier(
        n_estimators=space['n_estimators'],
        learning_rate=space['learning_rate'],
        criterion=space['criterion'],
        max_features=space['max_features'],
        loss=space['loss']
    )

    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    auc = roc_auc_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)

    m = {
        "accuracy": acc,
        "Precision": pre,
        "Recall": recall,
        "AUC": auc,
        "f_score": f_score
    }

    with open("data/trials/GradientBoost/metrics.json", "w") as f:
        json.dump(m, f)

    with open("data/models/GradientBoost/GradientBoost.pkl", "wb") as f:
        pickle.dump(clf, f)


def HistGradientBoost_cv():
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict

    try:
        with open("data/trials/HistGradientBoost/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'learning_rate': hp.randint('learning_rate', 100) / 100,
        'loss': 'binary_crossentropy'
    }

    def objective(space):
        clf = HistGradientBoostingClassifier(
            learning_rate=space['learning_rate'],
            loss=space['loss']
        )

        try:
            y_pred = cross_val_predict(clf, X, Y, cv=5, n_jobs=3)
            acc = accuracy_score(Y, y_pred)
        except:
            acc = -1

        try:
            with open("data/trials/HistGradientBoost/metric.txt") as f:
                max_acc = float(f.read().strip())  # read best metric,
        except FileNotFoundError:
            max_acc = -1

        if acc > max_acc:
            pickle.dump(clf, open('data/models/HistGradientBoost/HistGradientBoost.pkl', 'wb'))
            with open("data/trials/HistGradientBoost/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/HistGradientBoost/metric.txt", "w") as f:
                f.write(str(acc))

            auc = roc_auc_score(Y, y_pred)
            f_score = f1_score(Y, y_pred)
            pre = precision_score(Y, y_pred)
            recall = recall_score(Y, y_pred)
            m = {
                "accuracy": acc,
                "Precision": pre,
                "Recall": recall,
                "AUC": auc,
                "f_score": f_score
            }
            with open("data/trials/HistGradientBoost/metrics.json", "w") as f:
                json.dump(m, f)

        with open("data/trials/HistGradientBoost/results.pkl", 'wb') as output:
            pickle.dump(trials, output)

        return {'loss': -acc, 'status': STATUS_OK, 'space': space}

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=500,
        trials=trials,
        timeout=60 * 30
    )

    def typer(o):
        if isinstance(o, np.int32): return int(o)
        return o

    with open("data/trials/HistGradientBoost/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/HistGradientBoost/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


def HistGradientBoost():
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    with open("data/trials/HistGradientBoost/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    clf = HistGradientBoostingClassifier(
        learning_rate=space['learning_rate'],
        loss=space['loss']
    )

    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    auc = roc_auc_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)

    m = {
        "accuracy": acc,
        "Precision": pre,
        "Recall": recall,
        "AUC": auc,
        "f_score": f_score
    }

    with open("data/trials/HistGradientBoost/metrics.json", "w") as f:
        json.dump(m, f)

    with open("data/models/HistGradientBoost/HistGradientBoost.pkl", "wb") as f:
        pickle.dump(clf, f)


# summary
# - Stacking


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = models.Model
    cls.__reduce__ = __reduce__

# Run the function
make_keras_picklable()

def create_model():
    with open("data/trials/best_nn/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    model = models.Sequential()

    layer = space['layers']

    while layer:
        model.add(layers.Dense(
            layer['nodes_count'],
            kernel_initializer=space['init'],
            activation=layer['activation'])
        )
        if layer['dropout']:
            model.add(layers.Dropout(layer['dropout']['dropout_rate'], trainable=space['trainable_dropouts']))
        if layer['BatchNormalization']:
            model.add(layers.BatchNormalization(trainable=space['trainable_BatchNormalization']))

        layer = layer['next']

    model.add(layers.Dense(1, kernel_initializer=space['init'], activation='sigmoid'))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=space['learning_rate']),
        loss=losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model

ann = KerasClassifier(build_fn=create_model, epochs=150, batch_size=64, verbose=2)
ann._estimator_type = "classifier"

def Stacking():
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import BaggingClassifier

    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.naive_bayes import ComplementNB
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import StackingClassifier



    estimators = [
        ('ANN', ann),
        ('SVM', SVC(
            C=326.7,
            random_state=42,
            kernel='rbf',
            gamma='scale',
            # tol=1e-3
        )),
        ('GNB', GaussianNB()),
        ('BNB', BernoulliNB()),
        ('CNB', ComplementNB()),
        ('MNB', MultinomialNB()),
        ('RF', RandomForestClassifier(
            class_weight='balanced_subsample',
            n_estimators=100,
            max_features='log2',
            criterion='entropy'
        )),
        ('HGBC', HistGradientBoostingClassifier(
            learning_rate=0.25
        )),
        ('GBC', GradientBoostingClassifier(
            criterion='mse',
            learning_rate=0.62,
            loss='deviance',
            max_features=None,
            n_estimators=100
        )),
        ('AdaBoost_DT', AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=10),
            n_estimators=100,
            learning_rate=0.93,
            algorithm='SAMME'
        )),
        ('kNN', KNeighborsClassifier(
            weights='distance',
            n_neighbors=7,
            p=1
        )),
        ('ET', ExtraTreesClassifier(
            n_estimators=100,
            max_features='sqrt',
            criterion='gini',
            class_weight='balanced_subsample'
        )),
        ('DT', DecisionTreeClassifier(
            criterion='entropy',
            splitter='best',
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None
        )),
        ('Bagging_DT', BaggingClassifier(
            DecisionTreeClassifier(max_depth=14),
            n_estimators=100,
            bootstrap_features=True
        ))
    ]

    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        verbose=2,
        n_jobs=1
    )

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)

    pickle.dump(clf, open('data/models/Stacking (All)/StackingClassifier.pkl', 'wb'))

    auc = roc_auc_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    m = {
        "accuracy": acc,
        "Precision": pre,
        "Recall": recall,
        "AUC": auc,
        "f_score": f_score
    }
    with open("data/trials/Stacking (All)/metrics.json", "w") as f:
        json.dump(m, f)
