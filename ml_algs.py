import concurrent
from concurrent.futures import wait, ALL_COMPLETED
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
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, matthews_corrcoef
from math import ceil

frame = pandas.read_csv('data/datasets/OUTPUT2/dataset.csv')
cols = list(frame)[:-1]
X = frame[cols].to_numpy()
Y = frame['status'].to_numpy()


def plot_hyperop_score(dir, max=True, metric=None):
    with open("data/trials/{}/results.pkl".format(dir), 'rb') as f:
        data = pickle.load(f)
        data = data.results

    path = []
    max_v = 0

    if metric == None:
        if max:
            for idx, raw in enumerate(data):
                if 'loss' in raw:
                    if -raw['loss'] > max_v:
                        max_v = -raw['loss']
                    path.append(max_v)
        else:
            for idx, raw in enumerate(data):
                if metric in raw:
                    path.append(-raw[metric])
    else:
        if max:
            for idx, raw in enumerate(data):
                if 'metrics' in raw:
                    if raw['metrics'][metric] > max_v:
                        max_v = raw['metrics'][metric]
                    path.append(max_v)
        else:
            for idx, raw in enumerate(data):
                if 'metrics' in raw:
                    path.append(raw['metrics'][metric])


    plt.plot(path)
    plt.title("подбор гиперпараметров")
    plt.xlabel = 'итерация'
    plt.ylabel = 'точность'
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
            with open('data/trials/{}/stats.json'.format(dir), 'r') as f:
                metrics.append({
                    'dir': dir,
                    'm': json.load(f)
                })
        except:
            pass

    acc = [m['m']['accuracy'] for m in metrics]
    pre = [m['m']['precision'] for m in metrics]
    rec = [m['m']['recall'] for m in metrics]
    auc = [m['m']['auc'] for m in metrics]
    f_score = [m['m']['f_score'] for m in metrics]
    mcc = [m['m']['mcc'] for m in metrics]
    names = [m['dir'] for m in metrics]

    df = pandas.DataFrame([acc, pre, rec, auc, f_score, mcc]).T
    df.columns = ['accuracy', 'precision', 'recall', 'auc', 'f_score', 'mcc']
    df.index = names
    df.to_csv('data/trials/ratings.csv', index_label='algs')


def neural_networks_kfold():
    global X
    X = X * 0.9998 + 0.0001

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
                    # 'tanh',
                    # 'sigmoid',
                    # 'exponential'
                ]),
                'nodes_count': hp.randint('nodes_count_0', 30) * 5+5,
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
                    # 'tanh',
                    # 'sigmoid',
                    # 'exponential'
                ]),
                'nodes_count': hp.randint('nodes_count_{}'.format(M - N), 30)*5+5,
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
        'learning_rate': hp.choice(
            'lr', [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6]
        ),
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
                v_loss = np.around(logs.get('val_loss'), 3)
                loss = np.around(logs.get('loss'), 3)

                if np.less_equal(v_loss, loss):
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.stopped_epoch = epoch
                        self.model.stop_training = True

        def on_train_end(self, logs=None):
            if self.stopped_epoch > 0:
                print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


    def tf_callbacks():
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                min_delta=1e-4,
                mode='max'
            ),
            CustomEarlyStopping(
                patience=3,
                delay_epochs=5
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


    def objective(space):
        history = []

        stats = {'loss': [], 'acc': [], 'auc': [], 'pre': [], 'recall': [], 'fscore': [], 'mcc': []}


        def task(obj):
            train_index, val_index = obj

            xTrain, xVal = X[train_index], X[val_index]
            yTrain, yVal = Y[train_index], Y[val_index]

            try:
                model = create_model(space)

                h = model.fit(
                    xTrain, yTrain,
                    validation_data=(xVal, yVal),
                    epochs=50,
                    batch_size=space['batch_size'],
                    callbacks=tf_callbacks(),
                    shuffle=space['shuffle'],
                    verbose=0
                )

                history.append(h.history)
                result = model.evaluate(xVal, yVal, verbose=0)

                p = result[-2]
                r = result[-1]
                s = 2 * ((p * r) / (p + r + 0.00001))

                y_pred = (model.predict(xVal) > 0.5).astype("int32")

                v_loss = np.around(h.history['val_loss'][-1], 3)
                loss = np.around(h.history['loss'][-1], 3)

                mcc = matthews_corrcoef(yVal, y_pred)

                if not np.less_equal(v_loss, loss):
                    print('[OVERFITTING]')
                    result = [-r for r in result]
                    mcc = -mcc
                    s = -s

                stats["loss"].append(result[0])
                stats['acc'].append(result[1])
                stats['auc'].append(result[2])
                stats['pre'].append(result[3])
                stats['recall'].append(result[4])
                stats['fscore'].append(s)
                stats['mcc'].append(mcc)
            except Exception as ex:
                print(ex)

        with concurrent.futures.ThreadPoolExecutor(5) as executor:
            executor.map(task, KFold(5, shuffle=True, random_state=5).split(X, Y))

        m = {
            "loss": np.sum(stats['loss']) / 5,
            "accuracy": np.sum(stats['acc']) / 5,
            "Precision": np.sum(stats['pre']) / 5,
            "Recall": np.sum(stats['recall']) / 5,
            "AUC": np.sum(stats['auc']) / 5,
            "f_score": np.sum(stats['fscore']) / 5,
            "mcc": np.sum(stats['mcc']) / 5
        }

        try:
            with open("data/trials/neural_networks_kfold/metric.txt") as f:
                max_mcc = float(f.read().strip())
        except FileNotFoundError:
            max_mcc = -1

        if m['mcc'] >= max_mcc:
            with open("data/trials/neural_networks_kfold/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/neural_networks_kfold/metric.txt", "w") as f:
                f.write(str(m['mcc']))
            with open("data/trials/neural_networks_kfold/metrics.json", "w") as f:
                json.dump(m, f)

        try:
            telegram_info = pandas.read_csv('telegram_client.csv')
            bot = telepot.Bot(telegram_info['BOT_token'][0])
            bot.sendMessage(int(telegram_info['CHAT_ID'][0]), str(space))
            bot.sendMessage(int(telegram_info['CHAT_ID'][0]), str(m))
            bot.sendMessage(int(telegram_info['CHAT_ID'][0]), '{} -> {}'.format([len(h['loss']) for h in history], str(m['mcc'])))
            # draw(history, list(map(lambda x: x.lower(), metrics)) + ['loss', 'f1_score'], 'neural_networks_kfold')
            # bot.sendPhoto(int(telegram_info['CHAT_ID'][0]), photo=open(
            #     'data/trials/neural_networks_kfold/stats.png', 'rb'))
        except:
            pass

        with open("data/trials/neural_networks_kfold/results.pkl", 'wb') as output:
            pickle.dump(trials, output)

        return {'loss': -m['mcc'], 'status': STATUS_OK, 'space': space, "history": history, "metrics": m}

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=500,
        trials=trials,
        timeout=60 * 60 * 6
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
    global X
    X = X * 0.9998 + 0.0001

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, losses

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

    metrics = [
        'accuracy',
        'AUC',
        "Precision",
        "Recall"
    ]

    tf.compat.v1.enable_eager_execution()

    with open("data/trials/neural_networks_kfold/space.json", 'r') as f:
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

        def on_train_end(self, logs=None):
            if self.stopped_epoch > 0:
                print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

    def tf_callbacks():
        return [
            tf.keras.callbacks.ModelCheckpoint(
                'data/models/neural_networks/ANN.h5',
                monitor='accuracy',
                mode='max',
                verbose=0,
                save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min',
                verbose=2
            ),
            CustomEarlyStopping(
                patience=3,
                delay_epochs=3
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
        validation_data=(x_test, y_test),
        epochs=1000,
        batch_size=space['batch_size'],
        callbacks=tf_callbacks(),
        shuffle=space['shuffle'],
        verbose=2
    )

    res = model.evaluate(x_test, y_test)

    y_pred = (model.predict(x_test) > 0.5).astype("int32")
    f_score = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    stats = {
        "accuracy": res[1],
        "auc": res[2],
        "precision": res[3],
        "recall": res[4],
        "f_score": f_score,
        "mcc": mcc
    }

    with open("data/trials/neural_networks/stats.json", "w") as f:
        json.dump(stats, f)
    with open("data/trials/neural_networks/history.pkl", 'wb') as f:
        pickle.dump(history.history, f)

    draw(history.history, list(map(lambda x: x.lower(), metrics)) + ['loss', 'f1_score'], 'neural_networks')


def XGB_cv():
    global X
    X = X * 0.9998 + 0.0001

    import xgboost as xgb
    from sklearn.model_selection import KFold

    try:
        with open("data/trials/XGB/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'max_depth': hp.choice('max_depth', range(5, 30, 1)),
        'learning_rate': hp.quniform('learning_rate', 0.01, 0.5, 0.01),
        'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
        'gamma': hp.quniform('gamma', 0, 0.50, 0.01),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'subsample': hp.quniform('subsample', 0.1, 1, 0.01),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)}

    def objective(space):
        stats = {'acc': [], 'auc': [], 'pre': [], 'recall': [], 'fscore': [], 'mcc': []}

        def task(obj):
            train_index, test_index = obj
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            clf = xgb.XGBClassifier(
                use_label_encoder=False,
                n_estimators=space['n_estimators'],
                max_depth=int(space['max_depth']),
                learning_rate=space['learning_rate'],
                gamma=space['gamma'],
                min_child_weight=space['min_child_weight'],
                subsample=space['subsample'],
                colsample_bytree=space['colsample_bytree']
            )

            clf.fit(x_train, y_train,
                    eval_set=[(x_train, y_train), (x_test, y_test)],
                    eval_metric='logloss',
                    early_stopping_rounds=3,
                    verbose=False)

            y_pred = clf.predict(x_test)

            stats['acc'].append(accuracy_score(y_test, y_pred))
            stats['auc'].append(roc_auc_score(y_test, y_pred))
            stats['pre'].append(precision_score(y_test, y_pred))
            stats['recall'].append(recall_score(y_test, y_pred))
            stats['fscore'].append(f1_score(y_test, y_pred))
            stats['mcc'].append(matthews_corrcoef(y_test, y_pred))

        with concurrent.futures.ThreadPoolExecutor(5) as executor:
            executor.map(task, KFold(5, shuffle=True, random_state=5).split(X, Y))

        m = {
            "accuracy": np.average(stats['acc']),
            "Precision": np.average(stats['pre']),
            "Recall": np.average(stats['recall']),
            "AUC": np.average(stats['auc']),
            "f_score": np.average(stats['fscore']),
            "mcc": np.average(stats['mcc'])
        }

        try:
            with open("data/trials/XGB/metric.txt") as f:
                max_mcc = float(f.read().strip())
        except FileNotFoundError:
            max_mcc = -1

        if m['mcc'] >= max_mcc:
            with open("data/trials/XGB/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/XGB/metric.txt", "w") as f:
                f.write(str(m['mcc']))
            with open("data/trials/XGB/metrics.json", "w") as f:
                json.dump(m, f)

        return {'loss': -m['mcc'], 'status': STATUS_OK, 'space': space}

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

    with open("data/trials/XGB/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/XGB/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


def DT_cv():
    global X
    X = X * 0.9998 + 0.0001

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import KFold

    try:
        with open("data/trials/DT/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'splitter': hp.choice('splitter', ['best', 'random']),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
        'max_depth': hp.randint('max_depth', 1, 100),
        'min_samples_split': hp.uniform('min_samples_split', 0, 0.5),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5)
    }

    def objective(space):
        stats = {'acc': [], 'auc': [], 'pre': [], 'recall': [], 'fscore': [], 'mcc': []}

        def task(obj):
            train_index, test_index = obj
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            clf = DecisionTreeClassifier(
                criterion=space['criterion'],
                splitter=space['splitter'],
                max_features=space['max_features'],
                min_samples_leaf=space['min_samples_leaf'],
                min_samples_split=space['min_samples_split']
            )

            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)

            stats['acc'].append(accuracy_score(y_test, y_pred))
            stats['auc'].append(roc_auc_score(y_test, y_pred))
            stats['pre'].append(precision_score(y_test, y_pred))
            stats['recall'].append(recall_score(y_test, y_pred))
            stats['fscore'].append(f1_score(y_test, y_pred))
            stats['mcc'].append(matthews_corrcoef(y_test, y_pred))

        with concurrent.futures.ThreadPoolExecutor(5) as executor:
            executor.map(task, KFold(5, shuffle=True, random_state=5).split(X, Y))

        m = {
            "accuracy": np.average(stats['acc']),
            "Precision": np.average(stats['pre']),
            "Recall": np.average(stats['recall']),
            "AUC": np.average(stats['auc']),
            "f_score": np.average(stats['fscore']),
            "mcc": np.average(stats['mcc'])
        }

        try:
            with open("data/trials/DT/metric.txt") as f:
                max_mcc = float(f.read().strip())
        except FileNotFoundError:
            max_mcc = -1

        if m['mcc'] >= max_mcc:
            with open("data/trials/DT/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/DT/metric.txt", "w") as f:
                f.write(str(m['mcc']))
            with open("data/trials/DT/metrics.json", "w") as f:
                json.dump(m, f)

        return {'loss': -m['mcc'], 'status': STATUS_OK, 'space': space}

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

    with open("data/trials/DT/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/DT/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


def SVM_cv():
    global X
    X = X * 0.9998 + 0.0001

    from sklearn.svm import SVC
    from sklearn.model_selection import KFold

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
        stats = {'acc': [], 'auc': [], 'pre': [], 'recall': [], 'fscore': [], 'mcc': []}

        def task(obj):
            train_index, test_index = obj
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            clf = SVC(
                C=space['C'],
                random_state=space['random_state'],
                kernel=space['kernel']['type'],
                max_iter=1000000,
                # tol=1e-3,
                verbose=True
            )

            if 'coef0' in space['kernel']:
                clf.coef0 = space['kernel']['coef0']
            if 'degree' in space['kernel']:
                clf.degree = space['kernel']['degree']
            if 'gamma' in space['kernel']:
                clf.gamma = space['kernel']['gamma']

            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)

            stats['acc'].append(accuracy_score(y_test, y_pred))
            stats['auc'].append(roc_auc_score(y_test, y_pred))
            stats['pre'].append(precision_score(y_test, y_pred))
            stats['recall'].append(recall_score(y_test, y_pred))
            stats['fscore'].append(f1_score(y_test, y_pred))
            stats['mcc'].append(matthews_corrcoef(y_test, y_pred))

        with concurrent.futures.ThreadPoolExecutor(3) as executor:
            executor.map(task, KFold(5, shuffle=True, random_state=5).split(X, Y))

        m = {
            "accuracy": np.average(stats['acc']),
            "Precision": np.average(stats['pre']),
            "Recall": np.average(stats['recall']),
            "AUC": np.average(stats['auc']),
            "f_score": np.average(stats['fscore']),
            "mcc": np.average(stats['mcc'])
        }

        try:
            with open("data/trials/SVM/metric.txt") as f:
                max_mcc = float(f.read().strip())
        except FileNotFoundError:
            max_mcc = -1

        if m['mcc'] >= max_mcc:
            with open("data/trials/SVM/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/SVM/metric.txt", "w") as f:
                f.write(str(m['mcc']))
            with open("data/trials/SVM/metrics.json", "w") as f:
                json.dump(m, f)

        with open("data/trials/SVM/results.pkl", 'wb') as output:
            pickle.dump(trials, output)

        return {'loss': -m['mcc'], 'status': STATUS_OK, 'space': space}

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=500,
        trials=trials,
        timeout=60 * 60 * 10
    )

    def typer(o):
        if isinstance(o, np.int32): return int(o)
        return o

    with open("data/trials/SVM/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/SVM/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


def KNN_cv():
    global X
    X = X * 0.9998 + 0.0001

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import KFold

    try:
        with open("data/trials/KNN/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'k': hp.randint('k', 4)*2+1,
        'p': hp.randint('p', 2) + 1,
        'weights': hp.choice('weights', ['uniform', 'distance'])
    }

    def objective(space):
        stats = {'acc': [], 'auc': [], 'pre': [], 'recall': [], 'fscore': [], 'mcc': []}

        def task(obj):
            train_index, test_index = obj
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            clf = KNeighborsClassifier(
                n_neighbors=space['k'],
                p=space['p'],
                weights=space['weights']
            )

            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)

            stats['acc'].append(accuracy_score(y_test, y_pred))
            stats['auc'].append(roc_auc_score(y_test, y_pred))
            stats['pre'].append(precision_score(y_test, y_pred))
            stats['recall'].append(recall_score(y_test, y_pred))
            stats['fscore'].append(f1_score(y_test, y_pred))
            stats['mcc'].append(matthews_corrcoef(y_test, y_pred))

        with concurrent.futures.ThreadPoolExecutor(2) as executor:
            executor.map(task, KFold(5, shuffle=True, random_state=5).split(X, Y))

        m = {
            "accuracy": np.average(stats['acc']),
            "Precision": np.average(stats['pre']),
            "Recall": np.average(stats['recall']),
            "AUC": np.average(stats['auc']),
            "f_score": np.average(stats['fscore']),
            "mcc": np.average(stats['mcc'])
        }

        try:
            with open("data/trials/KNN/metric.txt") as f:
                max_mcc = float(f.read().strip())
        except FileNotFoundError:
            max_mcc = -1

        if m['mcc'] >= max_mcc:
            with open("data/trials/KNN/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/KNN/metric.txt", "w") as f:
                f.write(str(m['mcc']))
            with open("data/trials/KNN/metrics.json", "w") as f:
                json.dump(m, f)

        return {'loss': -m['mcc'], 'status': STATUS_OK, 'space': space}

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=25,
        trials=trials,
        timeout=60 * 20
    )

    def typer(o):
        if isinstance(o, np.int32): return int(o)
        return o

    with open("data/trials/KNN/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/KNN/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


# ansambles


def ET_cv():
    global X
    X = X * 0.9998 + 0.0001

    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import KFold

    try:
        with open("data/trials/ET/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
        'max_depth': hp.randint('max_depth', 1, 100),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
        'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None]),
        'min_samples_split': hp.uniform('min_samples_split', 0, 0.5),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
        'bootstrap': hp.choice('bootstrap', [
            {'value': True, 'oob_score': hp.choice('oob_score', [False, True])},
            {'value': False, 'oob_score': False},
        ]),
    }

    def objective(space):
        stats = {'acc': [], 'auc': [], 'pre': [], 'recall': [], 'fscore': [], 'mcc': []}

        def task(obj):
            train_index, test_index = obj
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            clf = ExtraTreesClassifier(
                n_estimators=space['n_estimators'],
                max_depth=space['max_depth'],
                criterion=space['criterion'],
                max_features=space['max_features'],
                class_weight=space['class_weight'],
                min_samples_split=space['min_samples_split'],
                min_samples_leaf=space['min_samples_leaf'],
                bootstrap=space['bootstrap']['value'],
                oob_score=space['bootstrap']['oob_score']
            )

            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)

            stats['acc'].append(accuracy_score(y_test, y_pred))
            stats['auc'].append(roc_auc_score(y_test, y_pred))
            stats['pre'].append(precision_score(y_test, y_pred))
            stats['recall'].append(recall_score(y_test, y_pred))
            stats['fscore'].append(f1_score(y_test, y_pred))
            stats['mcc'].append(matthews_corrcoef(y_test, y_pred))

        with concurrent.futures.ThreadPoolExecutor(5) as executor:
            executor.map(task, KFold(5, shuffle=True, random_state=5).split(X, Y))

        m = {
            "accuracy": np.average(stats['acc']),
            "Precision": np.average(stats['pre']),
            "Recall": np.average(stats['recall']),
            "AUC": np.average(stats['auc']),
            "f_score": np.average(stats['fscore']),
            "mcc": np.average(stats['mcc'])
        }

        try:
            with open("data/trials/ET/metric.txt") as f:
                max_mcc = float(f.read().strip())
        except FileNotFoundError:
            max_mcc = -1

        if m['mcc'] >= max_mcc:
            with open("data/trials/ET/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/ET/metric.txt", "w") as f:
                f.write(str(m['mcc']))
            with open("data/trials/ET/metrics.json", "w") as f:
                json.dump(m, f)

        return {'loss': -m['mcc'], 'status': STATUS_OK, 'space': space}

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

    with open("data/trials/ET/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/ET/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


def RF_cv():
    global X
    X = X * 0.9998 + 0.0001

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import KFold

    try:
        with open("data/trials/RF/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
        'max_depth': hp.randint('max_depth', 1, 100),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
        'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None]),
        'min_samples_split': hp.uniform('min_samples_split', 0, 0.5),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
        'bootstrap': hp.choice('bootstrap', [
            {'value': True, 'oob_score': hp.choice('oob_score', [False, True])},
            {'value': False, 'oob_score': False},
        ]),
    }

    def objective(space):
        stats = {'acc': [], 'auc': [], 'pre': [], 'recall': [], 'fscore': [], 'mcc': []}

        def task(obj):
            train_index, test_index = obj
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            clf = RandomForestClassifier(
                n_estimators=space['n_estimators'],
                max_depth=space['max_depth'],
                criterion=space['criterion'],
                max_features=space['max_features'],
                class_weight=space['class_weight'],
                min_samples_split=space['min_samples_split'],
                min_samples_leaf=space['min_samples_leaf'],
                bootstrap=space['bootstrap']['value'],
                oob_score=space['bootstrap']['oob_score']
            )

            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)

            stats['acc'].append(accuracy_score(y_test, y_pred))
            stats['auc'].append(roc_auc_score(y_test, y_pred))
            stats['pre'].append(precision_score(y_test, y_pred))
            stats['recall'].append(recall_score(y_test, y_pred))
            stats['fscore'].append(f1_score(y_test, y_pred))
            stats['mcc'].append(matthews_corrcoef(y_test, y_pred))

        with concurrent.futures.ThreadPoolExecutor(5) as executor:
            executor.map(task, KFold(5, shuffle=True, random_state=5).split(X, Y))

        m = {
            "accuracy": np.average(stats['acc']),
            "Precision": np.average(stats['pre']),
            "Recall": np.average(stats['recall']),
            "AUC": np.average(stats['auc']),
            "f_score": np.average(stats['fscore']),
            "mcc": np.average(stats['mcc'])
        }

        try:
            with open("data/trials/RF/metric.txt") as f:
                max_mcc = float(f.read().strip())
        except FileNotFoundError:
            max_mcc = -1

        if m['mcc'] >= max_mcc:
            with open("data/trials/RF/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/RF/metric.txt", "w") as f:
                f.write(str(m['mcc']))
            with open("data/trials/RF/metrics.json", "w") as f:
                json.dump(m, f)

        return {'loss': -m['mcc'], 'status': STATUS_OK, 'space': space}

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=250,
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


def AdaBoost_DT_cv():
    global X
    X = X * 0.9998 + 0.0001

    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import KFold

    try:
        with open("data/trials/AdaBoost_DT/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
        'learning_rate': hp.quniform('learning_rate', 0.01, 0.5, 0.01),
        'algorithm': hp.choice('algorithm', ['SAMME', 'SAMME.R']),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'splitter': hp.choice('splitter', ['best', 'random']),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
        'max_depth': hp.randint('max_depth', 1, 100),
        'min_samples_split': hp.uniform('min_samples_split', 0, 0.5),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5)
    }

    def objective(space):
        stats = {'acc': [], 'auc': [], 'pre': [], 'recall': [], 'fscore': [], 'mcc': []}

        def task(obj):
            train_index, test_index = obj
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            clf = AdaBoostClassifier(
                DecisionTreeClassifier(
                    criterion=space['criterion'],
                    splitter=space['splitter'],
                    max_features=space['max_features'],
                    min_samples_leaf=space['min_samples_leaf'],
                    min_samples_split=space['min_samples_split']
                ),
                n_estimators=space['n_estimators'],
                learning_rate=space['learning_rate'],
                algorithm=space['algorithm']
            )

            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)

            stats['acc'].append(accuracy_score(y_test, y_pred))
            stats['auc'].append(roc_auc_score(y_test, y_pred))
            stats['pre'].append(precision_score(y_test, y_pred))
            stats['recall'].append(recall_score(y_test, y_pred))
            stats['fscore'].append(f1_score(y_test, y_pred))
            stats['mcc'].append(matthews_corrcoef(y_test, y_pred))

        with concurrent.futures.ThreadPoolExecutor(5) as executor:
            executor.map(task, KFold(5, shuffle=True, random_state=5).split(X, Y))

        m = {
            "accuracy": np.average(stats['acc']),
            "Precision": np.average(stats['pre']),
            "Recall": np.average(stats['recall']),
            "AUC": np.average(stats['auc']),
            "f_score": np.average(stats['fscore']),
            "mcc": np.average(stats['mcc'])
        }

        try:
            with open("data/trials/AdaBoost_DT/metric.txt") as f:
                max_mcc = float(f.read().strip())
        except FileNotFoundError:
            max_mcc = -1

        if m['mcc'] >= max_mcc:
            with open("data/trials/AdaBoost_DT/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/AdaBoost_DT/metric.txt", "w") as f:
                f.write(str(m['mcc']))
            with open("data/trials/AdaBoost_DT/metrics.json", "w") as f:
                json.dump(m, f)

        return {'loss': -m['mcc'], 'status': STATUS_OK, 'space': space}

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=250,
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


def Bagging_DT_cv():
    global X
    X = X * 0.9998 + 0.0001

    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import KFold

    try:
        with open("data/trials/Bagging_DT/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
        'bootstrap_features': hp.choice('bootstrap_features', [False, True]),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'splitter': hp.choice('splitter', ['best', 'random']),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
        'max_depth': hp.randint('max_depth', 1, 100),
        'min_samples_split': hp.uniform('min_samples_split', 0, 0.5),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
        'bootstrap': hp.choice('bootstrap', [
            {'value': True, 'oob_score': hp.choice('oob_score', [False, True])},
            {'value': False, 'oob_score': False},
        ]),
    }

    def objective(space):
        stats = {'acc': [], 'auc': [], 'pre': [], 'recall': [], 'fscore': [], 'mcc': []}

        def task(obj):
            train_index, test_index = obj
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            clf = BaggingClassifier(
                DecisionTreeClassifier(
                    criterion=space['criterion'],
                    splitter=space['splitter'],
                    max_features=space['max_features'],
                    min_samples_leaf=space['min_samples_leaf'],
                    min_samples_split=space['min_samples_split']
                ),
                n_estimators=space['n_estimators'],
                bootstrap_features=space['bootstrap_features'],
                bootstrap=space['bootstrap']['value'],
                oob_score=space['bootstrap']['oob_score']
            )

            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)

            stats['acc'].append(accuracy_score(y_test, y_pred))
            stats['auc'].append(roc_auc_score(y_test, y_pred))
            stats['pre'].append(precision_score(y_test, y_pred))
            stats['recall'].append(recall_score(y_test, y_pred))
            stats['fscore'].append(f1_score(y_test, y_pred))
            stats['mcc'].append(matthews_corrcoef(y_test, y_pred))

        with concurrent.futures.ThreadPoolExecutor(5) as executor:
            executor.map(task, KFold(5, shuffle=True, random_state=5).split(X, Y))

        m = {
            "accuracy": np.average(stats['acc']),
            "Precision": np.average(stats['pre']),
            "Recall": np.average(stats['recall']),
            "AUC": np.average(stats['auc']),
            "f_score": np.average(stats['fscore']),
            "mcc": np.average(stats['mcc'])
        }

        try:
            with open("data/trials/Bagging_DT/metric.txt") as f:
                max_mcc = float(f.read().strip())
        except FileNotFoundError:
            max_mcc = -1

        if m['mcc'] >= max_mcc:
            with open("data/trials/Bagging_DT/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/Bagging_DT/metric.txt", "w") as f:
                f.write(str(m['mcc']))
            with open("data/trials/Bagging_DT/metrics.json", "w") as f:
                json.dump(m, f)

        return {'loss': -m['mcc'], 'status': STATUS_OK, 'space': space}

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=250,
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


def GradientBoost_cv():
    global X
    X = X * 0.9998 + 0.0001

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import KFold

    try:
        with open("data/trials/GradientBoost/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()


    space = {
        'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
        'learning_rate': hp.quniform('learning_rate', 0.01, 0.5, 0.01),
        'loss': hp.choice('loss', ['deviance', 'exponential']),
        'criterion': hp.choice('criterion', ['friedman_mse', 'mse']),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
        'max_depth': hp.randint('max_depth', 1, 100),
        'min_samples_split': hp.uniform('min_samples_split', 0, 0.5),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5)
    }

    def objective(space):
        stats = {'acc': [], 'auc': [], 'pre': [], 'recall': [], 'fscore': [], 'mcc': []}

        def task(obj):
            train_index, test_index = obj
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            clf = GradientBoostingClassifier(
                n_estimators=space['n_estimators'],
                learning_rate=space['learning_rate'],
                criterion=space['criterion'],
                max_features=space['max_features'],
                min_samples_leaf=space['min_samples_leaf'],
                min_samples_split=space['min_samples_split'],
                max_depth=space['max_depth'],
                loss=space['loss']
            )

            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)

            stats['acc'].append(accuracy_score(y_test, y_pred))
            stats['auc'].append(roc_auc_score(y_test, y_pred))
            stats['pre'].append(precision_score(y_test, y_pred))
            stats['recall'].append(recall_score(y_test, y_pred))
            stats['fscore'].append(f1_score(y_test, y_pred))
            stats['mcc'].append(matthews_corrcoef(y_test, y_pred))

        with concurrent.futures.ThreadPoolExecutor(5) as executor:
            executor.map(task, KFold(5, shuffle=True, random_state=5).split(X, Y))

        m = {
            "accuracy": np.average(stats['acc']),
            "Precision": np.average(stats['pre']),
            "Recall": np.average(stats['recall']),
            "AUC": np.average(stats['auc']),
            "f_score": np.average(stats['fscore']),
            "mcc": np.average(stats['mcc'])
        }

        try:
            with open("data/trials/GradientBoost/metric.txt") as f:
                max_mcc = float(f.read().strip())
        except FileNotFoundError:
            max_mcc = -1

        if m['mcc'] >= max_mcc:
            with open("data/trials/GradientBoost/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/GradientBoost/metric.txt", "w") as f:
                f.write(str(m['mcc']))
            with open("data/trials/GradientBoost/metrics.json", "w") as f:
                json.dump(m, f)

        return {'loss': -m['mcc'], 'status': STATUS_OK, 'space': space}

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=250,
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


def HistGradientBoost_cv():
    global X
    X = X * 0.9998 + 0.0001

    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import KFold

    try:
        with open("data/trials/HistGradientBoost/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'learning_rate': hp.quniform('learning_rate', 0.01, 0.5, 0.01),
        'loss': 'binary_crossentropy',
        'max_depth': hp.randint('max_depth', 1, 100),
        'min_samples_leaf': hp.randint('min_samples_leaf', 0, 50)
    }

    def objective(space):
        stats = {'acc': [], 'auc': [], 'pre': [], 'recall': [], 'fscore': [], 'mcc': []}

        def task(obj):
            train_index, test_index = obj
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            clf = HistGradientBoostingClassifier(
                learning_rate=space['learning_rate'],
                loss=space['loss'],
                min_samples_leaf=space['min_samples_leaf'],
                max_depth=space['max_depth']
            )

            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)

            stats['acc'].append(accuracy_score(y_test, y_pred))
            stats['auc'].append(roc_auc_score(y_test, y_pred))
            stats['pre'].append(precision_score(y_test, y_pred))
            stats['recall'].append(recall_score(y_test, y_pred))
            stats['fscore'].append(f1_score(y_test, y_pred))
            stats['mcc'].append(matthews_corrcoef(y_test, y_pred))

        with concurrent.futures.ThreadPoolExecutor(5) as executor:
            executor.map(task, KFold(5, shuffle=True, random_state=5).split(X, Y))

        m = {
            "accuracy": np.average(stats['acc']),
            "Precision": np.average(stats['pre']),
            "Recall": np.average(stats['recall']),
            "AUC": np.average(stats['auc']),
            "f_score": np.average(stats['fscore']),
            "mcc": np.average(stats['mcc'])
        }

        try:
            with open("data/trials/HistGradientBoost/metric.txt") as f:
                max_mcc = float(f.read().strip())
        except FileNotFoundError:
            max_mcc = -1

        if m['mcc'] >= max_mcc:
            with open("data/trials/HistGradientBoost/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/HistGradientBoost/metric.txt", "w") as f:
                f.write(str(m['mcc']))
            with open("data/trials/HistGradientBoost/metrics.json", "w") as f:
                json.dump(m, f)

        return {'loss': -m['mcc'], 'status': STATUS_OK, 'space': space}

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=150,
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


def logistic_regression_cv():
    global X
    X = X * 0.9998 + 0.0001

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    try:
        with open("data/trials/logistic_regression/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'C': hp.randint('C', 0, 1000000)/10,
        'l1_ratio': hp.uniform('l1_ratio', 0, 1),
        'fit_intercept': hp.choice('fit_intercept', [False, True]),
        'class_weight': hp.choice('class_weight', ['balanced', None])
    }

    def objective(space):
        stats = {'acc': [], 'auc': [], 'pre': [], 'recall': [], 'fscore': [], 'mcc': []}

        def task(obj):
            train_index, test_index = obj
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            clf = LogisticRegression(
                random_state=41,
                multi_class='ovr',
                n_jobs=2,
                C=space['C'],
                l1_ratio=space['l1_ratio'],
                fit_intercept=space['fit_intercept'],
                class_weight=space['class_weight'],
                solver='saga',
                penalty='elasticnet',
                max_iter=10000000,
                tol=1e-5
            )

            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)

            stats['acc'].append(accuracy_score(y_test, y_pred))
            stats['auc'].append(roc_auc_score(y_test, y_pred))
            stats['pre'].append(precision_score(y_test, y_pred))
            stats['recall'].append(recall_score(y_test, y_pred))
            stats['fscore'].append(f1_score(y_test, y_pred))
            stats['mcc'].append(matthews_corrcoef(y_test, y_pred))

        with concurrent.futures.ThreadPoolExecutor(5) as executor:
            executor.map(task, KFold(5, shuffle=True, random_state=5).split(X, Y))

        m = {
            "accuracy": np.average(stats['acc']),
            "Precision": np.average(stats['pre']),
            "Recall": np.average(stats['recall']),
            "AUC": np.average(stats['auc']),
            "f_score": np.average(stats['fscore']),
            "mcc": np.average(stats['mcc'])
        }

        try:
            with open("data/trials/logistic_regression/metric.txt") as f:
                max_mcc = float(f.read().strip())
        except FileNotFoundError:
            max_mcc = -1

        if m['mcc'] >= max_mcc:
            with open("data/trials/logistic_regression/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/logistic_regression/metric.txt", "w") as f:
                f.write(str(m['mcc']))
            with open("data/trials/logistic_regression/metrics.json", "w") as f:
                json.dump(m, f)

        return {'loss': -m['mcc'], 'status': STATUS_OK, 'space': space}

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=200,
        trials=trials,
        timeout=60 * 20
    )

    def typer(o):
        if isinstance(o, np.int32): return int(o)
        return o

    with open("data/trials/logistic_regression/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/logistic_regression/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


# ----------


def XGB():
    global X
    X = X * 0.9998 + 0.0001

    import xgboost as xgb

    with open("data/trials/XGB/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
    # x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=5)

    clf = xgb.XGBClassifier(
        use_label_encoder=False,
        n_estimators=space['n_estimators'],
        max_depth=int(space['max_depth']),
        learning_rate=space['learning_rate'],
        gamma=space['gamma'],
        min_child_weight=space['min_child_weight'],
        subsample=space['subsample'],
        colsample_bytree=space['colsample_bytree']
    )

    clf.fit(x_train, y_train,
            eval_set=[(x_train, y_train), (x_test, y_test)],
            eval_metric='logloss',
            early_stopping_rounds=3,
            verbose=True
            )

    y_pred = clf.predict(x_test)

    stats = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred),
        "f_score": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred)
    }

    with open("data/models/XGB/XGB.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("data/trials/XGB/params.pkl", "wb") as f:
        pickle.dump(clf.get_params(), f)
    with open("data/trials/XGB/stats.json", "w") as f:
        json.dump(stats, f)


def DT():
    global X
    X = X * 0.9998 + 0.0001

    from sklearn.tree import DecisionTreeClassifier

    with open("data/trials/DT/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

    clf = DecisionTreeClassifier(
        criterion=space['criterion'],
        splitter=space['splitter'],
        max_features=space['max_features'],
        min_samples_leaf=space['min_samples_leaf'],
        min_samples_split=space['min_samples_split']
    )

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    stats = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred),
        "f_score": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred)
    }

    with open("data/models/DT/DT.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("data/trials/DT/params.pkl", "wb") as f:
        pickle.dump(clf.get_params(), f)
    with open("data/trials/DT/stats.json", "w") as f:
        json.dump(stats, f)


def SVM():
    global X
    X = X * 0.9998 + 0.0001

    from sklearn.svm import SVC
    with open("data/trials/SVM/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

    clf = SVC(
        C=space['C'],
        random_state=space['random_state'],
        kernel=space['kernel']['type'],
        max_iter=100000,
        probability=True,
        verbose=True
    )

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    stats = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred),
        "f_score": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred)
    }

    with open("data/models/SVM/SVM.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("data/trials/SVM/params.pkl", "wb") as f:
        pickle.dump(clf.get_params(), f)
    with open("data/trials/SVM/stats.json", "w") as f:
        json.dump(stats, f)


def KNN():
    global X
    X = X * 0.9998 + 0.0001

    from sklearn.neighbors import KNeighborsClassifier

    with open("data/trials/KNN/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

    clf = KNeighborsClassifier(
        n_neighbors=space['k'],
        p=space['p'],
        weights=space['weights']
    )

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    stats = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred),
        "f_score": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred)
    }

    with open("data/models/KNN/KNN.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("data/trials/KNN/params.pkl", "wb") as f:
        pickle.dump(clf.get_params(), f)
    with open("data/trials/KNN/stats.json", "w") as f:
        json.dump(stats, f)


def Gaussian_NB():
    from sklearn.naive_bayes import GaussianNB

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    clf = GaussianNB()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    m = {
        "accuracy": np.average(accuracy_score(y_test, y_pred)),
        "precision": np.average(precision_score(y_test, y_pred)),
        "recall": np.average(recall_score(y_test, y_pred)),
        "auc": np.average(roc_auc_score(y_test, y_pred)),
        "f_score": np.average(f1_score(y_test, y_pred)),
        "mcc": np.average(matthews_corrcoef(y_test, y_pred))
    }

    pickle.dump(clf, open('data/models/Gaussian_NB/Gaussian_NB.pkl', 'wb'))
    with open("data/trials/Gaussian_NB/stats.json", "w") as f:
        json.dump(m, f)


def Bernoulli_NB():
    from sklearn.naive_bayes import BernoulliNB
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    clf = BernoulliNB()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    m = {
        "accuracy": np.average(accuracy_score(y_test, y_pred)),
        "precision": np.average(precision_score(y_test, y_pred)),
        "recall": np.average(recall_score(y_test, y_pred)),
        "auc": np.average(roc_auc_score(y_test, y_pred)),
        "f_score": np.average(f1_score(y_test, y_pred)),
        "mcc": np.average(matthews_corrcoef(y_test, y_pred))
    }

    pickle.dump(clf, open('data/models/Bernoulli_NB/Bernoulli_NB.pkl', 'wb'))
    with open("data/trials/Bernoulli_NB/stats.json", "w") as f:
        json.dump(m, f)


def Complement_NB():
    from sklearn.naive_bayes import ComplementNB
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    clf = ComplementNB()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    m = {
        "accuracy": np.average(accuracy_score(y_test, y_pred)),
        "precision": np.average(precision_score(y_test, y_pred)),
        "recall": np.average(recall_score(y_test, y_pred)),
        "auc": np.average(roc_auc_score(y_test, y_pred)),
        "f_score": np.average(f1_score(y_test, y_pred)),
        "mcc": np.average(matthews_corrcoef(y_test, y_pred))
    }

    pickle.dump(clf, open('data/models/Complement_NB/Complement_NB.pkl', 'wb'))
    with open("data/trials/Complement_NB/stats.json", "w") as f:
        json.dump(m, f)


def Multinomial_NB():
    from sklearn.naive_bayes import MultinomialNB
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    clf = MultinomialNB()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    m = {
        "accuracy": np.average(accuracy_score(y_test, y_pred)),
        "precision": np.average(precision_score(y_test, y_pred)),
        "recall": np.average(recall_score(y_test, y_pred)),
        "auc": np.average(roc_auc_score(y_test, y_pred)),
        "f_score": np.average(f1_score(y_test, y_pred)),
        "mcc": np.average(matthews_corrcoef(y_test, y_pred))
    }

    pickle.dump(clf, open('data/models/Multinomial_NB/Multinomial_NB.pkl', 'wb'))
    with open("data/trials/Multinomial_NB/stats.json", "w") as f:
        json.dump(m, f)


# ansambles


def ET():
    from sklearn.ensemble import ExtraTreesClassifier

    with open("data/trials/ET/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    clf = ExtraTreesClassifier(
        n_estimators=space['n_estimators'],
        max_depth=space['max_depth'],
        criterion=space['criterion'],
        max_features=space['max_features'],
        class_weight=space['class_weight'],
        min_samples_split=space['min_samples_split'],
        min_samples_leaf=space['min_samples_leaf'],
        bootstrap=space['bootstrap']['value'],
        oob_score=space['bootstrap']['oob_score'],
        verbose=True
    )

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    stats = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred),
        "f_score": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred)
    }

    with open("data/models/ET/ET.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("data/trials/ET/params.pkl", "wb") as f:
        pickle.dump(clf.get_params(), f)
    with open("data/trials/ET/stats.json", "w") as f:
        json.dump(stats, f)


def RF():
    from sklearn.ensemble import RandomForestClassifier

    with open("data/trials/RF/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(
        n_estimators=space['n_estimators'],
        max_depth=space['max_depth'],
        criterion=space['criterion'],
        max_features=space['max_features'],
        class_weight=space['class_weight'],
        min_samples_split=space['min_samples_split'],
        min_samples_leaf=space['min_samples_leaf'],
        bootstrap=space['bootstrap']['value'],
        oob_score=space['bootstrap']['oob_score'],
        verbose=True
    )

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    stats = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred),
        "f_score": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred)
    }

    with open("data/models/RF/RF.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("data/trials/RF/params.pkl", "wb") as f:
        pickle.dump(clf.get_params(), f)
    with open("data/trials/RF/stats.json", "w") as f:
        json.dump(stats, f)


def AdaBoost_DT():
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    with open("data/trials/AdaBoost_DT/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    clf = AdaBoostClassifier(
        DecisionTreeClassifier(
            criterion=space['criterion'],
            splitter=space['splitter'],
            max_features=space['max_features'],
            min_samples_leaf=space['min_samples_leaf'],
            min_samples_split=space['min_samples_split']
        ),
        n_estimators=space['n_estimators'],
        learning_rate=space['learning_rate'],
        algorithm=space['algorithm']
    )

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    stats = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred),
        "f_score": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred)
    }

    with open("data/models/AdaBoost_DT/AdaBoost_DT.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("data/trials/AdaBoost_DT/params.pkl", "wb") as f:
        pickle.dump(clf.get_params(), f)
    with open("data/trials/AdaBoost_DT/stats.json", "w") as f:
        json.dump(stats, f)


def Bagging_DT():
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier

    with open("data/trials/Bagging_DT/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    clf = BaggingClassifier(
        DecisionTreeClassifier(
            criterion=space['criterion'],
            splitter=space['splitter'],
            max_features=space['max_features'],
            min_samples_leaf=space['min_samples_leaf'],
            min_samples_split=space['min_samples_split']
        ),
        n_estimators=space['n_estimators'],
        bootstrap_features=space['bootstrap_features'],
        bootstrap=space['bootstrap']['value'],
        oob_score=space['bootstrap']['oob_score'],
        verbose=True
    )

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    stats = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred),
        "f_score": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred)
    }

    with open("data/models/Bagging_DT/Bagging_DT.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("data/trials/Bagging_DT/params.pkl", "wb") as f:
        pickle.dump(clf.get_params(), f)
    with open("data/trials/Bagging_DT/stats.json", "w") as f:
        json.dump(stats, f)


def GradientBoost():
    from sklearn.ensemble import GradientBoostingClassifier

    with open("data/trials/GradientBoost/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

    clf = GradientBoostingClassifier(
        n_estimators=space['n_estimators'],
        learning_rate=space['learning_rate'],
        criterion=space['criterion'],
        max_features=space['max_features'],
        min_samples_leaf=space['min_samples_leaf'],
        min_samples_split=space['min_samples_split'],
        max_depth=space['max_depth'],
        loss=space['loss'],
        verbose=True
    )

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    stats = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred),
        "f_score": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred)
    }

    with open("data/models/GradientBoost/GradientBoost.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("data/trials/GradientBoost/params.pkl", "wb") as f:
        pickle.dump(clf.get_params(), f)
    with open("data/trials/GradientBoost/stats.json", "w") as f:
        json.dump(stats, f)


def HistGradientBoost():
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingClassifier

    with open("data/trials/HistGradientBoost/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

    clf = HistGradientBoostingClassifier(
        learning_rate=space['learning_rate'],
        loss=space['loss'],
        min_samples_leaf=space['min_samples_leaf'],
        max_depth=space['max_depth'],
        verbose=True
    )

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    stats = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred),
        "f_score": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred)
    }

    with open("data/models/HistGradientBoost/HistGradientBoost.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("data/trials/HistGradientBoost/params.pkl", "wb") as f:
        pickle.dump(clf.get_params(), f)
    with open("data/trials/HistGradientBoost/stats.json", "w") as f:
        json.dump(stats, f)


def logistic_regression():
    from sklearn.linear_model import LogisticRegression

    with open("data/trials/logistic_regression/space.json", 'r') as f:
        space = json.loads(
            f.read().replace("'", '"').replace("False", "false").replace("True", 'true').replace("None", "null"))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    clf = LogisticRegression(
        random_state=41,
        multi_class='ovr',
        n_jobs=2,
        C=space['C'],
        l1_ratio=space['l1_ratio'],
        fit_intercept=space['fit_intercept'],
        class_weight=space['class_weight'],
        solver='saga',
        penalty='elasticnet',
        # max_iter=100,
        verbose=True
    )

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    stats = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred),
        "f_score": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred)
    }

    with open("data/models/logistic_regression/logistic_regression.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("data/trials/logistic_regression/params.pkl", "wb") as f:
        pickle.dump(clf.get_params(), f)
    with open("data/trials/logistic_regression/stats.json", "w") as f:
        json.dump(stats, f)


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
    with open("data/trials/neural_networks_kfold/space.json", 'r') as f:
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

try:
    ann = KerasClassifier(build_fn=create_model, epochs=len(pickle.load(open('data/trials/neural_networks/history.pkl','rb'))['loss'])-3, batch_size=64, verbose=2)
    ann._estimator_type = "classifier"


    def Stacking(estimators='All'):
        global X

        X = X * 0.998 + 0.001
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        import xgboost as xgb
        from sklearn.experimental import enable_hist_gradient_boosting
        from sklearn.linear_model import LogisticRegression
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

        clfs = {
            "XGB": xgb.XGBClassifier(**pickle.load(open('data/trials/XGB/params.pkl', 'rb'))),
            'LR': LogisticRegression(**pickle.load(open('data/trials/logistic_regression/params.pkl', 'rb'))),
            'ANN': ann,
            'SVM': SVC().set_params(**pickle.load(open('data/trials/SVM/params.pkl', 'rb'))),
            'GNB': GaussianNB(),
            'BNB': BernoulliNB(),
            'CNB': ComplementNB(),
            'MNB': MultinomialNB(),
            'RF': RandomForestClassifier().set_params(**pickle.load(open('data/trials/RF/params.pkl', 'rb'))),
            'HGB': HistGradientBoostingClassifier().set_params(
                **pickle.load(open('data/trials/HistGradientBoost/params.pkl', 'rb'))),
            'GB': GradientBoostingClassifier().set_params(
                **pickle.load(open('data/trials/GradientBoost/params.pkl', 'rb'))),
            'AB': AdaBoostClassifier().set_params(**pickle.load(open('data/trials/AdaBoost_DT/params.pkl', 'rb'))),
            'KNN': KNeighborsClassifier().set_params(**pickle.load(open('data/trials/KNN/params.pkl', 'rb'))),
            'ET': ExtraTreesClassifier().set_params(**pickle.load(open('data/trials/ET/params.pkl', 'rb'))),
            'DT': DecisionTreeClassifier().set_params(**pickle.load(open('data/trials/DT/params.pkl', 'rb'))),
            'B': BaggingClassifier().set_params(**pickle.load(open('data/trials/Bagging_DT/params.pkl', 'rb')))
        }

        if estimators == 'All':
            models = [(k, v) for k, v in clfs.items()]
        else:
            models = [(t.upper(), clfs[t.upper()]) for t in estimators.replace(' ', '').split(',')]

        clf = StackingClassifier(
            estimators=models,
            final_estimator=LogisticRegression(),
            verbose=0,
            n_jobs=3
        )

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        stats = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred),
            "f_score": f1_score(y_test, y_pred),
            "mcc": matthews_corrcoef(y_test, y_pred)
        }

        pickle.dump(clf, open('data/models/Stacking ({})/StackingClassifier.pkl'.format(estimators), 'wb'))
        with open("data/trials/Stacking ({})/stats.json".format(estimators), "w") as f:
            json.dump(stats, f)
except:
    pass


def search_data_size():
    import feature_extractor as fe
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import KFold

    sizes = list(range(5, 41, 1))
    values = []

    for size in sizes:
        print(size)

        frame = pandas.read_csv('data/datasets/OUTPUT2/dataset.csv')
        cols = [col for col in headers['stats'] if col in list(frame)][:-1]
        X = frame[cols]
        Y = frame['status'].to_numpy()

        X = fe.RFE(X, Y, size, 5)

        X = X.to_numpy() * 0.9998 + 0.0001

        trials = Trials()

        space = {
            'criterion': hp.choice('criterion', ['gini', 'entropy']),
            'splitter': hp.choice('splitter', ['best', 'random']),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
            'max_depth': hp.randint('max_depth', 1, 100),
            'min_samples_split': hp.uniform('min_samples_split', 0, 0.5),
            'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5)
        }

        def objective(space):
            stats = []

            def task(obj):
                train_index, test_index = obj
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]

                clf = DecisionTreeClassifier(
                    criterion=space['criterion'],
                    splitter=space['splitter'],
                    max_features=space['max_features'],
                    min_samples_leaf=space['min_samples_leaf'],
                    min_samples_split=space['min_samples_split']
                )

                clf.fit(x_train, y_train)

                y_pred = clf.predict(x_test)

                stats.append(accuracy_score(y_test, y_pred))

            with concurrent.futures.ThreadPoolExecutor(5) as executor:
                executor.map(task, KFold(5, shuffle=True, random_state=5).split(X, Y))

            return {'loss': -np.average(stats), 'status': STATUS_OK, 'space': space}

        fmin(
            objective,
            space,
            algo=tpe.suggest,
            max_evals=500,
            trials=trials,
            timeout=60 * 2
        )

        values.append(-trials.best_trial['result']['loss'])

    plt.plot(sizes, values)
    plt.title("поиск размерности")
    plt.xlabel = 'кол-во признаков'
    plt.ylabel = 'точность'
    plt.show()
    plt.clf()
    plt.close()