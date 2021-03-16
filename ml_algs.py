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
        axs[i % h, i // h].plot(history[metrics[i]])
        axs[i % h, i // h].plot(history['val_{}'.format(metrics[i])])
        axs[i % h, i // h].set(xlabel='epoch', ylabel=metrics[i])
        axs[i % h, i // h].set_title(metrics[i])
        axs[i % h, i // h].legend(['train', 'test'], loc='best')

    fig.savefig('data/trials/{}/stats.png'.format(dir))
    fig.clf()
    plt.close()


def get_rating():
    lst = os.listdir(os.getcwd() + '/data/trials')
    metrics = []
    for dir in lst:
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


def neural_networks_archSearch():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, losses

    import telepot


    metrics = [
        'accuracy',
        'Precision',
        'Recall',
        'AUC'
    ]


    tf.compat.v1.enable_eager_execution()

    try:
        with open("data/trials/neural_networks_archSearch/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    def layer(N, M=-1):
        if M == -1:
            M = N
        if N == 0:
            return None
        return hp.choice('layer_{}'.format(M - N), [
            {
                'activation': hp.choice('activation_{}'.format(M - N), [
                    'selu',
                    'relu',
                    'softmax',
                    'sigmoid',
                    'softplus',
                    'softsign',
                    'tanh',
                    'elu',
                    'exponential'
                ]),
                'nodes_count': hp.randint('nodes_count_{}'.format(M - N), 500)+2,
                'dropout': hp.choice('dropout_{}'.format(M - N), [
                    {'dropout_rate': hp.uniform('dropout_rate_{}'.format(M - N), 0, 0.5)},
                    None
                ]),
                'BatchNormalization': hp.choice('BatchNormalization_{}'.format(M - N), [False, True]),
                'next': layer(N - 1, M)
            },
            None
        ])

    space = {
        'decay_steps':
            hp.randint('decay_steps', 10000)*100,
        'layers': layer(5),
        'optimizer':
            hp.choice('optimizer', [
                {
                    'type': 'Adadelta',
                    'learning_rate': hp.uniform('Adadelta_lr', 0.001, 1),
                },
                {
                    'type': 'Adagrad',
                    'learning_rate': hp.uniform('Adagrad_lr', 0.001, 1),
                },
                {
                    'type': 'Adamax',
                    'learning_rate': hp.uniform('Adamax_lr', 0.001, 1),
                },
                {
                    'type': 'Adam',
                    'learning_rate': hp.uniform('Adam_lr', 0.001, 1),
                    'amsgrad': hp.choice('Adam_amsgrad', [False, True])
                },
                # {
                #     'type': 'Ftrl',
                #     'learning_rate': hp.uniform('Ftrl_lr', 0.001, 1),
                # },
                {
                    'type': 'Nadam',
                    'learning_rate': hp.uniform('Nadam_lr', 0.001, 1),
                },
                {
                    'type': 'RMSprop',
                    'learning_rate': hp.uniform('RMSprop_lr', 0.001, 1),
                    'centered': hp.choice('RMSprop_centered', [False, True]),
                    'momentum': hp.uniform('RMSprop_momentum', 0.001, 1),
                },
                {
                    'type': 'SGD',
                    'learning_rate': hp.uniform('SGD_lr', 0.001, 1),
                    'nesterov': hp.choice('SGD_nesterov', [False, True]),
                    'momentum': hp.uniform('SGD_momentum', 0.001, 1),
                }
            ]),
        'batch_size': 32,
        'init': hp.choice('init', [
            'glorot_normal',
            'truncated_normal',
            'glorot_uniform'
        ]),
        'trainable_BatchNormalization': hp.choice('trainable_BatchNormalization', [False, True]),
        'trainable_dropouts': hp.choice('trainable_dropouts', [False, True]),
        'shuffle': True
    }

    import tensorflow.keras.backend as K

    def f_score(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_val

    def objective(space):
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

        def scheduler(epoch, lr):
            return lr * tf.math.exp(-epoch / space['decay_steps'])

        def tf_callbacks():
            return [
                tf.keras.callbacks.ModelCheckpoint(
                    'data/models/neural_networks_archSearch/tmp.h5',
                    monitor='val_accuracy',
                    mode='max',
                    verbose=0,
                    save_best_only=True
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=15,
                    min_delta=0.0001,
                    mode='max',
                    verbose=0),
                tf.keras.callbacks.LearningRateScheduler(scheduler)
            ]

        if space['optimizer']['type'] == 'Adadelta':
            optimizer = optimizers.Adadelta()
        elif space['optimizer']['type'] == 'Adagrad':
            optimizer = optimizers.Adagrad()
        elif space['optimizer']['type'] == 'Adam':
            optimizer = optimizers.Adam()
        elif space['optimizer']['type'] == 'Adamax':
            optimizer = optimizers.Adamax()
        elif space['optimizer']['type'] == 'Ftrl':
            optimizer = optimizers.Ftrl()
        elif space['optimizer']['type'] == 'Nadam':
            optimizer = optimizers.Nadam()
        elif space['optimizer']['type'] == 'RMSprop':
            optimizer = optimizers.RMSprop()
        elif space['optimizer']['type'] == 'SGD':
            optimizer = optimizers.SGD()

        optimizer.learning_rate = space['optimizer']['learning_rate']
        if 'amsgrad' in space['optimizer']:
            optimizer.amsgrad = space['optimizer']['amsgrad']
        if 'centered' in space['optimizer']:
            optimizer.centered = space['optimizer']['centered']
        if 'momentrum' in space['optimizer']:
            optimizer.momentum = space['optimizer']['momentum']
        if 'nesterov' in space['optimizer']:
            optimizer.nesterov = space['optimizer']['nesterov']

        try:
            model.compile(
                optimizer=optimizer,
                loss=losses.BinaryCrossentropy(from_logits=True),
                metrics=metrics + [f_score]
            )

            history = model.fit(
                x_train, y_train,
                validation_data=(x_test, y_test),
                # validation_split=0.1,
                epochs=500,
                callbacks=tf_callbacks(),
                verbose=2,
                batch_size=space['batch_size'],
                shuffle=space['shuffle']
            )

            loss, acc, precision, recall, auc, fScore = model.evaluate(x_test, y_test, verbose=0)

            try:
                with open("data/trials/neural_networks_archSearch/metric.txt") as f:
                    max_fScore = float(f.read().strip())  # read best metric,
            except FileNotFoundError:
                max_fScore = -1

            m = {
                "loss": loss,
                "accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "AUC": auc,
                "f_score": fScore
            }

            if fScore > max_fScore:
                model.save("data/models/neural_networks_archSearch/nn1.h5")
                move('data/models/neural_networks_archSearch/tmp.h5', "data/models/neural_networks_archSearch/nn2.h5")
                with open("data/trials/neural_networks_archSearch/space.json", "w") as f:
                    f.write(str(space))
                with open("data/trials/neural_networks_archSearch/metric.txt", "w") as f:
                    f.write(str(fScore))
                with open("data/trials/neural_networks_archSearch/metrics.json", "w") as f:
                    json.dump(m, f)
                with open("data/trials/neural_networks_archSearch/history.pkl", 'wb') as f:
                    pickle.dump(history.history, f)

            try:
                telegram_info = pandas.read_csv('telegram_client.csv')
                bot = telepot.Bot(telegram_info['BOT_token'][0])
                bot.sendMessage(int(telegram_info['CHAT_ID'][0]), str(space))
                bot.sendMessage(int(telegram_info['CHAT_ID'][0]), str(m))
                draw(history.history, list(map(lambda x: x.lower(), metrics)) + ['loss', 'f_score'], 'neural_networks_archSearch')
                bot.sendPhoto(int(telegram_info['CHAT_ID'][0]), photo=open(
                    'data/trials/neural_networks_archSearch/stats.png', 'rb'))
            except:
                pass

            return {
                'loss': -fScore,
                'status': STATUS_OK,
                'history': history.history,
                'space': space,
                'metrics': m
            }
        except:
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
        max_evals=700,
                  # + len(trials),
        trials=trials,
        timeout=60 * 60 * 1
    )

    def typer(o):
        if isinstance(o, np.int32):
            return int(o)
        return o

    with open("data/trials/neural_networks_archSearch/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/neural_networks_archSearch/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


def find_best_NN(throughput=0.01, std=0.02, back_search=False):
    with open("data/trials/neural_networks_archSearch/results.pkl", 'rb') as f:
        data = pickle.load(f)

    metrics = []

    import numpy as np

    for idx, raw in enumerate(data.results):
        history = raw['history']

        if history:
            if back_search:
                maxf = -1
            else:
                maxf = np.argmax(history['val_f_score'])

            c_len = len(history['val_f_score']) // 3

            if (history['val_loss'][maxf] - history['loss'][maxf]) <= throughput \
                    and (history['val_accuracy'][maxf] - history['accuracy'][maxf]) >= -throughput \
                    and (history['val_f_score'][maxf] - history['f_score'][maxf]) >= -throughput \
                    and (history['val_precision'][maxf] - history['precision'][maxf]) >= -throughput \
                    and (history['val_auc'][maxf] - history['auc'][maxf]) >= -throughput \
                    and (history['val_recall'][maxf] - history['recall'][maxf]) >= -throughput \
                    and (np.array(history['val_loss'][:-c_len]) - np.array(history['loss'][:-c_len])).std() <= std\
                    and (np.array(history['val_precision'][:-c_len]) - np.array(history['precision'][:-c_len])).std() <= std\
                    and (np.array(history['val_recall'][:-c_len]) - np.array(history['recall'][:-c_len])).std() <= std:

                metrics.append([idx] + [history['val_f_score'][maxf]])

    id = pandas.DataFrame(metrics, columns=['id', 'max_f_score']).sort_values(
        'max_f_score', ascending=False
    ).head(1)['id']

    if not id.empty:
        print("id = {}; len = {}".format(int(id), len(metrics)))

        best = data.results[int(id)]

        draw(best['history'], [
            'accuracy',
            'precision',
            'recall',
            'auc',
            'f_score',
            'loss'
        ], 'best_nn')

        with open("data/trials/best_nn/space.json", "w") as f:
            f.write(str(best['space']))
        with open("data/trials/best_nn/metrics.json", "w") as f:
            json.dump(best['metrics'], f)

    else:
        print('ERROR: selection criteria are too strict!')


def neural_networks():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, losses

    metrics = ['accuracy']

    tf.compat.v1.enable_eager_execution()

    with open('data/trials/best_nn/space.json', 'r') as f:
        space = json.loads(
            str(f.read()).replace("'", '"').replace('None', 'null').replace('True', 'true').replace('False', 'false'))

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

    def scheduler(epoch, lr):
        return lr * tf.math.exp(-epoch / space['decay_steps'])

    def tf_callbacks():
        return [
            tf.keras.callbacks.ModelCheckpoint(
                'data/models/neural_networks/nn2.h5',
                monitor='val_accuracy',
                mode='max',
                verbose=0,
                save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=25,
                mode='max',
                verbose=0),
            tf.keras.callbacks.LearningRateScheduler(scheduler)
        ]

    if space['optimizer']['type'] == 'Adadelta':
        optimizer = optimizers.Adadelta()
    elif space['optimizer']['type'] == 'Adagrad':
        optimizer = optimizers.Adagrad()
    elif space['optimizer']['type'] == 'Adam':
        optimizer = optimizers.Adam()
    elif space['optimizer']['type'] == 'Adamax':
        optimizer = optimizers.Adamax()
    elif space['optimizer']['type'] == 'Ftrl':
        optimizer = optimizers.Ftrl()
    elif space['optimizer']['type'] == 'Nadam':
        optimizer = optimizers.Nadam()
    elif space['optimizer']['type'] == 'RMSprop':
        optimizer = optimizers.RMSprop()
    elif space['optimizer']['type'] == 'SGD':
        optimizer = optimizers.SGD()

    optimizer.learning_rate = space['optimizer']['learning_rate']
    if 'amsgrad' in space['optimizer']:
        optimizer.amsgrad = space['optimizer']['amsgrad']
    if 'centered' in space['optimizer']:
        optimizer.centered = space['optimizer']['centered']
    if 'momentrum' in space['optimizer']:
        optimizer.momentum = space['optimizer']['momentum']
    if 'nesterov' in space['optimizer']:
        optimizer.nesterov = space['optimizer']['nesterov']

    model.compile(
        optimizer=optimizer,
        loss=losses.BinaryCrossentropy(from_logits=True),
        metrics=metrics
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=160,
        callbacks=tf_callbacks(),
        verbose=2,
        batch_size=space['batch_size'],
        shuffle=space['shuffle']
    )

    loss, accuracy = model.evaluate(x_test, y_test)

    m = {"loss": loss, "accuracy": accuracy}

    model.save("data/models/neural_networks/nn1.h5")
    with open("data/trials/neural_networks/history.pkl", 'wb') as f:
        pickle.dump(history.history, f)
    with open("data/trials/neural_networks/metrics.json", "w") as f:
        json.dump(m, f)

    print(m)
    draw(history.history, metrics + ['loss'], 'neural_networks')


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


def SVM():
    from sklearn.svm import SVC

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    try:
        with open("data/trials/SVM/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'C': hp.uniform('C', 0, 100000),
        'random_state': 42,
        'kernel': hp.choice('kernel', [
            {
                'type': 'linear',
            },
            {
                'type': 'poly',
                'degree': hp.randint('degree_poly', 360),
                'coef0': hp.uniform('coef0_poly', -10, 10)
            },
            {
                'type': 'rbf',
                'gamma': hp.choice('gamma_rbf', ['scale', 'auto'])
            },
            {
                'type': 'sigmoid',
                'gamma': hp.choice('gamma_sigmoid', ['scale', 'auto']),
                'coef0': hp.uniform('coef0_sigmoid', -10, 10)
            },
            {
                'type': 'precomputed',
                'gamma': hp.choice('gamma_precomputed', ['scale', 'auto'])
            }
        ]),
    }

    def objective(space):
        clf = SVC(
            C=space['C'],
            random_state=space['random_state'],
            kernel=space['kernel']['type'],
            max_iter=10000,
            tol=1e-2
        )

        if 'coef0' in space['kernel']:
            clf.coef0 = space['kernel']['coef0']
        if 'degree' in space['kernel']:
            clf.degree = space['kernel']['degree']
        if 'gamma' in space['kernel']:
            clf.gamma = space['kernel']['gamma']

        try:
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
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
            with open("data/trials/SVM/metrics.json", "w") as f:
                json.dump(m, f)

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

    with open("data/trials/SVM/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/SVM/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


def KNN():
    from sklearn.neighbors import KNeighborsClassifier

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    try:
        with open("data/trials/kNN/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'k': hp.randint('k', 49) + 1,
        'p': hp.randint('p', 3) + 1,
        'weights': hp.choice('weights', ['uniform', 'distance']),
        'algorithm': hp.choice('algorithm', ['ball_tree', 'kd_tree', 'brute'])
    }

    def objective(space):
        clf = KNeighborsClassifier(
            n_neighbors=space['k'],
            p=space['p'],
            weights=space['weights'],
            algorithm=space['algorithm']
        )
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_true=y_test, y_pred=y_pred)

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
            with open("data/trials/kNN/metrics.json", "w") as f:
                json.dump(m, f)

        return {'loss': -acc, 'status': STATUS_OK, 'space': space}

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=100 + len(trials),
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


def ET():
    from sklearn.ensemble import ExtraTreesClassifier

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    try:
        with open("data/trials/ET/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'n_estimators': hp.randint('n_estimators', 98) + 2,
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
        'bootstrap': hp.choice('bootstrap', [False, True]),
        'oob_score': hp.choice('oob_score', [False, True]),
        'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None]),
    }

    def objective(space):
        clf = ExtraTreesClassifier(
            n_estimators=space['n_estimators'],
            criterion=space['criterion'],
            max_features=space['max_features'],
            bootstrap=space['bootstrap'],
            oob_score=space['oob_score'],
            class_weight=space['class_weight']
        )

        try:
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
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
            with open("data/trials/ET/metrics.json", "w") as f:
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

    with open("data/trials/ET/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/ET/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


def RF():
    from sklearn.ensemble import RandomForestClassifier

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    try:
        with open("data/trials/RF/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'n_estimators': hp.randint('n_estimators', 98) + 2,
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'max_features': hp.choice('max_features', ['sqrt', 'log2']),
        'bootstrap': hp.choice('bootstrap', [False, True]),
        'oob_score': hp.choice('oob_score', [False, True]),
        'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None]),
    }

    def objective(space):
        clf = RandomForestClassifier(
            n_estimators=space['n_estimators'],
            criterion=space['criterion'],
            max_features=space['max_features'],
            bootstrap=space['bootstrap'],
            oob_score=space['oob_score'],
            class_weight=space['class_weight']
        )

        try:
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
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
            with open("data/trials/RF/metrics.json", "w") as f:
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

    with open("data/trials/RF/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/RF/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


def AdaBoost_DT():
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    try:
        with open("data/trials/AdaBoost_DT/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'n_estimators': hp.randint('n_estimators',  98) + 2,
        'learning_rate': hp.uniform('learning_rate', 0, 1),
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
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
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
            with open("data/trials/AdaBoost_DT/metrics.json", "w") as f:
                json.dump(m, f)

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


def Bagging_DT():
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    try:
        with open("data/trials/Bagging_DT/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'n_estimators': hp.randint('n_estimators', 98) + 2,
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
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
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
            with open("data/trials/Bagging_DT/metrics.json", "w") as f:
                json.dump(m, f)

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


def GradientBoost():
    from sklearn.ensemble import GradientBoostingClassifier

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    try:
        with open("data/trials/GradientBoost/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'n_estimators': hp.randint('n_estimators',  98) + 2,
        'learning_rate': hp.uniform('learning_rate', 0, 1),
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
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
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
            with open("data/trials/GradientBoost/metrics.json", "w") as f:
                json.dump(m, f)

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


def HistGradientBoost():
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingClassifier

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    try:
        with open("data/trials/HistGradientBoost/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    space = {
        'learning_rate': hp.uniform('learning_rate', 0, 1),
        'loss': hp.choice('loss', ['auto', 'binary_crossentropy', 'categorical_crossentropy']),
    }

    def objective(space):
        clf = HistGradientBoostingClassifier(
            learning_rate=space['learning_rate'],
            loss=space['loss']
        )

        try:
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
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
            with open("data/trials/HistGradientBoost/metrics.json", "w") as f:
                json.dump(m, f)

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


# summary
# - Stacking


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
        ('SVM', SVC(
            C=42300.469903089426,
            random_state=42,
            kernel='sigmoid',
            gamma='scale',
            coef0=-4.669941268932323,
            max_iter=100000,
        )),
        ('GNB', GaussianNB()),
        ('BNB', BernoulliNB()),
        ('CNB', ComplementNB()),
        ('MNB', MultinomialNB()),
        ('RF', RandomForestClassifier(
            class_weight='balanced',
            n_estimators=96,
            max_features='sqrt',
            criterion='gini',
            bootstrap=False,
            oob_score=False
        )),
        ('HGBC', HistGradientBoostingClassifier(
            learning_rate=0.5410893212663248,
            loss='auto'
        )),
        ('GBC', GradientBoostingClassifier(
            learning_rate=0.7395211845757522,
            loss='deviance',
            n_estimators=92,
            criterion='mse',
            max_features=None
        )),
        ('AdaBoost_DT', AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=11),
            n_estimators=63,
            learning_rate=0.5492431252157344,
            algorithm='SAMME'
        )),
        ('kNN', KNeighborsClassifier(
            algorithm='kd_tree',
            weights='distance',
            n_neighbors=8,
            p=1
        )),
        ('ET', ExtraTreesClassifier(
            n_estimators=86,
            max_features=None,
            criterion='gini',
            class_weight='balanced_subsample'
        )),
        ('DT', DecisionTreeClassifier(
            criterion='entropy',
            splitter='best',
            max_features=None
        )),
        ('Bagging_DT', BaggingClassifier(
            DecisionTreeClassifier(max_depth=19),
            n_estimators=88,
            bootstrap_features=True
        ))
    ]

    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        stack_method='predict',
        verbose=2,
        n_jobs=3
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

