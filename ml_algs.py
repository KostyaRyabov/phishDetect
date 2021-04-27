import os
import json

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
    elif space['optimizer']['type'] == 'RMSprop':
        optimizer = optimizers.RMSprop()
    elif space['optimizer']['type'] == 'SGD':
        optimizer = optimizers.SGD()

    optimizer.learning_rate = space['learning_rate']

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
        metrics=['accuracy']
    )

    return model

ann = KerasClassifier(build_fn=create_model, epochs=200, batch_size=64, verbose=2)
ann._estimator_type = "classifier"