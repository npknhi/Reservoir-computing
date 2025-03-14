import tensorflow as tf
import tensorflow_addons as tfa

def conv2d(inputs, filters, kernel_size=3, groups=1, padding='same', strides=1, name=None, trainable=True):
    """ Short calling version of quantized conv2d layers """
    return tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, groups=groups, use_bias=True, trainable=trainable, name=f'{name}_C2D')(inputs)


def bn(inputs, name=None, trainable=True):
    """ short calling version of Batch Norm """
    return tf.keras.layers.BatchNormalization(name=f'{name}_BN', trainable=trainable)(inputs)


def Act(inputs, act='relu', name=None):
    """ Short calling version of Activation. Quantized activation can be used also """
    return tf.keras.layers.Activation(act, name=f'{name}_{act}')(inputs)


def mp2d(inputs, size=2, name=None):
    """ Short calling version of MaxPooling 2D. """
    return tf.keras.layers.MaxPool2D(pool_size=size, name=f'{name}_maxpool')(inputs)


def conv_block(inputs, filters, kernel_size=3, padding='same', strides=1, groups=1, act='relu', maxpool=False, name=None, trainable=True):
    """ Typical convolution block used in VGG model: Conv-BN-(MP2)-Act """
    x = conv2d(inputs, filters, kernel_size=kernel_size, groups=groups, padding=padding, strides=strides, name=name, trainable=trainable)
    x = bn(x, name=name, trainable=trainable)
    if maxpool: 
        x = mp2d(x, name=name)
    x = Act(x, act=act, name=name)
    return x 


def fc(inputs, units, name=None, trainable=True):
    """ Short calling version of Fully-Connected (Dense) Layers """
    return tf.keras.layers.Dense(units, use_bias=True, name=f'{name}_fc', trainable=trainable)(inputs)


def spatial_dropout(inputs, rate=0.0, name=None):
    """ Short calling version of Spatial Dropout. """
    return tf.keras.layers.SpatialDropout2D(rate, name=f'{name}_sdropout')(inputs)


def ReservoirLayer(rnn_type, units, name=None, trainable=True):
    if rnn_type == 'ESN':
        return tfa.layers.ESN(units=units, connectivity=0.1, spectral_radius=0.95, leaky=0.9, return_sequences=True, name=f'{name}_ESN', trainable=trainable)
    elif rnn_type == 'GRU':
        return tf.keras.layers.GRU(units=units, return_sequences=True, name=f'{name}_GRU', trainable=trainable)
    else:
        raise ValueError(f"Unknown reservoir type: must be 'ESN' or 'GRU'. Got {rnn_type}.")