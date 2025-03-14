import tensorflow as tf
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Lambda, Flatten
from models.functional_layers import *

def classif_model(config=1, dataset='stl10', rnn_type='ESN', trainable=True):
    if dataset == 'stl10':
        input_shape = (96, 96, 3)
        F_res = 256
        F_conv = 32

    inputs = Input(shape=input_shape)
    # ENCODER: convolutional part
    x = conv_block(inputs, F_conv, kernel_size=5, strides=2, name='Encoder_S1', trainable=trainable)
    x = conv_block(x, 2*F_conv, kernel_size=3, strides=1, name='Encoder_S2', trainable=trainable)
    x = conv_block(x, 4*F_conv, kernel_size=3, strides=1, maxpool=True, name='Encoder_S3', trainable=trainable)
    # Convert images into sequences along the width axis to get (B, W, CH) output
    x_hor = Lambda(lambda img: tf.reshape(tf.transpose(img, perm=[0, 2, 3, 1]), [tf.shape(img)[0], tf.shape(img)[2], tf.shape(img)[3]*tf.shape(img)[1]]), name='Encoder_S4_H_Reshape')(x)
    if config == 1:
        x = ReservoirLayer(rnn_type, units=F_res, name='Encoder_S4_V', trainable=trainable)(x_hor)
    else:
        # Convert images into sequences along the height axis to get (B, H, CW) output
        x_ver = Lambda(lambda img: tf.reshape(tf.transpose(img, perm=[0, 1, 3, 2]), [tf.shape(img)[0], tf.shape(img)[1], tf.shape(img)[3]*tf.shape(img)[2]]), name='Encoder_S4_V_Reshape')(x)
        if config == 2:
            # Apply the reservoir computing along the horizontal axis of image (Left-to-Right)
            x_LR = ReservoirLayer(rnn_type, units=F_res, name='Encoder_S4_LR', trainable=trainable)(x_hor)
            # Apply the reservoir computing along the vertical axis of image (Top-to-Bottom)  
            x_TB = ReservoirLayer(rnn_type, units=F_res, name='Encoder_S4_TB', trainable=trainable)(x_ver)
            x = tf.keras.layers.Concatenate(axis=1, name='Encoder_S4_Concat')([x_LR, x_TB])
        else:
            # Flip the image for bi-directional Reservoir Computation to model the spatial features
            x_hor_reverse = Lambda(lambda img: tf.reverse(img, axis=[1]))(x_hor)
            x_ver_reverse = Lambda(lambda img: tf.reverse(img, axis=[1]))(x_ver)
            # Apply the reservoir computing along the horizontal axis of image (Left-to-Right)
            x_LR = ReservoirLayer(rnn_type, units=F_res, name='Encoder_S4_LR', trainable=trainable)(x_hor)
            # Apply the reservoir computing along the vertical axis of image (Top-to-Bottom)  
            x_TB = ReservoirLayer(rnn_type, units=F_res, name='Encoder_S4_TB', trainable=trainable)(x_ver)
            # Apply the reservoir computing along the reverse horizontal axis of image (Right-to-Left)
            x_RL = ReservoirLayer(rnn_type, units=F_res, name='Encoder_S4_RL', trainable=trainable)(x_hor_reverse)
            # Apply the reservoir computing along the reverse vertical axis of image (Bottom-to-Top) 
            x_BT = ReservoirLayer(rnn_type, units=F_res, name='Encoder_S4_BT', trainable=trainable)(x_ver_reverse)
            # Concate the features of all directions
            x = tf.keras.layers.Concatenate(axis=1, name='Encoder_S4_Concat')([x_LR, x_RL, x_TB, x_BT])
    
    x = tf.keras.layers.Permute(dims=(2, 1), name='Encoder_S5_Permute')(x)
    # Squeeze along the positional dimension to get the latent representation 
    x = fc(x, 4, name='Encoder_S5', trainable=trainable)
    # Flatten the features to obtain the embedding vector
    x = Flatten(name='Embedding')(x)
    # Output of classif model
    outputs = fc(x, 10, name=f'Classification_output')

    model = Model(inputs=inputs, outputs=outputs, name=f"classif_{config}")
    model.summary()
    tf.keras.utils.plot_model(model, to_file=f"checkpoints/plot/{model.name}.png", show_shapes=True)
    return model
