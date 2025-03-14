import tensorflow as tf
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Lambda
from models.functional_layers import *

def reconstruction_model(config=1, dataset='stl10', rnn_type='ESN', trainable=True):
    if dataset == 'stl10':
        input_shape = (96, 96, 3)
        F_res = 256
        F_conv = 32

    inputs = Input(shape=input_shape)
    # ENCODER: Convolutional part
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

    # DECODER
    x = fc(x, 24, name='Decoder_S1')
    # Permute to get the shape (24, 256)
    x = tf.keras.layers.Permute(dims=(2, 1), name='Decoder_S1_Permute')(x)
    # Increase the dimensionality of the last axis to 24*32=768
    x = fc(x, 24*32, name='Decoder_S2')
    x = tf.keras.layers.MultiHeadAttention(num_heads=3, key_dim=16, value_dim=16, name='Decoder_S2_Attention')(x, x)
    # Reshape to image dimensions
    x = tf.keras.layers.Reshape((24, 24, 32), name='Decoder_S2_Reshape')(x)
    # Conv block to obtain tensor of shape (24, 24, 64)
    x = conv_block(x, 64, 3, name='Decoder_S3')
    # Depth2Space (aka Pixel shuffling) to double the spatial shape by reducing 4x the channel dimension (48, 48, 16)
    x = tf.nn.depth_to_space(x, 2)
    # Conv block to obtain tensor of shape (48, 48, 32)
    x = conv_block(x, 32, 3, name='Decoder_S4')
    # Depth2Space (aka Pixel shuffling) to double the spatial shape by reducing 4x the channel dimension (96, 96, 8)
    x = tf.nn.depth_to_space(x, 2)
    # Final conv to obtain 3 channel RGB 
    x_rec = conv2d(x, 3, name='Decoder_S5')
    x_rec = Act(x_rec, act='relu', name='Inter_Output')
    # Mask prediction 
    raw_mask = tf.where(inputs == 0., 1.0, 0.)
    mask = conv_block(inputs, 8, name='Mask_S1')
    mask = conv_block(mask, 1, act='sigmoid', name='Mask_S2')
    mask = tf.keras.layers.Multiply(name='Mask')([mask, raw_mask])
    # Finally compute the output by combining the reconstruction and the original image 
    outputs = tf.keras.layers.Add(name='Reconstruction_output')([x_rec*mask, (1.0 - mask)*inputs])

    model = Model(inputs=inputs, outputs=outputs, name=f"recon_{config}")
    model.summary()
    tf.keras.utils.plot_model(model, to_file=f"checkpoints/plot/{model.name}.png", show_shapes=True)
    return model
