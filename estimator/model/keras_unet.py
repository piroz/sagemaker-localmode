from tensorflow.python import keras
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout

INPUT_IMAGE_SIZE = 256
CONCATENATE_AXIS = -1
CONV_FILTER_SIZE = 4
CONV_STRIDE = 2
CONV_PADDING = (1, 1)
DECONV_FILTER_SIZE = 2
DECONV_STRIDE = 2
input_channel_count = 3
first_layer_filter_count = 64

def unet_model(output_channel_count):

    # (256 x 256 x input_channel_count)
    inputs = Input((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, input_channel_count))

    # エンコーダーの作成
    # (128 x 128 x N)
    enc1 = ZeroPadding2D(CONV_PADDING)(inputs)
    enc1 = Conv2D(first_layer_filter_count, CONV_FILTER_SIZE, strides=CONV_STRIDE)(enc1)

    # (64 x 64 x 2N)
    filter_count = first_layer_filter_count*2
    enc2 = _add_encoding_layer(filter_count, enc1)

    # (32 x 32 x 4N)
    filter_count = first_layer_filter_count*4
    enc3 = _add_encoding_layer(filter_count, enc2)

    # (16 x 16 x 8N)
    filter_count = first_layer_filter_count*8
    enc4 = _add_encoding_layer(filter_count, enc3)

    # (8 x 8 x 8N)
    enc5 = _add_encoding_layer(filter_count, enc4)

    # (4 x 4 x 8N)
    enc6 = _add_encoding_layer(filter_count, enc5)

    # (2 x 2 x 8N)
    enc7 = _add_encoding_layer(filter_count, enc6)

    # (1 x 1 x 8N)
    enc8 = _add_encoding_layer(filter_count, enc7)

    # デコーダーの作成
    # (2 x 2 x 8N)
    dec1 = _add_decoding_layer(filter_count, True, enc8)
    dec1 = concatenate([dec1, enc7], axis=CONCATENATE_AXIS)

    # (4 x 4 x 8N)
    dec2 = _add_decoding_layer(filter_count, True, dec1)
    dec2 = concatenate([dec2, enc6], axis=CONCATENATE_AXIS)

    # (8 x 8 x 8N)
    dec3 = _add_decoding_layer(filter_count, True, dec2)
    dec3 = concatenate([dec3, enc5], axis=CONCATENATE_AXIS)

    # (16 x 16 x 8N)
    dec4 = _add_decoding_layer(filter_count, False, dec3)
    dec4 = concatenate([dec4, enc4], axis=CONCATENATE_AXIS)

    # (32 x 32 x 4N)
    filter_count = first_layer_filter_count*4
    dec5 = _add_decoding_layer(filter_count, False, dec4)
    dec5 = concatenate([dec5, enc3], axis=CONCATENATE_AXIS)

    # (64 x 64 x 2N)
    filter_count = first_layer_filter_count*2
    dec6 = _add_decoding_layer(filter_count, False, dec5)
    dec6 = concatenate([dec6, enc2], axis=CONCATENATE_AXIS)

    # (128 x 128 x N)
    filter_count = first_layer_filter_count
    dec7 = _add_decoding_layer(filter_count, False, dec6)
    dec7 = concatenate([dec7, enc1], axis=CONCATENATE_AXIS)

    # (256 x 256 x output_channel_count)
    dec8 = Activation(activation='relu')(dec7)
    dec8 = Conv2DTranspose(output_channel_count, DECONV_FILTER_SIZE, strides=DECONV_STRIDE)(dec8)
    dec8 = Activation(activation='sigmoid')(dec8)

    return Model(inputs=inputs, outputs=dec8)

def _add_encoding_layer(filter_count, sequence):
    new_sequence = LeakyReLU(0.2)(sequence)
    new_sequence = ZeroPadding2D(CONV_PADDING)(new_sequence)
    new_sequence = Conv2D(filter_count, CONV_FILTER_SIZE, strides=CONV_STRIDE)(new_sequence)
    new_sequence = BatchNormalization()(new_sequence)
    return new_sequence

def _add_decoding_layer(filter_count, add_drop_layer, sequence):
    new_sequence = Activation(activation='relu')(sequence)
    new_sequence = Conv2DTranspose(filter_count, DECONV_FILTER_SIZE, strides=DECONV_STRIDE,
                                    kernel_initializer='he_uniform')(new_sequence)
    new_sequence = BatchNormalization()(new_sequence)
    if add_drop_layer:
        new_sequence = Dropout(0.5)(new_sequence)
    return new_sequence

