import keras
from keras_contrib import backend as K
from keras_contrib.layers import SubPixelUpscaling
from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.utils import conv_utils
from keras.engine import InputSpec


def resize_images(images, size, method='bilinear', align_corners=False):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/image/resize_images .
    Args
        method: The method used for interpolation. One of ('bilinear', 'nearest', 'bicubic', 'area').
    """
    methods = {
        'bilinear': tf.image.ResizeMethod.BILINEAR,
        'nearest' : tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bicubic' : tf.image.ResizeMethod.BICUBIC,
        'area'    : tf.image.ResizeMethod.AREA,
    }
    return tf.image.resize_images(images, size, methods[method], align_corners)


class UpsampleLike(keras.layers.Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """
    def __init__(self, multiplier=(2, 2),
                 target_size=None,
                 data_format=None,
                 method='bilinear', **kwargs):
        super(UpsampleLike, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.target_size = conv_utils.normalize_tuple(target_size, 2, 'target_size')
        self.multiplier = conv_utils.normalize_tuple(multiplier, 2, 'multiplier')
        self.method = method
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        input_shape = K.get_variable_shape(inputs)
        if K.image_data_format() == 'channels_last':
            src_height, src_width = input_shape[1:3]
        else:
            src_height, src_width = input_shape[2:4]

        if self.target_size is not None:
            target_size = (src_height, src_width)
            while target_size[0] < self.target_size[0] or \
                    target_size[1] < self.target_size[1]:
                target_size = (target_size[0] * self.multiplier[0],
                               target_size[1] * self.multiplier[1])

        else:
            target_size = (src_height * self.multiplier[0], src_width * self.multiplier[1])

        return resize_images(inputs, target_size, method=self.method)

    def compute_output_shape(self, input_shape):
        if K.image_data_format() == 'channels_last':
            src_height, src_width = input_shape[1:3]
        else:
            src_height, src_width = input_shape[2:4]

        target_size = (src_height * self.multiplier[0], src_width * self.multiplier[1])

        if self.target_size is not None:
            while target_size[0] < self.target_size[0] or \
                    target_size[1] < self.target_size[1]:
                target_size = (target_size[0] * self.multiplier[0],
                               target_size[1] * self.multiplier[1])

        return (input_shape[0],) + target_size + (input_shape[-1],)

