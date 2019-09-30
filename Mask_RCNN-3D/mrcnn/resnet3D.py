import numpy as np
import math
import tensorflow as tf
import keras
import keras.backend as K
import keras.engine as KE
import keras.models as KM
from keras.layers import Conv3D, MaxPooling3D, AveragePooling3D
from keras.layers import BatchNormalization, Activation, Add, ZeroPadding3D

## Tensorflow backend default input shape D x H x W x C
global CHANNEL_AXIS = 4

############################################################
#  ResNet Backbone
############################################################

class BatchNorm(BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.
    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)

    
def identity_block(input_shape, filters, stage, block, kernel_size=(3,3,3), \
               strides=(2,2,2), use_bias=True, train_bn=True):
    """The identity_block is the block that has NO conv layer at shortcut connection
    
    # Arguments
        input_shape: input shape of the tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: in Keras, strides is default to 1, but we set our default to (2,2,2)
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
        
    Note that from stage 3, the first conv layer at main path is with strides=(2,2,2)
    And the shortcut should have strides=(2,2,2) as well
    """
    filt1, filt2, filt3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    x = Conv3D(filters=filt1, kernel_size=(1,1,1), strides=strides, \
               name=conv_name_base + '2a', use_bias=use_bias)(input_shape)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)
    
    x = Conv3D(filters=filt2, kernel_size=kernel_size, padding='same', \
               name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)
    
    x = Conv3D(filters=filt3, kernel_size=(1,1,1), \
              name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)
    
    x = Add()([x, input_shape])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_shape, filters, stage, block, kernel_size=(3,3,3), \
               strides=(2,2,2), use_bias=True, train_bn=True):
    """The conv_block is the block that has a conv layer at shortcut connection
    
    # Arguments
        input_shape: input shape of the tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: in Keras, strides is default to 1, but we set our default to (2,2,2)
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
        
    Note that from stage 3, the first conv layer at main path is with strides=(2,2,2)
    And the shortcut should have strides=(2,2,2) as well
    """
    filt1, filt2, filt3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    x = Conv3D(filters=filt1, kernel_size=(1,1,1), strides=strides, \
               name=conv_name_base + '2a', use_bias=use_bias)(input_shape)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
#     x = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    
    x = Conv3D(filters=filt2, kernel_size=kernel_size, padding='same',\
               name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
#     x = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    
    x = Conv3D(filters=filt3, kernel_size=(1,1,1), \
              name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)
#     x = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base + '2c')(x)
    
    shortcut = Conv3D(filters=filt3, kernel_size=(1,1,1), strides=strides, \
                      name=conv_name_base + '1', use_bias=use_bias)(input_shape)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)
#     shortcut = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base + '1')(shortcut)
    
    x = Add()([x, shortcut])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x

class ResNet3D(object):
    def build(input_shape, architecture, stage5=False, train_bn=True):
        # Stage 1
        x = ZeroPadding3D(padding=(3,3,3))(input_shape)
        x = Conv3D(filters=64, kernel_size=(7,7,7), strides=(2,2,2), name='conv1', use_bias=True)(x)
        x = BatchNorm(name='bn_conv1')(x, training=train_bn)
        x = Activation('relu')(x)
        C1 = x = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2), padding='same')(x)

        # Stage 2
        x = conv_block(x, filters=[64, 64, 256], stage=2, block='a',strides=(1,1,1),train_bn=train_bn)
        x = identity_block(x, filters=[64, 64, 256], stage=2, block='b', train_bn=train_bn)
        C2 = x = identity_block(x, filters=[64, 64, 256], stage=2, block='c', train_bn=train_bn)

        # Stage 3
        x = conv_block(x, filters=[128, 128, 512], stage=3, block='a', train_bn=train_bn)
        x = identity_block(x, filters=[128, 128, 512], stage=3, block='b', train_bn=train_bn)
        x = identity_block(x, filters=[128, 128, 512], stage=3, block='c', train_bn=train_bn)
        C3 = x = identity_block(x, filters=[128, 128, 512], stage=3, block='d', train_bn=train_bn)

        # Stage 4
        x = conv_block(x, filters=[256, 256, 1024], stage=4, block='a', train_bn=train_bn)
        block_count = {'resnet50': 6, 'resnet101': 23}[architecture]
        for i in range(block_count-1):
            x = identity_block(x, filters=[256, 256, 1024], stage=4, \
                               block=chr(98 + i), train_bn=train_bn)
        C4 = x

        # Stage 5    
        if stage5:
            x = conv_block(x, filters=[512, 512, 2048], stage=5, block='a', train_bn=train_bn)
            x = identity_block(x, filters=[512, 512, 2048], stage=5, block='b', train_bn=train_bn)
            C5 = x = identity_block(x, filters=[512, 512, 2048], stage=5, block='c', train_bn=train_bn)
        else:
            C5 = None
        return [C1, C2, C3, C4, C5]

@staticmethod
def ResNet50(input_shape, stage5, train_bn):
    """Builds a ResNet50 model."""
    model = ResNet3D.build(input_shape, 'resnet50', stage5, train_bn)
    return model

@staticmethod
def ResNet101(input_shape, stage5, train_bn):
    """Builds a ResNet101 model."""
    model = ResNet3D.build(input_shape, 'resnet101', stage5, train_bn)
    return model

############################################################
#  FPN (Feature Pyramid Network)
############################################################