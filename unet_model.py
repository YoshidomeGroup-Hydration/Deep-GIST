#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import time
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import time
import math

import json
import tensorflow.keras as keras
from tensorflow.keras import initializers
import tensorflow as tf
#print(tf.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import BatchNormalization, Activation, Input, Concatenate, add, UpSampling3D, LayerNormalization, Multiply, Conv3DTranspose
from tensorflow.keras.layers import Conv3D, Dropout, AveragePooling3D, MaxPooling3D, Layer
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU, Softmax
#import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
#from keras.utils.training_utils import multi_gpu_model
def mse(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))
def lmse(y_true, y_pred):
    mses = mse(y_true[:,16:32,16:32,16:32,:], y_pred[:,16:32,16:32,16:32,:])
    return K.sum(mses)

init = keras.initializers.he_normal(seed=4423)

def UNet(input_dim=(48,48,48,5), n_filters=16, kernel_size=3, n_depth=3, pool_size=2,
         stack_num_down=2, stack_num_up=2, activation='ReLU', norm="batch", dropout_rate=0,
         down_type="average", up_type="upsample", residual=False, SE=True):
    
    # Image input
    input_layer = Input(shape=input_dim)
    
    activation_func = eval(activation)
    X_skip = []
    
    filter_list = [n_filters*(2**i) for i in range(n_depth)]
    #depth_ = len(filter_num)

    X = input_layer

    # stacked conv3d before downsampling
    i_depth = 1
    block_name = "EC{}".format(i_depth)
        
    X = conv_block(X, channel=filter_list[0], kernel_size=kernel_size, stack_num=stack_num_down, 
           dilation_rate=1, activation=activation, 
           norm=norm, X_residual=None, SE=False, name=block_name)
    X_skip.append(X)

    # downsampling blocks
    for i, n_filters in enumerate(filter_list[1:]):
        i_depth = i+2
        X = encode_block(X, i_depth=i_depth, n_filters=n_filters,
                         kernel_size=kernel_size, pool_size=pool_size, norm=norm,
                         dropout_rate=dropout_rate, down_type=down_type,
                         residual=residual, SE=SE, activation=activation, stack_num=stack_num_down)        
        X_skip.append(X)
        
    # reverse indexing encoded feature maps
    X_skip = X_skip[::-1]
    # upsampling begins at the deepest available tensor
    X = X_skip[0]
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    depth_decode = len(X_decode)

    # reverse indexing filter numbers
    filter_num_decode = filter_list[:-1][::-1]

    # upsampling with concatenation
    for i in range(depth_decode):
        i_depth = depth_decode -1 -i
        X = decode_block(X, X_skip=X_decode[i], i_depth=i_depth,
                         n_filters=filter_num_decode[i], kernel_size=kernel_size,pool_size=pool_size,
                         norm=norm, dropout_rate=dropout_rate, up_type=up_type,
                         residual=residual, SE=SE, activation=activation, stack_num=stack_num_up)
        
    output_layer = Conv3D(1, (1), padding='same', kernel_initializer=init, name='Conv3D_last')(X)
    return Model(inputs=[input_layer], outputs=[output_layer])

def encode_block(X, i_depth, n_filters, kernel_size=3, pool_size=2, norm="batch",
            dropout_rate=0, down_type='average', residual=False, SE=False, activation="ReLU", stack_num=2):
    
    block_name = "EC{}".format(i_depth)
    
    X = downsample(X, n_filters/2, pool_size, down_type, kernel_size='auto', 
                 activation='ReLU', norm=False, name=block_name+"_down")

    
    if residual == "convolution":
        X_residual = Conv3D(n_filters, kernel_size=1, strides=1, 
                   padding='same', use_bias=False, name='{}_residual_conv'.format(block_name))(X)
    elif residual == True:
        X_residual = Concatenate(axis=-1)([X, X])
    else:
        X_residual = None

    X = conv_block(X, n_filters, kernel_size=kernel_size, stack_num=2, 
           dilation_rate=1, activation=activation, 
           norm=norm, X_residual=X_residual, SE=SE, name=block_name)

    #ec = pooling(i_depth, ec, ptype=ptype)
    return X
     
def decode_block(X, X_skip, i_depth, n_filters, kernel_size=3,pool_size=2, norm="batch", 
            dropout_rate=0, up_type='upsample', residual=False, SE=False, activation="ReLU", stack_num=2):
    """
    X: input tensol
    i_depth: UNetの階層番号
    X_skip: Encode_block からの skip connection に使われるテンソル
    residual: bool値。TrueならConv前後を残渣接続する。そのため X_residual にupsampleしたテンソルを保存する
    """
    
    block_name = "DC{}".format(i_depth)
    X = upsample(X, n_filters, pool_size, up_type=up_type, kernel_size=3, 
                 activation='ReLU', norm=False, name=block_name+"_up")
    
    if residual is True:
        X_residual = X
    else:
        X_residual = None
    
    X = Concatenate(axis=-1)([X, X_skip])
    
    X = conv_block(X, n_filters, kernel_size=kernel_size, stack_num=2, 
                   dilation_rate=1, activation=activation, 
                   norm=norm, X_residual=X_residual, SE=False, name=block_name)
        
    return X

def downsample(X, channel, pool_size, down_type, kernel_size='auto', 
                 activation='ReLU', norm=False, name='iwashi'):
    '''
    An overall downsample layer, based on one of the:
    (1) max-pooling, (2) average-pooling, (3) strided conv2d.
    
    encode_layer(X, channel, pool_size, pool, kernel_size='auto', 
                 activation='ReLU', norm=False, name='encode')
    
    Input
    ----------
        X: input tensor.
        pool_size: the encoding factor.
        channel: (for strided conv only) number of convolution filters.
        down_type: "max", "average", or "convolution". If convolution is seleted, convolution with stride 2 is applied. Else, pooling is applied. 
        kernel_size: size of convolution kernels. 
                     If kernel_size='auto', then it equals to the `pool_size`.
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
        norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
        
    '''
    # parsers
    if (down_type in ['max', 'average', 'convolution']) is not True:
        raise ValueError('Invalid down_type keyword')
        
    if down_type == 'max':
        X = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size), name='{}_maxpool'.format(name))(X)
        
    elif down_type == 'average':
        X = AveragePooling3D(pool_size=(pool_size, pool_size, pool_size), name='{}_avepool'.format(name))(X)
        
    else:
        if kernel_size == 'auto':
            kernel_size = pool_size
        
        # linear convolution with strides
        X = Conv3D(channel, kernel_size, strides=(pool_size, pool_size, pool_size), 
                   padding='valid', name='{}_stride_conv'.format(name))(X)
        
        # batch normalization
            # parsers
        if (norm in [False, True, 'batch', 'layer']) is not True:
            raise ValueError('Invalid norm keyword')

        # Batchnormalization as default
        if norm is True:
            norm_func = BatchNormalization
        elif norm == "batch":
            norm_func = BatchNormalization
        elif norm == "layer":
            norm_func = LayerNormalization
        else:
            norm_func = None
        
        if norm_func:
            X = norm_func(axis=4, name='{}_norm'.format(name))(X)
   
        # activation
        if activation is not None:
            activation_func = eval(activation)
            X = activation_func(name='{}_activation'.format(name))(X)
            
    return X

def upsample(X, channel, pool_size, up_type="upsample", kernel_size=3, 
                 activation='ReLU', norm=False, name='kohada'):
    '''
    An overall upsample layer, based on either upsampling or trans conv.
    
    upsample(X, channel, pool_size, unpool, kernel_size=3,
                 activation='ReLU', norm=False, name='decode')
    
    Input
    ----------
        X: input tensor.
        pool_size: the decoding factor.
        channel: (for trans conv only) number of convolution filters.
        up_type: "upsample" or 'upconvolution'.           
        kernel_size: size of convolution kernels. 
                     If kernel_size='auto', then it equals to the `pool_size`.
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
        norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ---------- 
        X: output tensor.
    
    * The defaut: `kernel_size=3`, is suitable for `pool_size=2`.
    
    '''

    # Batchnormalization as default
    if norm is True:
        norm_func = BatchNormalization
    elif norm == "batch":
        norm_func = BatchNormalization
    elif norm == "layer":
        norm_func = LayerNormalization
    else:
        norm_func = None
    
    # parsers
    if (up_type in ['upsample', 'upconvolution', 'simple']) is not True:
        raise ValueError('Invalid up_type keyword')
    
    if up_type=="simple":
        X = UpSampling3D(size=(pool_size, pool_size, pool_size), name='{}_unpool'.format(name))(X)

    if up_type=="upsample":
        X = UpSampling3D(size=(pool_size, pool_size, pool_size), name='{}_unpool'.format(name))(X)
        X = Conv3D(channel, kernel_size=1, padding='same', use_bias=False, 
                   dilation_rate=1, name='{}_conv'.format(name))(X)
        if norm_func:
            X = norm_func(axis=4, name='{}_norm'.format(name))(X)
        if activation is not None:
            activation_func = eval(activation)
            X = activation_func(name='{}_activation'.format(name))(X)
        
    
    elif up_type == "upconvolution":
        if kernel_size == 'auto':
            kernel_size = pool_size
            
        X = Conv3DTranspose(channel, kernel_size, strides=(pool_size, pool_size, pool_size), 
                            padding='same', name='{}_trans_conv'.format(name))(X)
        
        
        
        if norm_func:
            X = norm_func(axis=4, name='{}_norm'.format(name))(X)
            
        # activation
        if activation is not None:
            activation_func = eval(activation)
            X = activation_func(name='{}_activation'.format(name))(X)
        
    return X

    
def downsample_old(layer_num, layer_input, filters, f_size=(3,3,3), norm="batch", dropout_rate=0):
    d = Conv3D(filters, name="Conv3D_{}_1".format(layer_num), kernel_size=f_size, strides=(1), padding='same', kernel_initializer=init)(layer_input)
    if norm == "batch":
        d = BatchNormalization(axis=-1, name='Norm_{}_1'.format(layer_num))(d)
    elif norm =="layer":
        d = LayerNormalization(axis=-1, name='Norm_{}1'.format(layer_num))(d)
    elif norm == "instance":
        d = InstanceNormalization(axis=-1, name='Norm_{}_1'.format(layer_num))(d)
    d = Activation('relu', name='Activation_{}_1'.format(layer_num))(d)
    d = Conv3D(filters, name="Conv3D_{}_2".format(layer_num), kernel_size=f_size, strides=(1), padding='same', kernel_initializer=init)(d)
    #d = InstanceNormalization(axis = -1, center = False, scale = False)(d)
    if norm == "batch":
        d = BatchNormalization(axis=-1, name='Norm_{}_2'.format(layer_num))(d)
    elif norm =="layer":
        d = LayerNormalization(axis=-1, name='Norm_{}_2'.format(layer_num))(d)
    #elif norm == "instance":
     #   d = tfa.layers.InstanceNormalization(axis=-1, name='Norm_{}_2'.format(layer_num))(d)
    if dropout_rate:
        d = Dropout(dropout_rate, name='Dropout_{}'.format(layer_num))(d)
    d = Activation('relu', name='Activation_{}_2'.format(layer_num))(d)
    return d

def pooling(layer_num, layer_input, ptype='max', pool_size=2):
    if ptype=='max':
        p = MaxPooling3D(pool_size, name='MaxPool_{}'.format(layer_num))(layer_input)
    elif ptype=='average':
        p = AveragePooling3D(pool_size, name='AvePool_{}'.format(layer_num))(layer_input)
    else:
        print('Invalid ptype. Please chose {max,average}')
        p = MaxPooling3D(pool_size, name='MaxPool_{}'.format(layer_num))(layer_input)
    return p

def conv_block(X, channel, kernel_size=3, stack_num=2, 
               dilation_rate=1, activation='ReLU', 
               norm="batch",X_residual=None, SE=False, name="EC"):
    """
    X_residual: 残差接続のテンソル or None。テンソルが入力されると Convolution 後に Add＆Activate される
    残差接続しない場合はNone (default is None)

    """
    
    # parsers
    if (norm in [False, True, 'batch', 'layer']) is not True:
        raise ValueError('Invalid norm keyword')
        
    # Batchnormalization as default
    if norm is True:
        norm_func = BatchNormalization
    elif norm == "batch":
        norm_func = BatchNormalization
    elif norm == "layer":
        norm_func = LayerNormalization
    elif norm is False:
        norm_func = None
       
    # stacking Convolutional layers
    for i in range(stack_num):
        
        activation_func = eval(activation)
        
        # linear convolution
        X = Conv3D(channel, kernel_size, padding='same', 
                   dilation_rate=dilation_rate, name='{}_conv{}'.format(name, i))(X)
        
        # batch normalization
        if norm_func:
            X = norm_func(axis=4, name='{}_norm{}'.format(name, i))(X)
        
        # activation
        activation_func = eval(activation)
        X = activation_func(name='{}_activation{}'.format(name, i))(X)
        
    if X_residual is not None:
        X = add([X_residual, X], name='{}_residual_add'.format(name))
        activation_func = eval(activation)
        X = activation_func(name='{}_residual_activation'.format(name))(X)
    if SE:
        X = SE_Block(X, name='{}_SE'.format(name))
        
    return X

def upsample_old(layer_num,layer_input, skip_input, filters, f_size=(3,3,3), norm="batch", dropout_rate=0):
    u = UpSampling3D(size=2, name='UpSample_{}'.format(layer_num))(layer_input)
    u = Conv3D(filters, kernel_size=(2), name='Conv3D_{}_0'.format(layer_num), strides=(1), padding='same', kernel_initializer=init)(u)
    u = Concatenate()([u, skip_input])
    u = Conv3D(filters, kernel_size=f_size, name='Conv3D_{}_1'.format(layer_num), strides=(1), padding='same', kernel_initializer=init)(u)
    if norm == "batch":
        u = BatchNormalization(axis=-1, name='Norm_{}_1'.format(layer_num))(u)
    elif norm =="layer":
        u = LayerNormalization(axis=-1, name='Norm_{}_1'.format(layer_num))(u)
    elif norm == "instance":
        u = tfa.layers.InstanceNormalization(axis=-1, name='Norm_{}_1'.format(layer_num))(u)
    u = Activation('relu')(u)
    u = Conv3D(filters, kernel_size=f_size, name='Conv3D_{}_2'.format(layer_num), strides=(1), padding='same', kernel_initializer=init)(u)
    if norm == "batch":
        u = BatchNormalization(axis=-1, name='Norm_{}_2'.format(layer_num))(u)
    elif norm =="layer":
        u = LayerNormalization(axis=-1, name='Norm_{}_2'.format(layer_num))(u)
    #elif norm == "instance":
     #   u = InstanceNormalization(axis=-1, name='Norm_{}_2'.format(layer_num))(u)
    if dropout_rate:
        u = Dropout(dropout_rate, name='Dropout_{}'.format(layer_num))(u)
    u = Activation('relu')(u)
    return u

def SE_Block(X, name):
    """
    sc-SE block
    """
    X_residual = X
    X = Conv3D(1, kernel_size=(1), name='{}_Conv'.format(name), strides=(1), padding='same', kernel_initializer=init)(X)
    X = Activation("sigmoid", name='{}_Sigmoid'.format(name))(X)
    X = Multiply()([X, X_residual])
    return X
   

def Spatial_SelfAttention(inputs, i_depth):
    #projection_dim = embed_dim // num_heads # 64/4=16
    batch_size = K.int_shape(inputs)[0] # 32とか
    voxel_x = K.int_shape(inputs)[1]#48
    voxel_y = K.int_shape(inputs)[2]
    voxel_z = K.int_shape(inputs)[3]
    num_ch = K.int_shape(inputs)[4] #5
    #num_outch = K.int_shape(inputs)[4] #今は同じ。論文はoutch=inch//8
    N = voxel_x*voxel_y*voxel_z

    query = Conv3D(1, (1), name="SA{}_ConvQ".format(i_depth), padding="same")(inputs)#(batch,x,y,z,ch)->(batch,x,y,z,1)
    key   = Conv3D(1, (1), name="SA{}_ConvK".format(i_depth), padding="same")(inputs)
    value = Conv3D(1, (1), name="SA{}_ConvV".format(i_depth), padding="same")(inputs)

    query = K.reshape(query, (batch_size, N, 1))#->(batch,x*y*z,ch)
    key   = K.reshape(key,   (batch_size, N, 1))
    value = K.reshape(value, (batch_size, N, 1))

    score = tf.matmul(query, key, transpose_b=True)#keyの最後2軸転置して積->(batch,xyz,xyz)
    #score = score/K.sqrt(K.cast(num_outch, 'float32'))

    weights = Activation('softmax', name='SA{}_weights'.format(i_depth))(score)

    attention0 = tf.matmul(weights, value)#->(batch,xyz,1)
    #attention = K.permute_dimensions(attention, (0, 2, 1, 3))
    attention1 = K.reshape(attention0, (batch_size, voxel_x, voxel_y, voxel_z, 1))#(batch,x,y,z,1)
    attention2 = K.repeat_elements(attention1, num_ch, axis=4)#(batch,x,y,z,1)->(batch,x,y,z,ch)
    #output = Dense(embed_dim, name='attention{}'.format(num))(attention)
    return attention2

def AttentionBlock(inputs, i_depth):
    attn_output = Spatial_SelfAttention(inputs, i_depth)
    #attn_output = Dropout(0.25, name='SA{}_Drop1'.format(i_depth))(attn_output)#(batch,w,h,ch)
    out1 = LayerNormalization(name='SA{}_LayerNormalization'.format(i_depth))(Add()([inputs, attn_output]))
    #out2 = Dropout(0.5, name='SA{}_Drop2'.format(i_depth))(out1)
    out3 = Activation('relu', name='SA{}_Activation'.format(i_depth9))(out1)
    return out3

def gelu_(X):

    return 0.5*X*(1.0 + tf.math.tanh(0.7978845608028654*(X + 0.044715*tf.math.pow(X, 3))))

class GELU(Layer):
    '''
    Gaussian Error Linear Unit (GELU), an alternative of ReLU
    
    Y = GELU()(X)
    
    ----------
    Hendrycks, D. and Gimpel, K., 2016. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415.
    
    Usage: use it as a tf.keras.Layer
    
    
    '''
    def __init__(self, trainable=False, **kwargs):
        super(GELU, self).__init__(**kwargs)
        self.supports_masking = True
        self.trainable = trainable

    def build(self, input_shape):
        super(GELU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return gelu_(inputs)

    def get_config(self):
        config = {'trainable': self.trainable}
        base_config = super(GELU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def compute_output_shape(self, input_shape):
        return input_shape
