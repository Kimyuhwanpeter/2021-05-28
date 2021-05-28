# -*- coding:utf -*-
import tensorflow as tf
import numpy as np

l2 = tf.keras.regularizers.l2
# https://openaccess.thecvf.com/content_CVPR_2020/papers/Svoboda_Two-Stage_Peer-Regularized_Feature_Recombination_for_Arbitrary_Image_Style_Transfer_CVPR_2020_paper.pdf
# Two-Stage Peer-Regularized Feature Recombination for Arbitrary Image Style Transfer   - Reference paper
# https://github.com/nnaisense/conditional-style-transfer

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def ResBlock(input, nof, weight_decay):

    h = tf.keras.layers.Conv2D(filters=nof,
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(input)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.PReLU()(h)

    h = tf.keras.layers.Conv2D(filters=nof,
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)

    return h + input

def Global_Encoder(input_shape=(64, 64, 256), N_style=512, weight_decay=0.00002):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.Conv2D(filters=N_style // 2,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=N_style // 2,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=N_style // 2,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def Encoder_multiscale(input_shape=(256, 256, 3), weight_decay=0.00002, nof=64, downsample=2):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=nof,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    
    i = 0
    mul = 2**i
    h = tf.keras.layers.Conv2D(filters=nof*mul*2,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    outputs = h
    model_enc_pre = tf.keras.Model(inputs=inputs, outputs=outputs)

    i = 1
    mul = 2**i
    h = inputs2 = tf.keras.Input((128, 128, 128))
    h = tf.keras.layers.Conv2D(filters=nof*mul*2,
                                         kernel_size=3,
                                         strides=2,
                                         padding="same",
                                         use_bias=False,
                                         kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    mul = 2**downsample
    for i in range(2):
        h = ResBlock(h, nof*mul, weight_decay)

    N_cont = nof * mul
    N_style = nof * mul * 2

    h = tf.keras.layers.Conv2D(filters=(N_cont + N_style),
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    outputs2 = h
    model_enc = tf.keras.Model(inputs2, outputs2)

    return model_enc, model_enc_pre, N_cont, N_style, mul


def Decoder_merge(input_shape=(64, 64, 768), weight_decay=0.00002, nof=64, downsample=2):

    h = inputs = tf.keras.Input(input_shape)

    mul = 2**downsample
    h = tf.keras.layers.Conv2D(filters=nof*mul,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def Decoder_model(input_shape=(64, 64, 64*4), downsample=2, weight_decay=0.00002, nof=64):

    h = inputs = tf.keras.Input(input_shape)

    mul = 2**downsample
    for i in range(2):
        h = ResBlock(h, nof*mul, weight_decay)

    for i in range(downsample):
        mul = 2**(downsample - i)
        h = tf.keras.layers.Conv2DTranspose(filters=nof * mul // 2,
                                            kernel_size=4,
                                            strides=2,
                                            padding="same",
                                            use_bias=False,
                                            kernel_regularizer=l2(weight_decay))(h)
        h = InstanceNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=3,
                               kernel_size=7,
                               strides=1,
                               padding="valid")(h)
    h = tf.nn.tanh(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def euclidean_dist(x, y):

    x_2 = tf.reduce_sum(tf.math.pow(x, 2), -1, keepdims=True)
    y_2 = tf.reduce_sum(tf.math.pow(y, 2), -1, keepdims=True)
    dists = x_2 - 2 * tf.matmul(x, tf.transpose(y, [1, 0])) + tf.transpose(y_2, [1, 0])
    return dists

def cosine_dist(x, y):

    x_norm = x / tf.norm(x, axis=-1, keepdims=True)
    y_norm = x / tf.norm(y, axis=-1, keepdims=True)
    dist_matrix = tf.matmul(x_norm, tf.transpose(y_norm, [1, 0]))
    return 1.0 - dist_matrix

def PeerRegularizationLayerAtt_model(input_shape1=(64, 64, 256),
                                     input_shape2=(64, 64, 512)):

    def featmap_2_pixwise(fmap):
        fmap = tf.transpose(fmap, [3, 1, 2, 0])
        fmap = tf.reshape(fmap, [fmap.shape[0], -1])
        fmap = tf.transpose(fmap, [1, 0])
        return fmap

    def pixwise_2_featmap(pwise, fmap_shape):
        pwise = tf.transpose(pwise, [1, 0])
        pwise = tf.reshape(pwise, [fmap_shape[3], fmap_shape[1], fmap_shape[2], fmap_shape[0]])
        pwise = tf.transpose(pwise, [3, 1, 2, 0])
        return pwise

    def recompose_style_features(inp_pwise, inp_pwise_style, peers_wise, peers_pwise_style):
        dist_matrix = euclidean_dist(inp_pwise, peers_wise) # KNN
        #topk_vals, topk_idxs = tf.math.top_k(dist_matrix, 5)    # Get the top-k (5)
        topk_vals = tf.math.top_k(tf.negative(dist_matrix), 5)
        topk_vals = tf.negative(topk_vals[0])

        inp_pixwise = tf.keras.layers.Dense(256 // 2)(inp_pwise)
        inp_pixwise = tf.keras.layers.BatchNormalization()(inp_pixwise)
        inp_pixwise = tf.keras.layers.ReLU()(inp_pixwise)
        inp_pixwise = tf.keras.layers.Dense(256 // 4)(inp_pixwise)
        inp_pixwise = tf.keras.layers.BatchNormalization()(inp_pixwise)
        inp_pixwise = tf.keras.layers.ReLU()(inp_pixwise)
        inp_pixwise = tf.expand_dims(inp_pixwise, 1)

        peers_pixwise = tf.keras.layers.Dense(256 // 2)(peers_wise)
        peers_pixwise = tf.keras.layers.BatchNormalization()(peers_pixwise)
        peers_pixwise = tf.keras.layers.ReLU()(peers_pixwise)
        peers_pixwise = tf.keras.layers.Dense(256 // 4)(peers_pixwise)
        peers_pixwise = tf.keras.layers.BatchNormalization()(peers_pixwise)
        peers_pixwise = tf.keras.layers.ReLU()(peers_pixwise)
        peers_pixwise = tf.expand_dims(peers_pixwise, 1)
        
        att1 = tf.keras.layers.Conv1D(filters=1,
                                      kernel_size=256 // 4,
                                      padding="same")(inp_pixwise)
        att2 = tf.keras.layers.Conv1D(filters=1,
                                      kernel_size=256 // 4,
                                      padding="same")(peers_pixwise)

        att = att1 + tf.transpose(att2, [1, 0, 2])
        att = att[:, :, 0]

        knn_filter = tf.cast(tf.greater(dist_matrix, topk_vals[:, 5-1:5]), tf.float32) * 1e-9
        att = tf.nn.softmax(tf.nn.softplus(att)+knn_filter, -1)

        out_pixwise_style = tf.matmul(att, peers_pwise_style)

        return out_pixwise_style

    h1 = inp_cont = tf.keras.Input(input_shape1, batch_size=1)
    h2 = inp_style = tf.keras.Input(input_shape2, batch_size=1)
    h3 = peers_cont = tf.keras.Input(input_shape1, batch_size=1)
    h4 = peers_style = tf.keras.Input(input_shape2, batch_size=1)

    inp_pixwise = featmap_2_pixwise(h1)
    inp_pixwise_style = featmap_2_pixwise(h2)
    peers_pixwise = featmap_2_pixwise(h3)
    peers_pixwise_style = featmap_2_pixwise(h4)

    out_pixwise_style = recompose_style_features(inp_pixwise, inp_pixwise_style, peers_pixwise, peers_pixwise_style)

    out_pixwise_style = pixwise_2_featmap(out_pixwise_style, inp_style.shape)

    outputs = tf.concat([h1, out_pixwise_style], -1)

    return tf.keras.Model(inputs=[inp_cont, inp_style, peers_cont, peers_style], outputs=outputs)

def Discriminator(input_shape=(256, 256, 3), input_shape2=(256, 256, 3), weight_decay=0.00002):

    h = inputs = tf.keras.Input(shape=input_shape, batch_size=1)
    h2 = inputs2 = tf.keras.Input(shape=input_shape, batch_size=1)

    h = tf.concat([h, h2], -1)
    noise = tf.random.normal([1, 256, 256, 6]) * 0.01
    h = h + noise
    # 1

    dim_ = dim = 64
    h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    for _ in range(3 - 1):
        dim = min(dim * 2, dim_ * 8)
        h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        #h = InstanceNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = tf.keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    #h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 3
    h = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)

    return tf.keras.Model(inputs=[inputs, inputs2], outputs=h)