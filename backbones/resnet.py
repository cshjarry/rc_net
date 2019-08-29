import os

import tensorflow

import sys


sys.path.append(os.curdir)
from backbones import DetectionBody
from backbones.darknet import DarknetConv2D_BN_Leaky, DarknetConv2D

# from keras.layers import Lambda, UpSampling2D
# from keras.utils import get_file
# import keras_resnet
# import keras_resnet.models

# from . import retinanet
# from . import Backbone
# from ..utils.image import preprocess_image
from tensorflow.python.keras import backend, Input, Model
from tensorflow.python.keras.layers import UpSampling2D, Add, Conv2D, Concatenate
from tensorflow.python.ops.gen_control_flow_ops import Merge

from utils.misc import compose


class BatchNormalization(tensorflow.keras.layers.BatchNormalization):
    """
    Identical to keras.layers.BatchNormalization, but adds the option to freeze parameters.
    """
    def __init__(self, freeze, *args, **kwargs):
        self.freeze = freeze
        super(BatchNormalization, self).__init__(*args, **kwargs)

        # set to non-trainable if freeze is true
        self.trainable = not self.freeze

    def call(self, *args, **kwargs):
        # return super.call, but set training
        return super(BatchNormalization, self).call(training=(not self.freeze), *args, **kwargs)

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        config.update({'freeze': self.freeze})
        return config

def ResNet(inputs, blocks, block, include_top=True, classes=1000, freeze_bn=True, numerical_names=None, *args, **kwargs):
    """
    Constructs a `keras.models.Model` object using the given block count.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param block: a residual block (e.g. an instance of `keras_resnet.blocks.basic_2d`)

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

    :param numerical_names: list of bool, same size as blocks, used to indicate whether names of layers should include numbers or letters

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.blocks
        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> blocks = [2, 2, 2, 2]

        >>> block = keras_resnet.blocks.basic_2d

        >>> model = keras_resnet.models.ResNet(x, classes, blocks, block, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if tensorflow.keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if numerical_names is None:
        numerical_names = [True] * len(blocks)

    x = tensorflow.keras.layers.ZeroPadding2D(padding=3, name="padding_conv1")(inputs)
    x = tensorflow.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1")(x)
    x = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn_conv1")(x)
    x = tensorflow.keras.layers.Activation("relu", name="conv1_relu")(x)
    x = tensorflow.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

    features = 64

    outputs = []

    for stage_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            x = block(features, stage_id, block_id, numerical_name=(block_id > 0 and numerical_names[stage_id]), freeze_bn=freeze_bn)(x)

        features *= 2

        outputs.append(x)

    if include_top:
        assert classes > 0

        x = tensorflow.keras.layers.GlobalAveragePooling2D(name="pool5")(x)
        x = tensorflow.keras.layers.Dense(classes, activation="softmax", name="fc1000")(x)

        return tensorflow.keras.models.Model(inputs=inputs, outputs=x, *args, **kwargs)
    else:
        # Else output each stages features
        return tensorflow.keras.models.Model(inputs=inputs, outputs=outputs, *args, **kwargs)


def bottleneck_2d(filters, stage=0, block=0, kernel_size=3, numerical_name=False, stride=None, freeze_bn=False):
    """
    A two-dimensional bottleneck block.

    :param filters: the output’s feature space

    :param stage: int representing the stage of this block (starting from 0)

    :param block: int representing this block (starting from 0)

    :param kernel_size: size of the kernel

    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})

    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id

    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

    Usage:

        >>> import keras_resnet.blocks

        >>> keras_resnet.blocks.bottleneck_2d(64)
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if tensorflow.keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = tensorflow.keras.layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch2a".format(stage_char, block_char), **{"kernel_initializer": "he_normal"})(x)
        y = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2a".format(stage_char, block_char))(y)
        y = tensorflow.keras.layers.Activation("relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = tensorflow.keras.layers.ZeroPadding2D(padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))(y)
        y = tensorflow.keras.layers.Conv2D(filters, kernel_size, use_bias=False, name="res{}{}_branch2b".format(stage_char, block_char), **{"kernel_initializer": "he_normal"})(y)
        y = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2b".format(stage_char, block_char))(y)
        y = tensorflow.keras.layers.Activation("relu", name="res{}{}_branch2b_relu".format(stage_char, block_char))(y)

        y = tensorflow.keras.layers.Conv2D(filters * 4, (1, 1), use_bias=False, name="res{}{}_branch2c".format(stage_char, block_char), **{"kernel_initializer": "he_normal"})(y)
        y = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2c".format(stage_char, block_char))(y)

        if block == 0:
            shortcut = tensorflow.keras.layers.Conv2D(filters * 4, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch1".format(stage_char, block_char), **{"kernel_initializer": "he_normal"})(x)
            shortcut = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch1".format(stage_char, block_char))(shortcut)
        else:
            shortcut = x

        y = Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])
        y = tensorflow.keras.layers.Activation("relu", name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f


def ResNet101(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    if blocks is None:
        blocks = [3, 4, 23, 3]
    numerical_names = [False, True, True, False]

    return ResNet(inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d, include_top=include_top, classes=classes, *args, **kwargs)


def __create_pyramid_features(C3, C4, C5, feature_size=256):
    """ Creates the FPN layers on top of the backbone features.

    Args
        C3           : Feature stage C3 from the backbone.
        C4           : Feature stage C4 from the backbone.
        C5           : Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.

    Returns
        A list of feature levels [P3, P4, P5, P6, P7].
    """
    # upsample C5 to get P5 from the FPN paper
    P5           = tensorflow.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    # print(P5, C4)
    P5_upsampled = UpSampling2D(2, name='P5_upsampled')(P5)
    # P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, C4])

    P5           = tensorflow.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4           = tensorflow.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = UpSampling2D(2, name='P4_upsampled')(P4)
    # P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, C3])
    P4           = tensorflow.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3 = tensorflow.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = Add(name='P3_merged')([P4_upsampled, P3])
    P3 = tensorflow.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = tensorflow.keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = tensorflow.keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = tensorflow.keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return [P3, P4, P5, P6, P7]


def get_resnet_retinanet(input_shape, output_channels=None):
    inputs = Input(shape=(input_shape[0], input_shape[1], 3))
    resnet = ResNet101(inputs, include_top=False, freeze_bn=False)
    C3, C4, C5 = resnet.outputs[1:]

    features = __create_pyramid_features(C3, C4, C5)
    # print(features)
    predicts = make_last_layer(features, 256, output_channels)
    model = Model(inputs=inputs, outputs=predicts)
    print(model.outputs)
    return model

def make_last_layer(feature_maps, num_filters,output_channels):
    """
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1), name=output_layer_name))(x)
    :param feature_map:
    :return:
    """
    _y = []
    out_c = sum(output_channels)
    for idx, output in enumerate(feature_maps):
        x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(output)
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
            DarknetConv2D(out_c, (1, 1),  name="output%d" % (idx + 1)))(x)


        # _fp = compose(
        #     Conv2D(num_filters * 4, kernel_size=1, strides=1, padding='same'),
        #     Conv2D(num_filters, kernel_size=3, strides=1, padding='same'),
        #     # Conv2D(num_filters, kernel_size=1, strides=1, padding='same'),
        #     # Conv2D(num_filters * 2, kernel_size=3, strides=1, padding='same'),
        #     DarknetConv2D_BN_Leaky(out_c, (1, 1)),
        #     Conv2D(out_c, kernel_size=1, strides=1, padding='same', name="output%d" % (idx + 1)),
        # )(output)
        _y.append(y)
    return _y


class ResNetFPNDet(DetectionBody):
    name = 'resnet101_fpn_s5'
    pretrain_path = "model_data/ResNet-101-model.keras.h5"
    num_freez_backbone_layer = 362

    # 5 strides
    # strides = [4, 8, 16, 32, 64]
    # desire_area = [[-float('inf'), 45**2], [30**2, 90 ** 2], [80 ** 2, 150 ** 2], [120 ** 2, 240 ** 2], [200 ** 2, float('inf')]]

    # 4 strides
    strides = [8, 16, 32, 64, 128]
    # desire_area = [[-float('inf'), float('inf')],[-float('inf'), float('inf')],[-float('inf'), float('inf')],[-float('inf'), float('inf')],[-float('inf'), float('inf')],]
    desire_area = [[0, 150 ** 2], [120 ** 2, 200 ** 2], [180 ** 2, 250 ** 2], [220 ** 2, float('inf')], [400 ** 2, float('inf')]]
    def get_detection_body(self, output_layers_nums=4):
        # return darknet4det5(self.input_shape, output_channels=self.output_channel)
        return get_resnet_retinanet(self.input_shape, output_channels=self.output_channel)


if __name__ == '__main__':
    m = ResNetFPNDet(input_shape=[512, 512], output_channel=85)
    m.get_detection_body()
