"""YOLO_v3 Model Defined in Keras."""

from functools import wraps
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D

from backbones import DetectionBody
from utils.misc import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4), 'kernel_initializer': 'he_uniform'}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    # x = DarknetConv2D(32, (32, 32), **{'use_bias': False})(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.1)(x)

    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters, output_layer_name=None):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1), name=output_layer_name))(x)
    return x, y


# 3 output feature map
def darknet4det3(input_shape, output_channels):
    """Create YOLO_V3 model CNN body in Keras."""
    inputs = Input(shape=(input_shape[0], input_shape[1], 3))
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, output_channels, output_layer_name='output1')

    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, output_channels, output_layer_name='output2')

    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, output_channels, output_layer_name='output3')

    return Model(inputs, [y1, y2, y3])


# 4 output feature map
def darknet4det4(input_shape, output_channels):
    inputs = Input(shape=(input_shape[0], input_shape[1], 3))
    darknet = Model(inputs, darknet_body(inputs))

    # downsample
    subsample_x = compose(
        DarknetConv2D_BN_Leaky(1024, (3, 3), strides=(1, 1)),
        DarknetConv2D_BN_Leaky(512, (3, 3), strides=(1, 1)),
        DarknetConv2D_BN_Leaky(512, (3, 3), strides=(2, 2), padding='SAME')
    )(darknet.output)
    _, y4 = make_last_layers(subsample_x, 512, output_channels, output_layer_name="output4")

    x, y1 = make_last_layers(darknet.output, 512, output_channels, output_layer_name='output1')
    # print(type(x))

    # upsmaple
    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, output_channels, output_layer_name='output2')

    # upsample
    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, output_channels, output_layer_name='output3')

    # order in descending feature size
    return Model(inputs, [y3, y2, y1, y4])


# 5 output feature map
def darknet4det5(input_shape, output_channels):
    inputs = Input(shape=(input_shape[0], input_shape[1], 3), name='image')
    darknet = Model(inputs, darknet_body(inputs))

    # downsample
    subsample_x = compose(
        DarknetConv2D_BN_Leaky(1024, (3, 3), strides=(1, 1)),
        DarknetConv2D_BN_Leaky(512, (3, 3), strides=(1, 1)),
        DarknetConv2D_BN_Leaky(512, (3, 3), strides=(2, 2), padding='SAME')
    )(darknet.output)
    _, y5 = make_last_layers(subsample_x, 512, output_channels, output_layer_name="output5")

    x, y1 = make_last_layers(darknet.output, 512, output_channels, output_layer_name='output1')

    # upsmaple
    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, output_channels, output_layer_name='output2')

    # upsample
    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, output_channels, output_layer_name='output3')

    x = compose(
        DarknetConv2D_BN_Leaky(64, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[32].output])
    x, y4 = make_last_layers(x, 64, output_channels, output_layer_name='output4')

    # order in descending feature size
    return Model(inputs, [y4, y3, y2, y1, y5])


class DarkNetDet(DetectionBody):
    name = 'darknet_s4'
    pretrain_path = 'model_data/yolo_pretrained.h5'
    num_freez_backbone_layer = 185

    # 5 strides
    # strides = [4, 8, 16, 32, 64]
    # desire_area = [[-float('inf'), 45**2], [30**2, 90 ** 2], [80 ** 2, 150 ** 2], [120 ** 2, 240 ** 2], [200 ** 2, float('inf')]]

    # 4 strides
    strides = [8, 16, 32, 64]
    desire_area = [[0, 150 ** 2], [120 ** 2, 200 ** 2], [180 ** 2, 250 ** 2], [220 ** 2, float('inf')]]
    def get_detection_body(self, output_layers_nums=4):
        # return darknet4det5(self.input_shape, output_channels=self.output_channel)
        return darknet4det4(self.input_shape, output_channels=self.output_channel)


