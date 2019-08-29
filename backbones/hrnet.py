from functools import wraps
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D,BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Lambda

from backbones import DetectionBody
from utils.misc import compose

@wraps(Conv2D)
def ResnetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    _conv_kwargs = dict(kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))
    _conv_kwargs['padding'] = 'same'
    _conv_kwargs.update(kwargs)
    return Conv2D(*args, **_conv_kwargs)

def Conv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        ResnetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def identity(x):
    return x


def bottleneck(x, num_channels):
    shortcut = x

    x = compose(
        Conv2D_BN_Leaky(num_channels // 4, kernel_size=(1, 1), strides=(1, 1)),
        Conv2D_BN_Leaky(num_channels // 4, kernel_size=(3, 3), strides=(1, 1)),
        Conv2D_BN_Leaky(num_channels, kernel_size=(1, 1), strides=(1, 1)),
    )(x)

    x = Add()([x, shortcut])
    x = LeakyReLU()(x)
    return x


def downsample(input_channels, times):
    downconvs_ops = [ResnetConv2D(input_channels * pow(2, i + 1), (3, 3), strides=(2, 2),) for i in range(times)]
    return compose(*downconvs_ops)


def upsample(input_channels, times):
    return compose(
        UpSampling2D(pow(2, times)),
        ResnetConv2D(input_channels // pow(2, times), (3, 3), strides=(1, 1)))



def residual_unit(branches, num_bottlenecks):
    new_branch_head = []
    for _head in branches:
        width = int(_head.shape[-1])
        for _ in range(num_bottlenecks):
            _head = bottleneck(_head, width)
        new_branch_head.append(_head)

    return new_branch_head


def multi_resolution_block(branches):
    if len(branches) == 0: return []

    width = [int(x.shape[-1]) for x in branches]

    num_branches = len(branches)
    _branches = [[] for _ in range(num_branches)]

    # do upsample for all branches
    for _source_idx in range(num_branches):
        for _target_idx in range(num_branches):
            # print(_source_idx, _target_idx)
            if _target_idx == _source_idx:
                _branches[_target_idx].append(Lambda(identity)(branches[_source_idx]))
            elif _source_idx > _target_idx:
                # do upsample

                _branches[_target_idx].append(upsample(width[_source_idx], _source_idx - _target_idx)(branches[_source_idx]))
            else:
                # do downsample
                _branches[_target_idx].append(downsample(width[_source_idx], _target_idx - _source_idx)(branches[_source_idx]))

    _branches = [x[0] if len(x) == 1 else Add()(x) for x in _branches]
    # print(_branches)
    return _branches


def add_branch(branches):
    width = [int(x.shape[-1]) for x in branches]
    num_branches = len(branches)
    # print(width)

    # add  a new branch
    next_branch = [downsample(width[idx], num_branches - idx)(prev_branch) for idx, prev_branch in enumerate(branches)]

    if len(next_branch) == 1:
        next_branch = next_branch[0]
    else:
        # print(next_branch)
        next_branch = Add()(next_branch)

    _branches = multi_resolution_block(branches)

    _branches.append(next_branch)

    return _branches


def resolution_fusion(branches, nums_outputs=6):
    _target_resolution = branches[0].shape[1:3]

    low_branches_upsampled = [UpSampling2D(pow(2, idx + 1))(x) for idx, x in enumerate(branches[1:])]

    concat_branches = branches[:1] + low_branches_upsampled
    output = Concatenate(axis=-1)(concat_branches)

    last_feature_map = output
    _output = [last_feature_map]

    for i in range(nums_outputs - 1):
        last_feature_map = MaxPooling2D(strides=(2, 2))(last_feature_map)
        _output.append(last_feature_map)
    # print(_output)
    return _output



def make_predict_head(y, num_filters, output_channels, name=None):
    y = compose(
        Conv2D_BN_Leaky(num_filters, kernel_size=(1, 1), strides=(1, 1)),
        Conv2D_BN_Leaky(num_filters * 2, kernel_size=(1, 1), strides=(1, 1)),
        # Conv2D_BN_Leaky(num_filters, kernel_size=(1, 1), strides=(1, 1)),
        ResnetConv2D(output_channels, kernel_size=(1, 1), strides=(1, 1), name=name)
    )(y)
    # print(y)
    return y


def hrnet4det(input_shape: tuple, init_width:int, output_channels: int) -> Model:
    assert init_width in [18, 32, 40], 'init width not in allowed list'

    inputs = Input(shape=(input_shape[0], input_shape[1], 3))

    x = Conv2D_BN_Leaky(init_width, (3, 3), )(inputs)
    branches = [x]

    branches = residual_unit(branches, 2)
    branches = add_branch(branches)
    # print('stage1', branches)

    branches = residual_unit(branches, 3)
    branches = add_branch(branches)
    # print('stage2', branches)

    branches = residual_unit(branches, 3)
    branches = add_branch(branches)
    # print('stage3', branches)

    branches = residual_unit(branches, 4)
    branches = multi_resolution_block(branches)
    # print('stage4', branches)

    outputs = resolution_fusion(branches)
    # y1 = make_predict_head(outputs[0], 1024, output_channels, name="resolution1")
    y2 = make_predict_head(outputs[1], 512, output_channels, name='output1')
    y3 = make_predict_head(outputs[2], 512, output_channels, name='output2')
    y4 = make_predict_head(outputs[3], 512, output_channels, name='output3')
    y5 = make_predict_head(outputs[4], 512, output_channels, name='output4')
    y6 = make_predict_head(outputs[5], 512, output_channels, name='output5')

    return Model(inputs, [y2, y3, y4, y5, y6])


class HRNetDet(DetectionBody):
    name = 'hrnet'
    strides = [2, 4, 8, 16, 32]
    desire_area = [[0, 50 ** 2], [45 ** 2, 100 ** 2], [80 ** 2, 150 ** 2], [120 ** 2, 240 ** 2],
                   [200 ** 2, float('inf')]]

    def set_width(self, val):
        self.init_width = val

    def get_detection_body(self, init_width=18):
        return hrnet4det(self.input_shape, self.init_width, self.output_channel)

