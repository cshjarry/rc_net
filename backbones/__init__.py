from tensorflow.python.keras import Model


class DetectionBody(object):
    name = ""
    pretrain_path = ""
    strides = []
    desire_area = []
    num_freez_backbone_layer = None

    def __init__(self, input_shape, output_channel):
        self.input_shape = input_shape
        self.output_channel = output_channel




    def get_detection_body(self, output_layers_nums = None) -> Model:
        pass
