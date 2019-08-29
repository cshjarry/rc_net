import os

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import set_image_data_format, get_session

from backbones.darknet import DarkNetDet, darknet_body
from backbones.hrnet import HRNetDet
from backbones.resnet import ResNetFPNDet
from src.rcnet import RcNet

from absl import app
from absl import flags
from absl import logging

from utils.coco import CocoUtil
from utils.pascal import PascalVOClUtil

flags.DEFINE_integer('gn', 1, 'gpu numbers')
flags.DEFINE_integer('bs', 8, 'train batch size')
flags.DEFINE_string('d', None, 'dir where the tfrecord stay')
flags.DEFINE_integer('i', 512, 'input size')
flags.DEFINE_string('t', None, 'which task to use `coco` or `pascal`')
flags.DEFINE_string('bb', 'resnet', 'which backbone to use')


FLAGS = flags.FLAGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '40'


def main(_):
    input_shape = (FLAGS.i, FLAGS.i)
    assert FLAGS.bb in ['darknet', 'resnet'], 'backbone not supported'
    if FLAGS.bb == 'darknet':
        det_body = DarkNetDet(input_shape, 85)
    else:
        det_body = ResNetFPNDet(input_shape, [1, 2, 2, 80])

    assert FLAGS.t in ['coco', 'pascal']
    if FLAGS.t == 'pascal':
        task_util = PascalVOClUtil('/media/csh/data/VOCdevkit/VOC2012')
    else:
        # task_util = CocoUtil("/home/csh/coco_2017", './model_data/coco_classes.txt')
        task_util = CocoUtil("/media/csh/data/coco_2017", './model_data/coco_classes.txt')

    m = RcNet(tu=task_util, data_dir=FLAGS.d, detection_body=det_body, batch_size=FLAGS.bs, gpu_nums=FLAGS.gn)

    # check everything is right before training, usually only need one run before train
    # m.input_test()
    m.train()


def load_weight_test():
    det_body = DarkNetDet((None, None), 85)
    model = det_body.get_detection_body()
    # model.summary()
    print(len(model.layers))

    model.load_weights("yolo_pretrained.h5", by_name=True)

    w = model.layers[181].weights
    print(w)
    sess = get_session()
    print(sess.run([w])[0][0][0][0])


def count_obj_num():
    import json
    json_file = '/media/csh/data/coco_2017/annotations/instances_train2017.json'
    with open(json_file) as f:
        data = json.load(f)

    # desire_area = [[-float('inf'), 45**2], [30**2, 90 ** 2], [80 ** 2, 150 ** 2], [120 ** 2, 240 ** 2], [200 ** 2, float('inf')]]
    desire_area = [[0, 8 ** 2], [8 ** 2, 16 ** 2], [16 ** 2, 32 ** 2], [32 ** 2, 64**2], [64**2, float('inf')]]

    counter = {x: 0 for x in range(5)}
    img_ids = set()
    for x in data['annotations']:
        area = x['bbox'][2] *x['bbox'][3]
        img_ids.add(x['image_id'])
        for idx, (area_min, area_max) in enumerate(desire_area):
            if area_min <= area <= area_max:
                counter[idx] += 1
    #
    print(len(img_ids))
    counter = {x: counter[x] / len(img_ids) for x in counter}
    print(counter)


if __name__ == '__main__':
    # load_weight_test()
    app.run(main)
    # count_obj_num()


