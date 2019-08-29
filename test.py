import os

from backbones.darknet import DarkNetDet
from backbones.hrnet import HRNetDet
from backbones.resnet import ResNetFPNDet
from src.rcnet import RcNet
from utils.coco import CocoUtil

from absl import app
from absl import flags
from absl import logging

from utils.pascal import PascalVOClUtil

flags.DEFINE_string('ckpt', None, 'model param path')
flags.DEFINE_string('a', 'eval', 'eval entire or inspect per image')
flags.DEFINE_integer('i', 512, 'input size')
flags.DEFINE_string('t', None, 'which task to use `coco` or `pascal`')
flags.DEFINE_string('bb', 'resnet', 'which backbone to use')

FLAGS = flags.FLAGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '40'


def main(_):
    input_shape = (FLAGS.i, FLAGS.i)
    if FLAGS.bb == 'darknet':
        det_body = DarkNetDet(input_shape, 85)
    else:
        det_body = ResNetFPNDet(input_shape, [1, 2, 2, 80])

    # task_util = CocoUtil("/media/csh/data/coco_2017", './model_data/coco_classes.txt')

    if FLAGS.t == 'pascal':
        task_util = PascalVOClUtil('/media/csh/data/VOCdevkit/VOC2012')
    else:
        task_util = CocoUtil("/media/csh/data/coco_2017", './model_data/coco_classes.txt')

    if FLAGS.a == 'eval':
        logging.info('eval model')
        rcm = RcNet(tu=task_util, detection_body=det_body, training=False,
                    ckpt_path=FLAGS.ckpt, center_confidence=0.4)
        rcm.eval()
        rcm.eval_on_test_dev(imgs=get_test_dev())

    else:
        logging.info('inspect model')
        rcm = RcNet(tu=task_util, detection_body=det_body, training=False,
                    ckpt_path=FLAGS.ckpt, center_confidence=0.3)
        rcm.interactive_inspect(show_anno=False)


def get_test_dev():
    import  json
    with open('/media/csh/data/coco_2017/annotations_test/image_info_test-dev2017.json') as f:
        a = json.load(f)

    imgs = ['/media/csh/data/coco_2017/test2017/%s' % x['file_name'] for x in a['images']]
    return imgs

if __name__ == '__main__':
    app.run(main)


