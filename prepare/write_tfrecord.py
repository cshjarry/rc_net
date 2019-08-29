import cv2
import glob
import os
from functools import partial
from multiprocessing import Pool
import numpy as np
import six
import tensorflow as tf
from tensorflow.python.keras.backend import get_session, set_session


from pycocotools.coco import COCO
from utils.coco import CocoUtil

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    if six.PY3 and isinstance(value, six.text_type):
        value = six.binary_type(value, encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def test(data_dir):
    file_name = tf.gfile.Glob(os.path.join(data_dir, 'train*'))[0]
    tf_config = tf.ConfigProto(allow_soft_placement=False, device_count = {'GPU': 1})
    tf_config.gpu_options.allow_growth = True
    s = tf.Session(config=tf_config)
    set_session(s)

    sess = get_session()
    cnt = 0
    desire_area = [[0, 50 ** 2], [45**2, 100**2], [80 ** 2, 150 ** 2], [120 ** 2, 240 ** 2], [200**2, float('inf')]]

    area_m = {x: 0 for x in range(len(desire_area))}

    for example in tf.python_io.tf_record_iterator(file_name):
        # img,bbox = parse_example_proto(example)
        # print(sess.run([img, bbox])[1].shape)
        image, bboxes, labels, _,_,_, _ = parse_example_protobuf(example)
        cnt += 1
        a, b, c = sess.run([image, bboxes, labels])
        img = np.array(a)
        # bbox = np.array(b)
        # draw_box(img, bbox)
        # show_img(img)
        for box in b:
            box_area = box[2] * box[3]
            for idx in range(len(desire_area)):
                if box_area >= desire_area[idx][0] and box_area <= desire_area[idx][1]:
                    area_m[idx] += 1
        # break
    print(area_m)



# ***************************************************************************************************
def parse_example_protobuf(serialized_example, decode_image=True):
    feature_map = {
        'image_id': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image_encoded': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
        'bboxes': tf.io.VarLenFeature(dtype=tf.float32),
        'labels': tf.io.VarLenFeature(dtype=tf.int64),
        # 'iscrowd': tf.VarLenFeature(dtype=tf.int64),
        # 'category_id': tf.VarLenFeature(dtype=tf.int64),
        # 'area': tf.VarLenFeature(dtype=tf.float32),
        'height': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'width': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'bbox_nums': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
    }

    parsed_example = tf.io.parse_single_example(serialized_example, feature_map)
    if decode_image:
        image = tf.image.decode_image(parsed_example['image_encoded'], channels=3)
        image.set_shape((None, None, 3))
    else:
        image = parsed_example['image_encoded']
    # print(image.shape)
    # bbox_nums = tf.cast(parsed_example['bbox_nums'], tf.int16)
    bboxes = parsed_example['bboxes']
    bboxes = tf.sparse.to_dense(bboxes)
    bboxes = tf.reshape(bboxes, (-1, 4))
    labels = tf.cast(parsed_example['labels'], dtype=tf.float32)
    labels = tf.sparse.to_dense(labels)

    return image, bboxes, labels, parsed_example['height'], parsed_example['width'], parsed_example['bbox_nums'], parsed_example['image_id']

def _convert_single_example(image_id, image_array, annotations):
    image_id = str(image_id)
    # nparr = np.fromstring(image_bytes, np.uint8)
    # height, width, _ = cv2.imdecode(nparr, cv2.IMREAD_COLOR).shape
    # image_bytes = image_array.tobytes()
    success, image_encoded = cv2.imencode('.jpg', image_array)
    image_bytes = image_encoded.tobytes()
    height, width, _ = image_array.shape

    bbox_list = [];label_list = [];iscrowd_list = [];category_id_list = [];area_list = []
    bbox_nums = len(annotations)
    for anno in annotations:
        bbox = [float(x) for x in anno['bbox']]
        # iscrowd = anno['iscrowd']
        category_id = anno['category_id']

        label = anno['label']

        bbox_list.extend(bbox)
        label_list.append(label)
        # iscrowd_list.append(iscrowd)
        category_id_list.append(category_id)
        # area_list.append(area)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image_id': _bytes_feature(image_id),
        'image_encoded': _bytes_feature(image_bytes),
        'bboxes': _float_feature(bbox_list),
        'labels': _int64_feature(label_list),
        # 'iscrowd': _int64_feature(iscrowd_list),
        'category_id': _int64_feature(category_id_list),
        # 'area': _float_feature(area_list),
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'bbox_nums': _int64_feature(bbox_nums)
    }))
    return example


def shard_handler(data, coco=None, record_dest=None, voc_dir=None):
    shard_images, shard_name = data
    shard_path = os.path.join(record_dest, shard_name)

    print("shard: %s store images: %d" % (shard_name, len(shard_images)))

    chip_nums = 0
    examples = []
    if coco is not None:
        for image_path in shard_images:
            image_id = int(os.path.splitext(os.path.basename(image_path))[0])

            # image_bytes = tf.gfile.FastGFile(image_path, 'rb').read()
            image_array = cv2.imread(image_path)
            anno_ids = coco.getAnnIds([image_id])
            annos = coco.loadAnns(anno_ids)
            annos = [x.update({"label": CocoUtil.category_id_to_label(x['category_id'])}) for x in annos]
            example = _convert_single_example(image_id, image_array, annos)
            examples.append(example)

    elif voc_dir is not None:
        for image_path in shard_images:
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            image_array = cv2.imread(image_path)

            annos = voc_convert_annotation(voc_dir, image_id)
            example = _convert_single_example(image_id, image_array, annos)
            examples.append(example)

    writer = tf.python_io.TFRecordWriter(shard_path + "_%d" % len(shard_images))
    for _e in examples:
        writer.write(_e.SerializeToString())
    writer.close()
    print('shard: {} finish, find chip {}'.format(shard_name, chip_nums))


def coco_record_generate(coco_origin_dir, record_dest):
    data_type = 'val2017'
    data_dir = os.path.join(coco_origin_dir, data_type)
    train_images = glob.glob(os.path.join(data_dir, '*.jpg'))
    annotation_path = os.path.join(coco_origin_dir, 'annotations/instances_{}.json'.format(data_type))
    coco = COCO(annotation_path)
    print("find train image: %d" % len(train_images))

    shard_nums = 16

    shard_images_range = np.linspace(0, len(train_images), shard_nums + 1).astype(int)
    shard_images = [train_images[shard_images_range[i]:shard_images_range[i+1]] for i in range(shard_nums)]
    shard_names = ["%s-%03d-of-%03d" % (data_type, x, shard_nums) for x in range(shard_nums)]

    worker_pool = Pool(processes=2)

    worker_pool.map(partial(shard_handler, coco=coco, record_dest=record_dest), zip(shard_images, shard_names))



def voc_convert_annotation(voc_dir, image_id):
    import xml.etree.ElementTree as ET
    in_file = open('%s/Annotations/%s.xml' % (voc_dir, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    annotations = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = [int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)]
        b = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
        annotations.append({
            'bbox': b,
            'label': cls_id,
            'category_name': cls,
            'category_id': cls_id
        })

    return annotations


def voc2012_record_generate(voc_dir, record_dest):
    # /media/csh/data/VOCdevkit/VOC%s
    sets = [('trainval', 16), ('val', 4)]

    for image_set, shard_nums in sets:
        image_ids = open('%s/ImageSets/Main/%s.txt' % (voc_dir, image_set)).read().strip().split()
        set_image_dirs = ["%s/JPEGImages/%s.jpg" % (voc_dir, str(image_id)) for image_id in image_ids]
        # shard_nums = 16

        shard_images_range = np.linspace(0, len(set_image_dirs), shard_nums + 1).astype(int)
        shard_images = [set_image_dirs[shard_images_range[i]:shard_images_range[i + 1]] for i in range(shard_nums)]
        shard_names = ["%s-%03d-of-%03d" % (image_set, x, shard_nums) for x in range(shard_nums)]
        # print(shard_images)

        worker_pool = Pool(processes=2)

        worker_pool.map(partial(shard_handler, voc_dir=voc_dir, record_dest=record_dest), zip(shard_images, shard_names))


def main():
    data_dest = "/media/csh/data/record_repo"

    dataset_name = 'voc'
    record_dest = os.path.join(data_dest, dataset_name)
    if not os.path.exists(record_dest):
        os.makedirs(record_dest)

    if dataset_name == 'coco':
        coco_record_generate('/media/csh/data/coco_2017', record_dest)
    if dataset_name == 'voc':
        voc2012_record_generate('/media/csh/data/VOCdevkit/VOC2012', record_dest)



if __name__ == '__main__':
    main()
    # test('/media/csh/data/record_repo/coco')