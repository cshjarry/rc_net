import json
import os
import random

import numpy as np
import tensorflow as tf
import cv2

from tensorflow.python.keras.callbacks import TensorBoard, CSVLogger
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.backend import get_session
from tqdm import tqdm
from absl import logging

from backbones import DetectionBody
from evaluate import coco_api_eval
from utils.bbox import xywh2xymimx
from utils.coco import CocoUtil
from utils.dataset import tf_distort_image, tf_random_number, get_ds
from utils.image import read_img
from utils.inspect_util import draw_box
from utils.misc import get_dir, get_clean_dir
from prepare.write_tfrecord import parse_example_protobuf
from utils.pascal import TargetMissionUtil

def _loss_wrapper(mean_true):
    def _loss(y_true, y_predict):
        confidence_logits = y_predict[..., 0]
        delta_yx_logits = y_predict[..., 1:3]
        side_distance_logits = y_predict[..., 3:5]
        class_label = y_predict[..., 5:]

        alpha = .25
        gamma = 2.
        epsion = 1e-7

        # focal loss
        confidence_loss = -alpha * tf.pow(1 - tf.sigmoid(confidence_logits), gamma) * tf.math.log(
            epsion + tf.sigmoid(confidence_logits)) * y_true[..., 0] + \
                          -(1 - alpha) * tf.pow(tf.sigmoid(confidence_logits), gamma) * tf.math.log(
            epsion + 1 - tf.sigmoid(confidence_logits)) * (1 - y_true[..., 0])
        confidence_loss = tf.reduce_sum(confidence_loss)

        # confidence_loss = y_true[..., 0] * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 0], logits=confidence_logits) + \
        #                   (1 - y_true[..., 0]) * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 0], logits=confidence_logits)

        object_mask = tf.equal(y_true[..., 0], 1.)
        true_nums = tf.reduce_sum(tf.cast(object_mask, tf.float32))
        non_zero_true_nums = tf.cond(tf.equal(true_nums, 0), lambda: mean_true, lambda: true_nums)

        delta_yx_loss = tf.reduce_sum(tf.boolean_mask(
            tf.compat.v1.losses.huber_loss(y_true[..., 1:3], tf.tanh(delta_yx_logits), delta=0.5, reduction='none'),
            object_mask))

        side_distance_loss = tf.reduce_sum(tf.boolean_mask(
            tf.compat.v1.losses.huber_loss(y_true[..., 3:5], tf.sigmoid(side_distance_logits), delta=0.5, reduction='none'),
            object_mask))

        class_loss = tf.reduce_sum(tf.boolean_mask(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:], logits=class_label),
            object_mask))

        # confidence_loss = tf.math.divide(confidence_loss, non_zero_true_nums)
        # delta_yx_loss = tf.math.divide(delta_yx_loss, non_zero_true_nums)
        # side_distance_loss = tf.math.divide(side_distance_loss, non_zero_true_nums)
        # class_loss = tf.math.divide(class_loss, non_zero_true_nums)

        total_loss = confidence_loss + delta_yx_loss + 4 * side_distance_loss + 0.2 * class_loss

        # todo not work
        tf.compat.v1.summary.scalar("total_loss", total_loss)
        tf.compat.v1.summary.scalar("confidence_loss", confidence_loss)
        tf.compat.v1.summary.scalar("delta_yx_loss", delta_yx_loss)
        tf.compat.v1.summary.scalar("side_distance_oss", side_distance_loss)
        tf.compat.v1.summary.scalar("class_loss", class_loss)

        # total_loss = tf.Print(total_loss, [total_loss, non_zero_true_nums, confidence_loss, delta_yx_loss, side_distance_loss, class_loss], message='loss: ')
        return total_loss

    return _loss


def _image_preprocess(origin_image, input_shape):
    origin_h, origin_w = origin_image.shape[:-1]

    aspect_ratio = origin_h / origin_w

    input_h, input_w = input_shape

    if origin_w < origin_h:
        new_h = input_h
        new_w = new_h / aspect_ratio
    elif origin_w > origin_h:
        new_w = input_w
        new_h = new_w * aspect_ratio
    else:
        new_w = input_w
        new_h = input_h

    new_h = int(new_h)
    new_w = int(new_w)

    # resize image with keep the aspect ratio, then do padding
    image = cv2.resize(origin_image, (new_w, new_h), cv2.INTER_CUBIC)
    image = image.astype(np.float32)
    image = image / 255

    new_image = np.zeros((input_h, input_w, 3), dtype=np.float32)

    if origin_w < origin_h:
        offset_y = 0
        offset_x = int((input_w - new_w) / 2 - 1)
    elif origin_w > origin_h:
        offset_x = 0
        offset_y = int((input_h - new_h) / 2 - 1)
    else:
        offset_x = offset_y = 0
    new_image[offset_y:offset_y + image.shape[0], offset_x: offset_x + image.shape[1]] = image
    return new_image, (offset_x, offset_y), (new_w / origin_w, new_h / origin_h)


def pyfunc_prepare_true(image, gts, labels, height, width, input_shape, num_classes, whether_flip,
                        stride_list, desire_area):
    input_h, input_w = input_shape

    new_image, (offset_x, offset_y), (w_resize_ratio, h_resize_ratio) = _image_preprocess(image, input_shape)

    # fit the gt into new image
    gts[:, [0, 2]] = gts[:, [0, 2]] * w_resize_ratio
    gts[:, [1, 3]] = gts[:, [1, 3]] * h_resize_ratio
    gts[:, 0] = gts[:, 0] + offset_x
    gts[:, 1] = gts[:, 1] + offset_y


    assert len(gts) == len(labels), "gt and labels must have same length"

    # convert from (xmin, ymin, width, height) to (ymin, xmin, ymax, xmax)
    gts = np.stack([gts[:, 1], gts[:, 0], (gts[:, 1] + gts[:, 3]), (gts[:, 0] + gts[:, 2])], axis=1)

    # print(whether_flip)
    if whether_flip:
        gts[:, [1, 3]] = input_w - gts[:, [3, 1]]

    y_true = []

    # choose the desire area for output, etc, the smallest output will predict the largest objects
    areas = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])

    for idx, stride in enumerate(stride_list):
        cur_desire_area = desire_area[idx]
        desired_gt_indice = np.where((areas >= cur_desire_area[0]) & (areas <= cur_desire_area[1]))

        cur_gt = gts[desired_gt_indice]
        cur_label = labels[desired_gt_indice]

        if len(desired_gt_indice) == 0:
            continue

        grid_h, grid_w = input_h // stride, input_w // stride
        cur_output_y_true = np.zeros((grid_h + 1, grid_w + 1, 1 + 2 + 2 + num_classes), dtype=np.float32)
        # ymin, xmin, ymax, xmax
        cur_gt = cur_gt / stride

        cur_gt_yx = np.stack([(cur_gt[:, 0] + cur_gt[:, 2]) / 2, (cur_gt[:, 1] + cur_gt[:, 3]) / 2], axis=1)

        all_kinds = []
        gts_yx_near_top_left = np.floor(cur_gt_yx).astype(int)
        all_kinds.append(gts_yx_near_top_left)
        if stride == 8:
            gts_yx_near_top_right = np.concatenate([np.floor(cur_gt_yx[:, 0:1]), np.ceil(cur_gt_yx[:, 1:2])],
                                               axis=1).astype(int)
            gts_yx_near_bottom_left = np.stack([np.ceil(cur_gt_yx[:, 0]), np.floor(cur_gt_yx[:, 1])], axis=1).astype(int)
            gts_yx_near_bottom_right = np.ceil(cur_gt_yx).astype(int)
            all_kinds.append(gts_yx_near_top_right)
            all_kinds.append(gts_yx_near_bottom_left)
            all_kinds.append(gts_yx_near_bottom_right)



        # grid_y = np.arange(0, grid_h + 1)
        # grid_x = np.arange(0, grid_w + 1)
        # shift_x, shift_y = np.meshgrid(grid_y, grid_x)
        # # (grid_x * grid_y, 2)
        # coordinates = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
        #
        # grid_gaussin = np.exp(np.log(0.5) * 0.5 * np.sum(np.square(coordinates.reshape((-1, 1, 2)) - cur_gt_yx.reshape((1, -1, 2))), axis=2))
        # if grid_gaussin.shape[-1] != 0:
        #     cur_output_y_true[coordinates[..., 0], coordinates[..., 1], 0] = np.clip(
        #         np.max(grid_gaussin, axis=1),
        #         a_min=0., a_max=1.)

        for idx, one_kind in enumerate(all_kinds):
            # print(one_kind, cur_output_y_true.shape)
            # if len(one_kind) > 0:
            #     print(cur_output_y_true.shape, stride)
            #     print(cur_gt_yx, one_kind)

            cur_output_y_true[one_kind[..., 0], one_kind[..., 1], 0] = 1
            cur_output_y_true[one_kind[..., 0], one_kind[..., 1], 1:3] = cur_gt_yx - one_kind
            # left, top, right, bottom
            cur_output_y_true[one_kind[..., 0], one_kind[..., 1], 3:5] = np.stack(
                [(cur_gt_yx[:, 1] - cur_gt[:, 1]) / grid_w,
                 (cur_gt_yx[:, 0] - cur_gt[:, 0]) / grid_h], axis=1)
            cur_output_y_true[one_kind[..., 0], one_kind[..., 1], 5 + cur_label] = 1
        y_true.append(cur_output_y_true[:-1, :-1])

    ret = [new_image, *y_true]
    return ret


def tf_input_preprocess(image, bboxes, labels, height, width, input_shape, num_classes, is_training,
                        strides, desire_area):
    # do data augmentation on image with distort and flip
    if is_training:
        image = tf_distort_image(image)
        whether_flip = tf_random_number() < 0.5
        image = tf.cond(whether_flip, lambda: tf.image.flip_left_right(image), lambda: image)
    else:
        whether_flip = False

    ret_types = [tf.float32] + [tf.float32] * len(strides)

    py_ret = tf.numpy_function(pyfunc_prepare_true, [image, bboxes, labels, height, width,
                                                     input_shape, num_classes, whether_flip,
                                                     strides, desire_area], ret_types)

    return py_ret[0], tuple(py_ret[1:])


def _parse_record(serialized_example, input_shape=None, num_classes=None, is_training=True,
                  strides=None, desire_area=None):
    # extract raw image and annotation from serialized string
    # image tensor: [height, width, channel]
    # bboxes tensor: [N, 4] xmin, ymin, w, h
    # labels tensor: [N, ]
    image, bboxes, labels, height, width, bbox_nums, image_id = parse_example_protobuf(serialized_example,
                                                                                       decode_image=True)
    labels = tf.cast(labels, tf.int32)

    # img, y1, y2, y3 = tf_input_preprocess(image, bboxes, labels, height, width, bbox_nums, input_shape, num_classes=num_classes)
    image, y_true = tf_input_preprocess(image, bboxes, labels, height, width, input_shape, num_classes, is_training,
                                        strides, desire_area)
    image.set_shape((None, None, None))
    for y in y_true: y.set_shape((None, None, None))
    # print(bbox_nums)
    bbox_nums = tf.squeeze(bbox_nums)
    return image, y_true


def tf_eval(outputs, confidence=0.3, strides=None, input_shape=None, from_logits=False):
    input_h, input_w = input_shape

    boxes = []
    scores = []
    labels = []
    boxes_confidence = []
    logging.info("%d output feature map", len(outputs))
    assert len(outputs) == len(strides), 'output nums must match the stride nums'

    for output, stride in zip(outputs, strides):
        if from_logits:
            box_confidence = tf.sigmoid(output[..., 0:1])
            delta_yx = tf.tanh(output[..., 1:3])
            side_distance = tf.sigmoid(output[..., 3:5])
            pred_label = tf.sigmoid(output[..., 5:])
        else:
            box_confidence = output[..., 0:1]
            delta_yx = output[..., 1:3]
            side_distance = output[..., 3:5]
            pred_label = output[..., 5:]

        valid_indice = tf.where(box_confidence >= confidence)

        valid_delta_yx = tf.gather_nd(delta_yx, valid_indice[..., :-1])
        valid_side_distance = tf.gather_nd(side_distance, valid_indice[..., :-1])
        valid_predict_label = tf.gather_nd(pred_label, valid_indice[..., :-1])
        valid_predict_label_sparse = tf.argmax(valid_predict_label, axis=-1)
        valid_predict_label_score = tf.reduce_max(valid_predict_label, axis=-1)

        valid_center_yx = tf.cast(valid_indice[..., 1:3], tf.float32) + valid_delta_yx
        valid_center_side = valid_side_distance * (input_h / stride, input_w / stride)
        # box on predict feature map
        # ymin, xmin, ymax, xmax
        pred_boxes = tf.concat(
            [valid_center_yx - valid_center_side[:, ::-1], valid_center_yx + valid_center_side[:, ::-1]], axis=1)
        pred_boxes *= stride

        # pred_boxes = tf.reshape(pred_boxes, (-1, 4))
        box_scores = tf.reshape(tf.gather_nd(box_confidence, valid_indice[..., :-1]), (-1,))

        boxes.append(tf.stack(
            [tf.clip_by_value(pred_boxes[..., 1], 0, input_w),
             tf.clip_by_value(pred_boxes[..., 0], 0, input_h),
             tf.clip_by_value(pred_boxes[..., 3], 0, input_w),
             tf.clip_by_value(pred_boxes[..., 2], 0, input_h)],
            axis=1))
        boxes_confidence.append(box_scores)
        labels.append(tf.reshape(valid_predict_label_sparse, (-1,)))
        scores.append(tf.reshape(valid_predict_label_score, (-1,)))

    boxes = tf.concat(boxes, axis=0)
    scores = tf.concat(scores, axis=0)

    labels = tf.concat(labels, axis=0)
    labels = tf.reshape(labels, (-1,))
    boxes_confidence = tf.concat(boxes_confidence, axis=0)
    return boxes, boxes_confidence, labels, scores


def _bbox_proprocess_soft(boxes, box_confidence, labels, label_score, image=None):
    all_labels = set(labels)

    _idx = []
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    for cur_label in all_labels:
        cur_class_box_idx = np.where((labels == cur_label))[0]
        # draw_box(image.copy(), boxes[cur_class_box_idx], debug=True, title='before')
        keep_index = []
        while cur_class_box_idx.size > 0:
            cur_class_box_idx_sorted = np.array(
                sorted(cur_class_box_idx, key=lambda x: box_confidence[x], reverse=True))
            # print(box_confidence[cur_class_box_idx_sorted])

            if len(cur_class_box_idx_sorted) == 1:
                keep_index.append(cur_class_box_idx_sorted[0])
                break
            else:
                cur_idx = cur_class_box_idx_sorted[0]
                keep_index.append(cur_idx)

                cur = boxes[cur_idx]
                others = boxes[cur_class_box_idx_sorted[1:]]
                x1 = np.maximum(cur[0], others[:, 0])
                y1 = np.maximum(cur[1], others[:, 1])
                x2 = np.minimum(cur[2], others[:, 2])
                y2 = np.minimum(cur[3], others[:, 3])
                w = np.maximum(0.0, x2 - x1)
                h = np.maximum(0.0, y2 - y1)
                inter = w * h
                # 交/并得到iou值
                ovr = inter / (areas[cur_idx] + areas[cur_class_box_idx_sorted[1:]] - inter)
                out_thresh_idx = np.where(ovr <= 0.5)[0]

                # ti = image.copy()
                # print(ovr)
                # print(out_thresh_idx)
                # draw_box(ti, boxes[cur_idx], color=(0, 255, 0), title='dfs')
                # draw_box(ti, boxes[cur_class_box_idx_sorted[1:]], color=(255, 0, 255), debug=True, title='campare')

                cur_class_box_idx = cur_class_box_idx_sorted[out_thresh_idx + 1]

        _idx.extend(keep_index)
        # draw_box(image.copy(), boxes[keep_index], debug=True, title='after')

    # draw_box(image.copy(), boxes[_idx], title="final", debug=True)
    return boxes[_idx], box_confidence[_idx], labels[_idx], label_score[_idx]


class RcNet(object):
    # params
    num_classes = 80
    epochs = 50

    log_dir = 'log'
    ckpt_dir = 'checkpoint'

    wheather_freeze_backbone = False

    def __init__(self, training=True, data_dir=None, ckpt_path=None,
                 center_confidence=None, batch_size=None, gpu_nums=None,
                 detection_body: DetectionBody = None,
                 tu: TargetMissionUtil = None):

        logging.info("init rcnet under the configuration of %s task and %s detection backbone with %d num_classes" % (
            tu.name, detection_body.name, tu.num_classes
        ))

        self.lr = 1e-3
        self.start_epoch = 0

        self.training = training
        self.batch_size = batch_size
        self.gpu_nums = gpu_nums

        self.detection_body = detection_body
        self.task_util = tu
        self.backbone_name = detection_body.name
        self.input_shape = detection_body.input_shape
        self.output_channel = detection_body.output_channel
        self.strides = detection_body.strides
        self.desire_area = detection_body.desire_area
        self.pretrain_weights = detection_body.pretrain_path
        self.num_freez_backbone_layer = detection_body.num_freez_backbone_layer

        self.parse_args = dict(input_shape=self.input_shape, num_classes=self.num_classes, strides=self.strides,
                               desire_area=self.desire_area)

        if training:  # train
            assert data_dir is not None, "specify data dir for training"
            self.train_file_dir = os.path.join(data_dir, 'train*')
            self.val_file_dir = os.path.join(data_dir, 'val*')

            self.log_dir = get_dir(self.log_dir, "%s_%s" % (self.backbone_name, self.task_util.name))
            self.ckpt_dir = get_dir(self.ckpt_dir, "%s_%s" % (self.backbone_name, self.task_util.name))

            # output_map = {
            #     'output5': _loss_wrapper(3.08 * self.batch_size),
            #     'output1': _loss_wrapper(2.56 * self.batch_size),
            #     'output2': _loss_wrapper(1.31 * self.batch_size),
            #     'output3': _loss_wrapper(1.20 * self.batch_size),
            #     'output4': _loss_wrapper(1.08 * self.batch_size),
            # }

            output_map = {
                # mean true is dummy
                'output%d' % (x + 1): _loss_wrapper(1. * self.batch_size)
                for x in range(len(self.strides))
            }


            if self.wheather_freeze_backbone:
                logging.info("freeze backbone params")

            """ important """
            if self.gpu_nums > 1:
                logging.info("multi gpu training")

                # devices=["/device:GPU:0", "/device:GPU:1"]
                mirrored_strategy = tf.distribute.MirroredStrategy()
                with mirrored_strategy.scope():
                    self._model = self.detection_body.get_detection_body(output_layers_nums=5)
                    if self.wheather_freeze_backbone:
                        for i in range(self.num_freez_backbone_layer): self._model.layers[i].trainable = False
                    self._model.compile(optimizer=Adam(lr=self.lr), loss=output_map)
                    self._model_restore()
            else:
                logging.info("single gpu")
                self._model = self.detection_body.get_detection_body(output_layers_nums=5)
                if self.wheather_freeze_backbone:
                    for i in range(self.num_freez_backbone_layer): self._model.layers[i].trainable = False

                self._model_restore()
                self._model.compile(optimizer=Adam(lr=self.lr), loss=output_map)

        else:  # test
            self._model = detection_body.get_detection_body(output_layers_nums=5)
            if ckpt_path is None or not os.path.exists(ckpt_path):
                raise ValueError("ckeck ckpt file")

            if center_confidence is not None:
                self.center_confidence = center_confidence
            boxes, box_confidence, object_labels, object_scores = tf_eval(self._model.output, strides=self.strides,
                                                                          input_shape=self.input_shape,
                                                                          from_logits=True,
                                                                          confidence=self.center_confidence)
            logging.info("load ckpt from %s " % ckpt_path)
            self._model.load_weights(ckpt_path)

            self.input_image = self._model.input
            self._box = boxes
            self._box_confidence = box_confidence
            self._scores = object_scores
            self._labels = object_labels
        logging.info("detection backbone with %d output feature map" % len(self.strides))

    def _model_restore(self):
        if len(os.listdir(self.ckpt_dir)) >= 1:
            restore_epoch, last_ckpt = self._restore_epoch()
            self._model.load_weights(last_ckpt)
            logging.info('restore from exist model checkpoint')
        else:
            # using the parameter pretrained on ImageNet
            logging.info('find no exist model checkpoint, start from pretrained params')
            self._model.load_weights(self.pretrain_weights, by_name=True)
            logging.info('load pretrain weights')
            restore_epoch = 0

        self.start_epoch = restore_epoch

    def _restore_epoch(self):
        ckpts = os.listdir(self.ckpt_dir)

        last_ckpt = sorted(ckpts, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))[-1]

        name = os.path.splitext(os.path.basename(last_ckpt))[0]
        logging.info(name)
        last_epoch = int(name.split('_')[-1])
        return last_epoch, os.path.join(self.ckpt_dir, last_ckpt)

    def input_test(self):
        train_ds, train_example_nums = get_ds(self.train_file_dir, batch_size=1,
                                              parse_fn=_parse_record, parse_record_args=self.parse_args,
                                              trainval_split=False)

        X, Y = train_ds.make_one_shot_iterator().get_next()

        print(X)
        print(Y)
        config = tf.ConfigProto(
            device_count={'GPU': 1}
        )

        box, box_score, label, label_score = tf_eval(Y, input_shape=self.input_shape, strides=self.strides,
                                                     confidence=0.6)


        with tf.Session(config=config) as sess:
            for _ in range(40):
                x, b, bc, l, ls, y = sess.run([X, box, box_score, label, label_score, Y[0]])
                # _box, _box_confidence, _label, _score = _bbox_proprocess_soft(b, bc, l, ls)
                print(len(b))
                print(b)
                # print(np.where(y == 1))
                show_tags = list(zip(bc, l))
                draw_box(x[0], b, debug=True, title='input_test', text=show_tags)

    def train(self):
        tb = TensorBoard(log_dir=self.log_dir, write_graph=True,
                         update_freq='epoch', histogram_freq=20)

        def scheduler(epoch):
            if epoch < 10:
                return self.lr
            elif 10 <= epoch < 15:
                return self.lr * np.exp(-0.3)
            elif 15 <= epoch < 20:
                return self.lr * np.exp(-0.4)
            elif 20 <= epoch < 35:
                return self.lr * np.exp(-0.45)
            else:
                return self.lr * np.exp(-0.4)
                # return self.lr * np.exp(0.4 * (10 - epoch))

        lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

        ckpt = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.ckpt_dir, 'rcnet-%s-%d*%d-l{loss:.1f}-vl{val_loss:.1f}_{epoch:03d}.h5' %
                                  (self.backbone_name, self.input_shape[0], self.input_shape[1])),
            monitor='val_loss',
            verbose=1,
            load_weights_on_restart=False,
            save_best_only=False,
            save_weights_only=True)
        csv_logger = CSVLogger('training.log')

        # get tf.data.Dataset
        train_ds, train_example_nums = get_ds(self.train_file_dir, batch_size=self.batch_size,
                                              epochs=self.epochs,
                                              parse_fn=_parse_record, parse_record_args=self.parse_args,
                                              trainval_split=False)
        val_ds, val_example_nums = get_ds(self.val_file_dir, batch_size=self.batch_size, trainval_split=False,
                                          parse_fn=_parse_record, epochs=self.epochs,
                                          parse_record_args=dict(is_training=False, **self.parse_args))

        logging.info("start from epoch %d, using gpu nums %d, batch size %d, initial lr %f, input_shape %s"
                     % (self.start_epoch + 1, self.gpu_nums, self.batch_size, self.lr, self.input_shape))

        self._model.fit(train_ds, validation_data=val_ds, initial_epoch=self.start_epoch,
                        callbacks=[tb, lr_schedule, ckpt, csv_logger], epochs=self.epochs,
                        steps_per_epoch=train_example_nums // self.batch_size, verbose=1,
                        validation_steps=val_example_nums // self.batch_size)
        self._model.save_weights(os.path.join(self.ckpt_dir, 'rcnet-%s-%d*%d_final.h5' %
                                              (self.backbone_name, self.input_shape[0], self.input_shape[1])))

    def predict(self, image, specific=True, sess: tf.compat.v1.Session = None):
        if sess is None:
            sess = get_session()
        if isinstance(image, str):
            origin_image = read_img(image)
        else:
            origin_image = image

        origin_h, origin_w = origin_image.shape[:-1]
        # resize image and return the image with model input size and offset and scale infomation
        input_image, offset_xy, scale_xy = _image_preprocess(origin_image, input_shape=self.input_shape)
        input_image = np.expand_dims(input_image, 0)

        # box [[xmin, ymin, xmax, ymax]...]
        # box scores [s1, s2, s3...] 0 <= s_i <= 1
        # labels etc [c1, c2, c3 ...] 0 <= c_i <= C
        # score etc  [t1, t2, t3 ...] 0 <= t_i <= 1
        b, bc, l, ls = sess.run(
            [self._box, self._box_confidence, self._labels, self._scores], feed_dict={
                self.input_image: input_image
            })

        b[:, [0, 2]] = b[:, [0, 2]] - offset_xy[0]
        b[:, [1, 3]] = b[:, [1, 3]] - offset_xy[1]

        b[:, [0, 2]] = b[:, [0, 2]] / scale_xy[0]
        b[:, [1, 3]] = b[:, [1, 3]] / scale_xy[1]

        b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, origin_w - 1)
        b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, origin_h - 1)
        # nms
        _box, _box_confidence, _label, _score = _bbox_proprocess_soft(b, bc, l, ls, image=origin_image)

        if specific:
            return _box, _box_confidence, _label, _score
        else:
            return _box, _label, _score

    def eval(self):
        sess = get_session()
        result_json_path = 'result/coco_result.json'
        result_dir = get_clean_dir('result', self.task_util.name)

        img_path = self.task_util.get_eval_imgs()
        predictions = []
        for i in tqdm(img_path[:]):
            predict_box, predict_label, predict_score = self.predict(i, sess=sess, specific=False)
            image_id = os.path.splitext(os.path.basename(i))[0].lstrip('0')
            predict_human_label = [self.task_util.label_classes[x] for x in predict_label]

            with open(os.path.join(result_dir, "%s.txt") % image_id, 'a') as f:
                if len(predict_box) == 0:
                    f.write("\n")

                for n, b, l, s in zip(predict_human_label, predict_box, predict_label, predict_score):
                    if 'coco' in self.task_util.name:
                        predictions.append({
                            'image_id': int(image_id),
                            'category_id': int(CocoUtil.label_to_category_id(l)),
                            'bbox': list(map(lambda x: float(x), [b[0], b[1], b[2] - b[0], b[3] - b[1]])),
                            'score': float(s)
                        })

                        f.write("%s %f %s %s %s %s\n" % (n, s, b[0], b[1], b[2], b[3]))

        if 'coco' in self.task_util.name:
            with open(result_json_path, 'w') as f:
                json.dump(predictions, f)
            coco_api_eval(self.task_util.coco_anno_dir, result_json_path)

    def eval_on_test_dev(self, imgs):
        sess = get_session()
        result_json_path = 'result/detections_test-dev2017_rcnet.json'

        predictions = []
        for i in tqdm(imgs):
            predict_box, predict_label, predict_score = self.predict(i, sess=sess, specific=False)
            image_id = os.path.splitext(os.path.basename(i))[0].lstrip('0')
            predict_human_label = [self.task_util.label_classes[x] for x in predict_label]

            for n, b, l, s in zip(predict_human_label, predict_box, predict_label, predict_score):
                if 'coco' in self.task_util.name:
                    predictions.append({
                        'image_id': int(image_id),
                        'category_id': int(CocoUtil.label_to_category_id(l)),
                        'bbox': list(map(lambda x: float(x), [b[0], b[1], b[2] - b[0], b[3] - b[1]])),
                        'score': float(s)
                    })

        with open(result_json_path, 'w') as f:
            json.dump(predictions, f)

    def interactive_inspect(self, show_anno=False, imgs=None):
        if imgs is None:
            img_path = self.task_util.get_eval_imgs()
        else:
            img_path = imgs

        for p in img_path:
            print(p)
            image = read_img(p)
            image_id = os.path.splitext(os.path.basename(p))[0].lstrip('0')

            anno_box, anno_human_label = self.task_util.get_image_anno(image_id)

            predict_box, predict_box_confidence, predict_label, predict_score = self.predict(image)
            predict_human_label = [self.task_util.label_classes[x] for x in predict_label]

            show_tags = list(zip(predict_human_label, ["%.2f %.2f" % (box_conf, label_score)
                                                       for box_conf, label_score in
                                                       zip(predict_box_confidence, predict_score)]))

            print('predict labels: %s' % str(show_tags))
            print('annotate label: %s' % anno_human_label)
            if show_anno and len(anno_box) != 0:
                draw_box(image.copy(), np.array(anno_box), debug=True, title='anno', text=anno_human_label)
            if len(predict_box) != 0:
                draw_box(image.copy(), predict_box, debug=True, title='predict', text=show_tags)
