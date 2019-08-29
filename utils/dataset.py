import os

import tensorflow as tf
import random

from absl import logging
from tensorflow.python.data import Dataset

_DTYPE = tf.float32
# ******************************************** data augmentation *****************************************************
def tf_distort_image(im, hue=.1, sat=1.5, val=1.5):
    # im = tf.image.random_saturation(im, 1, sat) if tf_random_number() < 0.5 else tf.image.random_saturation(im, 1/sat, 1)
    im = tf.cond(tf_random_number() < 0.5, lambda :tf.image.random_saturation(im, 1, sat), lambda :tf.image.random_saturation(im, 1/sat, 1) )
    im = tf.image.random_hue(im, hue)
    return im


# tf preprocess workflow, which means everything in this process will be a graph node
def tf_random_number(a=0., b=1.):
    return tf.random.uniform([], dtype=_DTYPE) * (b - a) + a
# ******************************************** data augmentation *****************************************************


def data_gen(filenames,epochs=None, batch_size=None, parse_record_fn=None, parse_args=None) -> Dataset:
    dataset = Dataset.from_tensor_slices(filenames)

    dataset = dataset.shuffle(buffer_size=64)

    # shuffle by interleave across the tfrecord files
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x).map(
            lambda val: parse_record_fn(val, **parse_args), num_parallel_calls=batch_size),
        cycle_length=4, block_length=16, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if epochs is not None:
        dataset = dataset.repeat(epochs + 1)
    else:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size=batch_size)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def get_example_nums(filenames):
    nums = 0
    for file in filenames:
        nums += int(os.path.basename(file).split('_')[1])
    return nums


def validate_tfrecord_file(filepaths):
    total = 0
    for f in filepaths:
        record_iterator = tf.io.tf_record_iterator(path=f)
        total += sum([1 for _ in record_iterator])
    return total


def get_ds(files, split_ratio=None, batch_size=None, trainval_split=True, parse_fn=None, parse_record_args=None, epochs=None):
    if isinstance(files, str):
        filenames = tf.io.gfile.glob(files)
    else:
        filenames = files

    random.shuffle(filenames)

    if trainval_split:
        split_point = int(split_ratio * len(filenames))
        train_files = filenames[:split_point]
        val_files = filenames[split_point:]

        train_example_nums = validate_tfrecord_file(train_files)
        val_example_nums = validate_tfrecord_file(val_files)

        logging.info("find %d train files: %d for train, %d for validate; train example %d, val example %d" %
              (len(filenames), split_point, len(filenames) - split_point, train_example_nums, val_example_nums))
        train_ds = data_gen(train_files, batch_size=batch_size, parse_record_fn=parse_fn, parse_args=parse_record_args, epochs=epochs)
        val_ds = data_gen(val_files, batch_size=batch_size, parse_record_fn=parse_fn, parse_args=parse_record_args, epochs=epochs)
        return train_ds, val_ds, train_example_nums, val_example_nums
    else:
        train_example_nums = get_example_nums(filenames)
        logging.info("find %d files and %d examples, all for train" % (len(filenames), train_example_nums))
        ds = data_gen(filenames, batch_size=batch_size, parse_record_fn=parse_fn, parse_args=parse_record_args, epochs=epochs)
        return ds, train_example_nums
