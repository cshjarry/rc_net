import numpy as np

def bbox_proprocess(bboxs, origin_shape):
    #bbox: top left bottom right
    #origin_shape: height, width
    # convert coordinate to xmin ymin xmax ymax (left top bottom right)
    new_bbox = np.empty(bboxs.shape, dtype=np.int)
    new_bbox[..., 1] = np.maximum(0, np.floor(bboxs[..., 0] + 0.5).astype('int32')) # top
    new_bbox[..., 0] = np.maximum(0, np.floor(bboxs[..., 1] + 0.5).astype('int32')) # left
    new_bbox[..., 3] = np.minimum(origin_shape[0], np.floor(bboxs[..., 2] + 0.5).astype('int32')) # bottom
    new_bbox[..., 2] = np.minimum(origin_shape[1], np.floor(bboxs[..., 3] + 0.5).astype('int32')) # right
    return new_bbox

def xywh2xymimx(bbox):
    new_bbox = np.empty((bbox.shape[0], 4), dtype=np.float32)
    # xmin, ymin, xmax, ymax
    new_bbox[..., 0] = bbox[...,0]
    new_bbox[...,1] = bbox[...,1]
    new_bbox[...,2] = bbox[...,0] + bbox[...,2]
    new_bbox[...,3] = bbox[...,1] + bbox[...,3]
    return new_bbox

def bbox_xyxy_to_xywh(xyxy):
    """Convert bounding boxes from format (xmin, ymin, xmax, ymax) to (x, y, w, h).
    Parameters
    ----------
    xyxy : list, tuple or numpy.ndarray
        The bbox in format (xmin, ymin, xmax, ymax).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.
    Returns
    -------
    tuple or numpy.ndarray
        The converted bboxes in format (x, y, w, h).
        If input is numpy.ndarray, return is numpy.ndarray correspondingly.
    """
    if isinstance(xyxy, (tuple, list)):
        if not len(xyxy) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xyxy)))
        x1, y1 = xyxy[0], xyxy[1]
        w, h = xyxy[2] - x1 + 1, xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        if not xyxy.size % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xyxy.shape))
        return np.hstack((xyxy[:, :2], xyxy[:, 2:4] - xyxy[:, :2] + 1))
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xyxy)))

