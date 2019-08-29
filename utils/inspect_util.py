import json
import os

import cv2
import numpy as np
from PIL import Image

from utils.bbox import xywh2xymimx
from utils.image import read_img, show_img


def draw_text(img, coord, text, color=(255, 0, 0)):
    font = cv2.FONT_ITALIC
    fontScale = 0.3
    lineType = 1
    coord[1] -= 5
    img = cv2.putText(img,str(text), tuple(coord),font,fontScale, tuple(color), lineType)
    # cv2.putText(img, str(text), coord, cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

    return img


def draw_box(img, bbox, random_color=True, color=(255, 0, 0), debug=False, text=None, title='draw_box'):
    """
    draw bbox on image
    :param img: np.ndarray
    :param bbox: xmin, ymin, xmax, ymax
    :return: image with box
    """
    bbox = bbox.astype(int)
    if bbox.ndim == 1:
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    elif bbox.ndim == 2:
        for idx,(xmin, ymin, xmax, ymax) in enumerate(bbox):
            inner_color = (np.random.random((3,)) * 255).astype(int).tolist() if random_color else color
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), inner_color, 1)
            if text is not None:
                draw_text(img,[xmin, ymin], str(text[idx]), color=inner_color)

    if debug:
        show_img(img, title=title)
    return img

def anno_inspect():
    with open('/media/csh/data/coco_2017/annotations/instances_val2017.json') as f:
        res = json.load(f)
    res = res['annotations']

    coco_anno = "/media/csh/data/coco_2017/val2017"
    for anno in res[:10]:
        image_id = anno['image_id']
        img_path = os.path.join(coco_anno, "%012d.jpg" % image_id)
        bbox = np.array(list(map(lambda x:int(x), anno['bbox'])))
        bbox[..., 2] = bbox[..., 0] + bbox[..., 2]
        bbox[..., 3] = bbox[..., 1] + bbox[..., 3]
        img = read_img(img_path)
        print(image_id, bbox)
        draw_box(img, bbox, debug=True)

if __name__ == '__main__':
    image_dir = "/media/csh/data/coco_2017/val2017"
    anno_dir = "/media/csh/data/coco_2017/annotations/instances_val2017.json"

    with open(anno_dir) as f:
        coco_anno = json.load(f)

    for anno in coco_anno['annotations']:
        image_path = os.path.join(image_dir, "%012d" % anno['image_id']) + '.jpg'
        bbox = anno['bbox']
        break
    img = read_img(image_path)
    bbox = xywh2xymimx(bbox)
    draw_box(img, bbox, debug=True)


    # test_img = ["/media/csh/data/coco_2017/val2017/000000477118.jpg",
    #             "/media/csh/data/coco_2017/val2017/000000006818.jpg",
    #             "/media/csh/data/coco_2017/val2017/000000004765.jpg"]
    # for t_img in test_img:
    #     img = read_img(t_img)
    #     letterbox_resize(img, (416, 416), show=True)
