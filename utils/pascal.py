import os

import xml.etree.ElementTree as ET

from utils.misc import get_dir


class TargetMissionUtil(object):
    name = ""
    num_classes = -1
    eval_dir = ""
    label_classes = []

    def __init__(self):
        pass

    def get_eval_imgs(self):
        pass

    def human_label_to_label(self, human_label):
        pass

    def label_to_human_label(self, label):
        pass

    def get_image_anno(self, image_id: str):
        pass


class PascalVOClUtil(TargetMissionUtil):
    name = "pascal"
    label_classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    num_classes = len(label_classes)


    def __init__(self, voc_dir=None):
        self.voc_dir = voc_dir


    def get_eval_imgs(self):
        image_ids = open('%s/ImageSets/Main/val.txt' % self.voc_dir).read().strip().split()
        image_path = ["%s/JPEGImages/%s.jpg" % (self.voc_dir, str(image_id)) for image_id in image_ids]
        return image_path

    def get_image_anno(self, image_id: str):
        anno_xml_path = os.path.join('%s/Annotations/%s.xml' % (self.voc_dir, image_id))
        return self.parse_xml_annotation(anno_xml_path)

    def label_to_human_label(self, label: int):
        return self.label_classes[label]

    @staticmethod
    def parse_xml_annotation(path):
        root = ET.parse(path).getroot()

        anno_boxes = []; anno_labels = []
        for obj in root.findall('object'):
            obj_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            anno_boxes.append([xmin, ymin, xmax, ymax])
            anno_labels.append(obj_name)
        return anno_boxes, anno_labels

    @staticmethod
    def convert_eval_annotations():
        import argparse
        parser = argparse.ArgumentParser()

        parser.add_argument('-vd', '--voc_dir', default='/media/csh/data/VOCdevkit/VOC2012')
        parser.add_argument('-d', '--dest_dir', default='result/pascal_voc_gt')

        args = parser.parse_args()
        dest_dir = get_dir(args.dest_dir)

        image_ids = open('%s/ImageSets/Main/val.txt' % args.voc_dir).read().strip().split()
        print(len(image_ids))

        for image_id in image_ids:
            # print(tmp_file)
            # 1. create new file (VOC format)
            with open(os.path.join(dest_dir, "%s.txt" % image_id), "a") as new_f:
                root = ET.parse(os.path.join('%s/Annotations/%s.xml' % (args.voc_dir, image_id))).getroot()
                for obj in root.findall('object'):
                    obj_name = obj.find('name').text
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))


