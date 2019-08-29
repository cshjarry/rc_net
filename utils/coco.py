import glob
import os

from pycocotools.coco import COCO

from utils.pascal import TargetMissionUtil


class CocoUtil(TargetMissionUtil):
    name = "coco"
    num_classes = 80


    def __init__(self, coco_dir, coco_label_classes_path):
        self.coco_dir = coco_dir
        self.eval_dir = self.coco_eval_dir = os.path.join(self.coco_dir, 'val2017')
        self.coco_anno_dir = os.path.join(coco_dir, "annotations/instances_val2017.json")
        self.coco_train_anno_dir = os.path.join(coco_dir, "annotations/instances_train2017.json")
        self.coco = COCO(self.coco_anno_dir)
        self.train_coco = COCO(self.coco_train_anno_dir)
        self.label_classes = self.get_label_classes(coco_label_classes_path)

    def get_eval_imgs(self):
        return glob.glob(os.path.join(self.eval_dir, '*.jpg'))

    def label_to_human_label(self, label):
        return self.label_classes[label]

    @staticmethod
    def category_id_to_label(cat):
        if cat >= 1 and cat <= 11:
            cat = cat - 1
        elif cat >= 13 and cat <= 25:
            cat = cat - 2
        elif cat >= 27 and cat <= 28:
            cat = cat - 3
        elif cat >= 31 and cat <= 44:
            cat = cat - 5
        elif cat >= 46 and cat <= 65:
            cat = cat - 6
        elif cat == 67:
            cat = cat - 7
        elif cat == 70:
            cat = cat - 9
        elif cat >= 72 and cat <= 82:
            cat = cat - 10
        elif cat >= 84 and cat <= 90:
            cat = cat - 11
        return cat

    @staticmethod
    def label_to_category_id(cat):
        if 0 <= cat <= 10:
            cat = cat + 1
        elif 11 <= cat <= 23:
            cat = cat + 2
        elif 24 <= cat <= 25:
            cat = cat + 3
        elif 26 <= cat <= 39:
            cat = cat + 5
        elif 40 <= cat <= 59:
            cat = cat + 6
        elif cat == 60:
            cat = cat + 7
        elif cat == 61:
            cat = cat + 9
        elif 62 <= cat <= 72:
            cat = cat + 10
        elif 73 <= cat <= 79:
            cat = cat + 11
        return cat

    def get_label_classes(self, path):
        with open(path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_image_anno(self, image_id, for_train=False):
        if isinstance(image_id, list):
            image_ids = [int(x) for x in image_id]
        else:
            image_ids = [int(image_id)]

        if not for_train:
            anno_ids = self.coco.getAnnIds(image_ids)
            annos = self.coco.loadAnns(anno_ids)

            anno_box = [x['bbox'] for x in annos]
            anno_box = [[x[0], x[1], x[0] + x[2], x[1] + x[3]] for x in anno_box]
            anno_labels = [x['name'] for x in self.coco.loadCats([x['category_id'] for x in annos])]
            return anno_box, anno_labels
        else:
            anno_ids = self.train_coco.getAnnIds(image_ids)
            annos = self.train_coco.loadAnns(anno_ids)

            anno_box = [x['bbox'] for x in annos]
            # anno_box = [[x[0], x[1], x[0] + x[2], x[1] + x[3]] for x in anno_box]
            anno_labels = [x['name'] for x in self.train_coco.loadCats([x['category_id'] for x in annos])]
            return anno_box, anno_labels



