import json


def get_coco_anno():
    anno_path = "/media/csh/data/coco_2017/annotations/instances_val2017.json"
    with open(anno_path) as f:
        coco_annotations = json.load(f)
    img_box = {}
    categories = set()
    for anno in coco_annotations['annotations']:
        image_id = str(anno['image_id'])
        if image_id in img_box:
            img_box[image_id]['bbox'].append(anno['bbox'])
            img_box[image_id]['label'].append(anno['category_id'])
            img_box[image_id]['iscrowd'].append(anno['iscrowd'])
        else:
            img_box[image_id] = dict(bbox=[anno['bbox']], label=[anno['category_id']], iscrowd=[anno['iscrowd']])
        categories.add(anno['category_id'])
    # print(coco_annotations['annotations'][0])
    # print(categories)
    return img_box


def coco_api_eval(anno_path, result_path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    cocoGt = COCO(anno_path)

    cocoDt = cocoGt.loadRes(result_path)

    imgIds = cocoGt.getImgIds()

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def main():
    annFile = "/media/csh/data/coco_2017/annotations/instances_val2017.json"
    resFile = "results/fcos_result.json"
    coco_api_eval(annFile, resFile)


if __name__ == '__main__':
    main()