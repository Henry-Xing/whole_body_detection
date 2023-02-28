import json
from tqdm import trange


# 存储新建的json文件，输入为新建的标注字典和存储路径
def save_json(anns_new, path):
    with open(path, 'w') as f:
        json.dump(anns_new, f)


# 只获取annotation，以便内存可以不出问题
def cap_anns(anns, path):
    anns_new = {
        'annotations': anns["annotations"],
    }

    save_json(anns_new, path)
    return


# 构建新的annotation词条，输入为一条whole-body和coco的annotation
def get_new_ann(whole, normal):
    extra_kpts = []
    EXTRA_NUM = 18

    if whole['foot_valid']:
        extra_kpts.extend(whole['foot_kpts'])
    else:
        extra_kpts.extend([0] * EXTRA_NUM)

    if whole['righthand_valid']:
        for i in range(0, len(whole['righthand_kpts']), 12):
            for j in range(3):
                extra_kpts.append(whole['righthand_kpts'][i + j])
    else:
        extra_kpts.extend([0] * EXTRA_NUM)

    if whole['lefthand_valid']:
        for i in range(0, len(whole['lefthand_kpts']), 12):
            for j in range(3):
                extra_kpts.append(whole['lefthand_kpts'][i + j])
    else:
        extra_kpts.extend([0] * EXTRA_NUM)

    extra_nums = (len(extra_kpts) - extra_kpts.count(0)) // 3

    new = {
        'segmentation': normal['segmentation'],
        'num_keypoints': normal['num_keypoints'] + extra_nums,
        'area': normal['area'],
        'iscrowd': normal['iscrowd'],
        'keypoints': normal['keypoints'] + extra_kpts,
        'image_id': normal['image_id'],
        'bbox': normal['bbox'],
        'category_id': normal['category_id'],
        'id': normal['id']
    }
    return new


# 构建新的whole-body coco数据集，输入为whole-body的annotation和原版coco的全部标注
def create_whole_body_ann(whole_body_anns, coco_ann):
    new_category = coco_ann["categories"]
    new_kpts = ['right_foot_thumb', 'right_foot_little', 'right_heel', 'left_foot_thumb', 'left_foot_little',
                'left_heel', 'right_palm', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little',
                'left_palm', 'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_little']
    new_category[0]['keypoints'] += new_kpts
    new_skeleton = [[17, 20], [20, 18], [20, 19], [18, 19], [16, 23], [23, 21], [23, 22], [21, 22], [11, 24], [24, 25],
                    [24, 26], [24, 27], [24, 28], [24, 29], [10, 30], [30, 31], [30, 32], [30, 33], [30, 34], [30, 35]]
    new_category[0]['skeleton'] += new_skeleton

    anns_new = {
        'info': coco_ann["info"],
        'licenses': coco_ann["licenses"],
        'images': coco_ann["images"],
        'annotations': [],
        'categories': new_category
    }

    for i in trange(len(coco_ann["annotations"])):
        whole = whole_body_anns["annotations"][i]
        normal = coco_ann["annotations"][i]
        anns_new["annotations"].append(get_new_ann(whole, normal))

    return anns_new


if __name__ == '__main__':
    # 载入训练集标注文件
    dataDir = 'F:/baidu/mmpose/data/'

    # 原始标注
    # train
    coco_whole_body_train = 'coco_whole_body/coco_wholebody_train_v1.0'
    coco_body_train = 'coco/person_keypoints_train2017'

    # val
    coco_body_val = 'coco/person_keypoints_val2017'
    coco_whole_body_val = 'coco_whole_body/coco_wholebody_val_v1.0'

    # 只有annotation的whole-body标注, 只有train的
    whole_body_anns_train = 'coco_whole_body/whole_body_ann_only_train'

    # load train
    coco_whole_body_train_path = '{}/{}.json'.format(dataDir, coco_whole_body_train)
    coco_body_train_path = '{}/{}.json'.format(dataDir, coco_body_train)
    whole_body_anns_train_path = '{}/{}.json'.format(dataDir, whole_body_anns_train)
    # coco_whole_body_anns_train = json.load(open(coco_whole_body_train_path, "r"))
    coco_body_anns_train = json.load(open(coco_body_train_path, "r"))
    whole_body_anns_only_train = json.load(open(whole_body_anns_train_path, "r"))

    # load val
    coco_whole_body_val_path = '{}/{}.json'.format(dataDir, coco_whole_body_val)
    coco_body_val_path = '{}/{}.json'.format(dataDir, coco_body_val)
    coco_whole_body_anns_val = json.load(open(coco_whole_body_val_path, "r"))
    coco_body_anns_val = json.load(open(coco_body_val_path, "r"))

    # 获取单独的annotations，只有train的里面需要
    # only_anns_save_train_path = 'coco_whole_body/whole_body_ann_only_train'
    # oas_train_path = '{}/{}.json'.format(dataDir, only_anns_save_train_path)
    # cap_anns(coco_whole_body_anns_train, oas_path)

    # get new train annotations
    anns_new_train = create_whole_body_ann(whole_body_anns_only_train, coco_body_anns_train)
    anns_new_train_path = '{}/{}.json'.format(dataDir, 'coco_whole_body/whole_body_new_train')
    save_json(anns_new_train, anns_new_train_path)

    # get new val annotations
    anns_new_val = create_whole_body_ann(coco_whole_body_anns_val, coco_body_anns_val)
    anns_new_val_path = '{}/{}.json'.format(dataDir, 'coco_whole_body/whole_body_new_val')
    save_json(anns_new_val, anns_new_val_path)
