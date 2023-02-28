import os
import matplotlib.pyplot as plt
import cv2
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result, process_mmdet_results
from mmdet.apis import inference_detector, init_detector
from pycocotools.coco import COCO


def init_path():
    os.chdir('F:\\baidu\\mmpose')
    os.listdir()


# 定义可视化图像函数，输入图像路径，可视化图像
def show_img_from_path(img_path):
    # opencv 读入图像，matplotlib 可视化格式为 RGB，因此需将 BGR 转 RGB，最后可视化出来
    img = cv2.imread(img_path)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()


# 定义可视化图像函数，输入图像 array，可视化图像
def show_img_from_array(img):
    # 输入 array，matplotlib 可视化格式为 RGB，因此需将 BGR 转 RGB，最后可视化出来
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()


def vis(path):
    coco = COCO(path)
    for image in coco.imgs.keys():
        file_name = image['file_name']
        image_id = image['id']
        annotation_ids = coco.loadImgs([image_id])
        annotation = coco.loadAnns([image_id])
        coco.showAnns(annotation)


if __name__ == '__main__':
    # init os path
    init_path()

    # set img path
    img_path = 'data/test1.jpg'

    # set path for models
    # 目标检测模型
    det_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
    det_checkpoint = 'F:\\baidu\\checkpoints\\faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    # 人体姿态估计模型
    pose_config = 'F:/baidu/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py'
    pose_checkpoint = 'F:/baidu/mmpose/work_dirs/res50_coco_wholebody_256x192/latest.pth'

    # 目标检测模型
    det_model = init_detector(det_config, det_checkpoint)

    # 人体姿态估计模型
    pose_model = init_pose_model(pose_config, pose_checkpoint)

    # detect image using det_model, the 0 index of result is pedestrian
    mmdet_results = inference_detector(det_model, img_path)

    # get bbox for met result
    person_results = process_mmdet_results(mmdet_results, cat_id=1)
    # print(person_results)

    # get pose result from bbox
    pose_results, returned_outputs = inference_top_down_pose_model(pose_model, img_path, person_results, bbox_thr=0.3,
                                                                   format='xyxy', dataset='TopDownCocoDataset')

    # visualize result
    vis_result = vis_pose_result(pose_model,
                                 img_path,
                                 pose_results,
                                 radius=2,
                                 thickness=1,
                                 dataset='TopDownCocoDataset',
                                 show=False)
    show_img_from_array(vis_result)

    # data_root = 'F:/baidu/mmpose/data/coco_whole_body'
    # ann_file = f'{data_root}/whole_body_new_val.json'
    # vis(ann_file)

    # save result
    cv2.imwrite('outputs/test5.jpg', vis_result)

#       拍视频，每个状态拍一个视频，然后切帧。opencv
#       根据关键点生成一个新的训练集。
#       处理关键点，提取特征：角度（），向量（模长，身高归一化，角度，以人体中轴线为基准长度归一化处理，坐标变换）95%以上 仿射变换
#
