from mmcv import Config
from mmpose.datasets import build_dataset
from mmpose.models import build_posenet
from mmpose.apis import train_model
import mmcv

if __name__ == '__main__':
    # 模型 config 配置文件
    cfg = Config.fromfile(
        'F:/baidu/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py')

    cfg.data_root = 'F:/baidu/mmpose/data/coco_whole_body'
    cfg.work_dir = 'F:/baidu/mmpose/work_dirs/res50_coco_wholebody_256x192'
    cfg.gpu_ids = range(1)
    cfg.seed = 0

    # 日志间隔
    cfg.log_config.interval = 1

    # 评估指标
    cfg.evaluation.interval = 10
    cfg.evaluation.metric = 'mAP'
    cfg.evaluation.save_best = 'AP'

    # 学习率和训练策略
    # cfg.lr_config = dict(
    #     policy='step',
    #     warmup='linear',
    #     warmup_iters=10,
    #     warmup_ratio=0.001,
    #     step=[17, 35])
    cfg.total_epochs = 6

    # load from
    # cfg.load_from = 'F:/baidu/checkpoints/res50_coco_256x192-ec54d7f3_20200709.pth'

    cfg.load_from = 'F:/baidu/mmpose/work_dirs/res50_coco_wholebody_256x192/latest.pth'

    # batch size
    cfg.data.samples_per_gpu = 8
    cfg.data.val_dataloader = dict(samples_per_gpu=8)
    cfg.data.test_dataloader = dict(samples_per_gpu=8)

    # 数据集配置
    cfg.data.train.ann_file = f'{cfg.data_root}/whole_body_new_train.json'
    cfg.data.train.img_prefix = 'F:/baidu/dataset/train2017/train2017'

    cfg.data.val.ann_file = f'{cfg.data_root}/whole_body_new_val.json'
    cfg.data.val.img_prefix = 'F:/baidu/dataset/val2017/val2017'
    cfg.data.val.data_cfg.use_gt_bbox = True

    cfg.data.test.ann_file = f'{cfg.data_root}/whole_body_new_val.json'
    cfg.data.test.img_prefix = 'F:/baidu/dataset/val2017/val2017'
    cfg.data.test.data_cfg.use_gt_bbox = True

    # build 数据集
    datasets = [build_dataset(cfg.data.train)]

    # build 模型
    model = build_posenet(cfg.model)

    # 创建 work_dir 目录
    mmcv.mkdir_or_exist(cfg.work_dir)

    # training start
    train_model(model, datasets, cfg, distributed=False, validate=True, meta=dict())
