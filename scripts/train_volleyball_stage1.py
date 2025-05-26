import sys
# sys.path.append(".")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录
current_dir = os.path.dirname(current_file_path)
# 获取上一级目录，即 project 的路径
parent_dir = os.path.dirname(current_dir)
# 现在 parent_dir 就是 project 的路径
root_dir = parent_dir
sys.path.append(root_dir)
from train_net import *

cfg=Config('voll')

cfg.use_multi_gpu = False
cfg.device_list = "0"
cfg.training_stage = 1
cfg.stage1_model_path = ''
cfg.train_backbone = True
cfg.test_before_train = False

# VGG16
# cfg.backbone = 'vgg16'
# cfg.image_size = 720, 1280
# cfg.out_size = 22, 40
# cfg.emb_features = 512

# res18
# cfg.backbone = 'res18'
# cfg.image_size = 720, 1280
# cfg.out_size = 23, 40
# cfg.emb_features = 512

# inv3
cfg.backbone = 'inv3'
cfg.image_size = 720, 1280
# cfg.out_size = 57, 87
cfg.out_size = 87, 157
cfg.emb_features = 1056

cfg.num_before = 5
cfg.num_after = 4

cfg.batch_size = 16
cfg.test_batch_size = 16
cfg.num_frames = 1
# cfg.train_learning_rate=1e-5
# cfg.lr_plan={}
# cfg.max_epoch=200
cfg.train_learning_rat = 1e-4
cfg.lr_plan = {30:5e-5, 60:2e-5, 90:1e-5}
cfg.max_epoch = 120
cfg.set_bn_eval = False
cfg.actions_weights = [1., 1., 2., 3., 1., 2., 2., 0.2, 1.]

cfg.exp_note = 'Volleyball_stage1'
train_net(cfg)
