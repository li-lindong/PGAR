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

cfg = Config('cad')

cfg.device_list = "1"
cfg.training_stage = 1
cfg.train_backbone = True

# inv3 setup
cfg.backbone = 'inv3'  # inv3, res18, vgg16
cfg.image_size = 480, 720   # CAD, CAED为 (480, 720), Volleyball为 (720, 1280)
cfg.out_size = 57, 87   # (57, 87)  (15, 23)  (15. 22)  (87, 157)
cfg.emb_features = 1056  # output feature map channel of backbone（inv3: 1056, res18: 512, vgg16: 512）


cfg.num_boxes = 13  # CAD, CAE为13，Volleyball为12
cfg.num_actions = 5 # CAD为5，CAE为7，volleyball为9
cfg.actions_weights = None
cfg.num_activities = 4    # CAD为4，CAE为6，volleyball为8
cfg.num_frames = 10

cfg.batch_size = 16
cfg.test_batch_size = 16
cfg.train_learning_rate = 1e-5  # CAD和CAE为5e-5保持不变，Volleyball为1e-4并且每10epoch衰减为原来的1/3
cfg.train_dropout_prob = 0.5
cfg.weight_decay = 1e-2
cfg.lr_plan = {}
cfg.max_epoch = 100

cfg.exp_note = 'Collective_stage1'
train_net(cfg)
