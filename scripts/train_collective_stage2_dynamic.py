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

from train_net_dynamic import *

cfg = Config('cae')
cfg.inference_module_name = 'bb_ggnn_edge_ff_skeleton_trans_collective'

cfg.device_list = "0"
cfg.training_stage = 2
cfg.use_gpu = True
cfg.use_multi_gpu = False
cfg.train_backbone = False

# lld
cfg.backbone = 'inv3'
cfg.image_size = 480, 720
cfg.out_size = 57, 87   # (57, 87)  (15, 23)  (15, 22)
cfg.emb_features = 1056 # output feature map channel of backbone（inv3: 1056, res18: 512, vgg16: 512）
cfg.load_backbone_stage2 = False
cfg.stage1_model_path = r"/mnt/disk/LiLinDong/DIN-GAR/result/CAD/inv3/[Collective_stage1_stage1]<2024-10-22_20-12-49>/stage1_epoch38_96.94%_97.25%.pth"
# cfg.stage1_model_path = r"/mnt/disk/LiLinDong/DIN-GAR/result/CAE/inv3/[Collective_stage1_stage1_(57, 87)]<2024-10-22_20-16-10>/stage1_epoch52_97.59%_97.22%.pth"
cfg.load_ff_skeleton_trans_stage2 = False
cfg.ff_skeleton_trans_model_path = r"/mnt/disk/LiLinDong/GAR/model_files/pretrained_model/CAD/ff/2024-10-31_21-47-56_ff_skeleton_trans_cad/072_ff_skeleton_trans_78.82_70.05.pth"
# cfg.ff_skeleton_trans_model_path = r"/mnt/disk/LiLinDong/GAR/model_files/pretrained_model/CAE/ff/2024-12-27_15-00-35_ff_skeleton_trans_cae/048_ff_skeleton_trans_70.36_65.66.pth"


# ResNet18
# cfg.backbone = 'res18'
# cfg.image_size = 480, 720
# cfg.out_size = 57, 87
# cfg.emb_features = 512
# cfg.load_backbone_stage2 = False
# cfg.stage1_model_path = r"/mnt/disk/LiLinDong/DIN-GAR/result/CAD/res18/[Collective_stage1_stage1_(15, 23)]<2024-10-25_01-00-59>/stage1_epoch89_94.46%_95.11%.pth"
# cfg.load_ff_skeleton_trans_stage2 = False
# cfg.ff_skeleton_trans_model_path = r"/mnt/disk/LiLinDong/GAR/model_files/pretrained_model/CAD/ff/2024-10-31_21-47-56_ff_skeleton_trans_cad/072_ff_skeleton_trans.pth"

# VGG16
# cfg.backbone = 'vgg16'
# cfg.image_size = 480, 720
# cfg.out_size = 57, 87
# cfg.emb_features = 512
# cfg.load_backbone_stage2 = False
# cfg.stage1_model_path = r"/mnt/disk/LiLinDong/DIN-GAR/result/CAD/vgg16/[Collective_stage1_stage1_(15, 22)]<2024-10-25_01-01-38>/stage1_epoch93_95.25%_93.82%.pth"
# cfg.load_ff_skeleton_trans_stage2 = False
# cfg.ff_skeleton_trans_model_path = r"/mnt/disk/LiLinDong/GAR/model_files/pretrained_model/CAD/ff/2024-10-31_21-47-56_ff_skeleton_trans_cad/072_ff_skeleton_trans.pth"

cfg.num_boxes = 13  # CAD, CAE为13，Volleyball为12
cfg.num_actions = 7 # CAD为5，CAE为7
cfg.num_activities = 6  # CAD为4，CAE为6
# cfg.activity_weights = [1, 3, 1, 1]
cfg.activity_weights = None
cfg.num_frames = 10
cfg.num_graph = 4
cfg.tau_sqrt = True
cfg.batch_size = 8
cfg.test_batch_size = 4
cfg.test_interval_epoch = 1
cfg.train_learning_rate = 5e-5
cfg.train_dropout_prob = 0.5
cfg.weight_decay = 1e-4
cfg.lr_plan = {}
cfg.max_epoch = 30

# 0类和1类的对比损失
cfg.contractive_loss_01 =  False

# Dynamic Inference setup
# cfg.group = 1
# cfg.stride = 1
# cfg.ST_kernel_size = (3, 3)
# cfg.dynamic_sampling = True
# cfg.sampling_ratio = [1]  # [1,2,4]
# cfg.lite_dim = None # 128
# cfg.scale_factor = True
# cfg.beta_factor = False
# cfg.hierarchical_inference = False
# cfg.parallel_inference = False

# 测试相关
cfg.only_test = True
cfg.test_before_train = True
cfg.load_stage2model = True
# cfg.stage2model = None
# cfg.stage2model = r"/mnt/disk/LiLinDong/DIN-GAR/result/CAD/inv3/[bb_ggnn_edge_ff_skeleton_trans_collective_stage2]<2024-11-04_00-27-33>/stage2_epoch12_97.52%_96.98%.pth"
# cfg.stage2model = r"/mnt/disk/LiLinDong/DIN-GAR/result/CAD/res18/[bb_ggnn_edge_ff_skeleton_trans_collective_stage2]_(15, 23)<2024-11-06_16-12-19>/stage2_epoch4_95.82%_96.94%.pth"
cfg.stage2model = r"/mnt/disk/LiLinDong/DIN-GAR/result/CAE/inv3/[bb_ggnn_edge_ff_skeleton_trans_collective_stage2_(57, 87)]<2024-12-28_22-04-22>/stage2_epoch1_97.36%_96.01%.pth"

cfg.exp_note = cfg.inference_module_name
train_net(cfg)