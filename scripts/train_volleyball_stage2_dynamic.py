import sys
# sys.path.append(".")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录
current_dir = os.path.dirname(current_file_path)
# 获取上一级目录，即 project 的路径
parent_dir = os.path.dirname(current_dir)
# 现在 parent_dir 就是 project 的路径
root_dir = parent_dir
sys.path.append(root_dir)
from train_net_dynamic import *

cfg = Config('voll')
cfg.inference_module_name = 'bb_ggnn_ff_skeleton_trans_volleyball'

cfg.device_list = "1"
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.training_stage = 2
cfg.train_backbone = True
cfg.test_interval_epoch = 1

# vgg16 setup
# cfg.backbone = 'vgg16'
# cfg.load_backbone_stage2 = True
# cfg.stage1_model_path = '/mnt/disk/lld_data/DIN-GAR/result/Voll/vgg16/[Volleyball_stage1_stage1]<2024-10-28_16-43-56>/stage1_epoch100_92.52%_92.71%.pth'
# cfg.load_ff_skeleton_trans_stage2 = True
# cfg.ff_skeleton_trans_model_path = r"/mnt/disk/lld_data/GroupFormer-main/results/ff_new(加入pose支路，使用Transformer在individual级融合三支路特征)/checkpoint_256_58_84.443_99.028.pth.tar"
# cfg.out_size = 22, 40
# cfg.emb_features = 512

# res18 setup
# cfg.backbone = 'res18'
# cfg.load_backbone_stage2 = True
# cfg.stage1_model_path = '/mnt/disk/LiLinDong/DIN-GAR/result/Voll/res18/[Volleyball_stage1_stage1]<2024-10-31_19-34-51>/stage1_epoch109_90.88%_91.50%.pth'
# cfg.load_ff_skeleton_trans_stage2 = True
# cfg.ff_skeleton_trans_model_path = r"/mnt/disk/LiLinDong/GroupFormer-main/results/fformation/ff_new(加入pose支路，使用Transformer在individual级融合三支路特征)/checkpoint_256_58_84.443_99.028.pth.tar"
# cfg.image_size = 720, 1280
# cfg.out_size = 23, 40
# cfg.emb_features = 512

# inv3 setup
cfg.backbone = 'inv3'
cfg.load_backbone_stage2 = True
cfg.stage1_model_path = r"/mnt/disk/LiLinDong/DIN-GAR/result/Voll/inv3/[Volleyball_stage1_stage1_(57, 87)]<2024-10-31_00-12-38>/stage1_epoch58_92.07%_92.54%.pth"
cfg.load_ff_skeleton_trans_stage2 = True
cfg.ff_skeleton_trans_model_path = r"/mnt/disk/LiLinDong/GroupFormer-main/results/fformation/ff_new(加入pose支路，使用Transformer在individual级融合三支路特征)/checkpoint_256_58_84.443_99.028.pth.tar"
cfg.image_size = 720, 1280
cfg.out_size = 57, 87
cfg.emb_features = 1056

# Dynamic Inference setup
cfg.group = 1
cfg.stride = 1
cfg.ST_kernel_size = [(3, 3)] #[(3, 3),(3, 3),(3, 3),(3, 3)]
cfg.dynamic_sampling = True
cfg.sampling_ratio = [1]
cfg.lite_dim = 128 # None # 128
cfg.scale_factor = True
cfg.beta_factor = False
cfg.hierarchical_inference = False
cfg.parallel_inference = False
cfg.num_DIM = 1
cfg.train_dropout_prob = 0.3

cfg.batch_size = 6
cfg.test_batch_size = 6
cfg.num_frames = 10
cfg.train_learning_rate = 1e-4
# cfg.lr_plan = {11: 3e-5, 21: 1e-5}
# cfg.max_epoch = 60
cfg.lr_plan = {11: 3e-5, 21: 1e-5}
# cfg.lr_plan = {11: 1e-5}
cfg.max_epoch = 200  # 30
cfg.actions_weights = [[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]

# 测试相关
cfg.only_test = False
cfg.test_before_train = False
cfg.load_stage2model = False
cfg.stage2model = r"/mnt/disk/LiLinDong/DIN-GAR/result/Voll/res18/[bb_ggnn_edge_ff_skeleton_trans_volleyball_stage2]<2024-11-01_15-22-37>/stage2_epoch17_91.70%_92.08%.pth"

cfg.exp_note = cfg.inference_module_name
train_net(cfg)
