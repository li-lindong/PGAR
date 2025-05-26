import time
import os


class Config(object):
    """
    class to save config parameter
    """

    def __init__(self, dataset_name):
        # 获取根目录（以当前文件夹作为根目录）
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        self.root_dir = current_dir

        # Global
        self.image_size = 720, 1280  #input image size
        self.batch_size =  32  #train batch size 
        self.test_batch_size = 8  #test batch size
        self.num_boxes = 12  #max number of bounding boxes in each frame
        
        # Gpu
        self.use_gpu = True
        self.use_multi_gpu = True
        self.device_list = "0, 1, 2, 3"  #id list of gpus used for training
        
        # Dataset
        assert(dataset_name in ['voll', 'cad', 'cae'])
        self.dataset_name = dataset_name
        
        if dataset_name == 'voll':
            self.data_path = '/mnt/disk/LiLinDong/dataset/Volleyball/videos' #data path for the volleyball dataset
            self.train_seqs = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54,
                            0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]  #video id list of train set
            self.test_seqs = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]  #video id list of test set
            self.anno_ff_train = r"/mnt/disk/LiLinDong/GAR/my_code/data_processing/voll/voll_train.json"
            self.anno_ff_test = r"/mnt/disk/LiLinDong/GAR/my_code/data_processing/voll/voll_test.json"
            self.anno_skeleton = r"/mnt/disk/LiLinDong/dataset/Volleyball/volleyball_result_kpt.json"
            self.top_coordinate_x = -90.64021548726014, 111.6840163861746  # top view视角下x轴取值范围
            self.top_coordinate_y = 0.0, 255.0  # top view视角下y轴取值范围
        elif dataset_name == 'cad':
            self.data_path = '/mnt/disk/LiLinDong/dataset/CAD/seqs'  #data path for the collective dataset
            self.test_seqs = [5, 6, 7, 8, 9, 10, 11, 15, 16, 25, 28, 29]
            self.train_seqs = [s for s in range(1, 45) if s not in self.test_seqs]
            self.anno_ff_skeleton_train = r"/mnt/disk/LiLinDong/GAR/my_code/data_processing/cad_cae/cad_train_kp.json"
            self.anno_ff_skeleton_test = r"/mnt/disk/LiLinDong/GAR/my_code/data_processing/cad_cae/cad_test_kp.json"
            self.top_coordinate_x = -39.19795231930578, 43.75902707580871   # top view视角下x轴取值范围
            self.top_coordinate_y = 8.0, 255.0  # top view视角下y轴取值范围
        else:   # cae
            self.data_path = '/mnt/disk/LiLinDong/dataset/CAD/seqs'  #data path for the collective dataset
            self.test_seqs = [5, 6, 7, 8, 9, 10, 11, 15, 16, 25, 28, 29] + [52, 53, 59, 68, 69, 70]
            self.train_seqs = [s for s in range(1, 73) if s not in self.test_seqs]
            self.anno_ff_skeleton_train = r"/mnt/disk/LiLinDong/GAR/my_code/data_processing/cad_cae/cae_train_kp.json"
            self.anno_ff_skeleton_test = r"/mnt/disk/LiLinDong/GAR/my_code/data_processing/cad_cae/cae_test_kp.json"
            self.top_coordinate_x = -72.5372954930138, 67.49740394600208  # top view视角下x轴取值范围
            self.top_coordinate_y = 0.0, 255.0  # top view视角下y轴取值范围

        # Backbone 
        self.backbone = 'res18'    # inv3, vgg16, vgg19, res18
        self.crop_size = 5, 5  # crop size of roi align
        self.train_backbone = False  # if freeze the feature extraction part of network, True for stage 1, False for stage 2
        self.out_size = 87, 157  # output feature map size of backbone
        self.emb_features = 512   # output feature map channel of backbone（inv3: 1056, res18: 512, vgg16: 512）

        
        # Activity Action
        self.num_actions = 9  #number of action categories
        self.num_activities = 8  #number of activity categories
        self.actions_loss_weight = 1.0  #weight used to balance action loss and activity loss
        self.actions_weights = None

        # Sample
        self.num_frames = 3 
        self.num_before = 5
        self.num_after = 4

        # ARG params
        self.num_features_boxes = 1024
        self.num_features_relation = 256
        self.num_graph = 16  #number of graphs
        self.num_features_gcn = self.num_features_boxes
        self.gcn_layers = 1  #number of GCN layers
        self.tau_sqrt = False
        self.pos_threshold = 0.2  #distance mask threshold in position relation

        # Training Parameters
        self.train_random_seed = 0
        self.train_learning_rate = 1e-4  #initial learning rate
        self.lr_plan = {11:3e-5, 21:1e-5}  #change learning rate in these epochs
        self.train_dropout_prob = 0.3  #dropout probability
        self.weight_decay = 0  #l2 weight decay
    
        self.max_epoch = 30  #max training epoch
        self.test_interval_epoch = 1
        
        # Exp
        self.training_stage = 1  #specify stage1 or stage2
        self.stage1_model_path = ''  #path of the base model, need to be set in stage2
        self.test_before_train = False
        self.exp_note = 'Group-Activity-Recognition'
        self.exp_name = None
        self.set_bn_eval = False
        self.inference_module_name = 'dynamic_volleyball'

        # Dynamic Inference
        self.stride = 1
        self.ST_kernel_size = 3
        self.dynamic_sampling = True
        self.sampling_ratio = [1, 3]  # [1,2,4]
        self.group = 1
        self.scale_factor = True
        self.beta_factor = True
        self.load_backbone_stage2 = False
        self.parallel_inference = False
        self.hierarchical_inference = False
        self.lite_dim = None
        self.num_DIM = 1
        self.load_stage2model = False
        self.stage2model = None

        # Actor Transformer
        self.temporal_pooled_first = False

        # SACRF + BiUTE
        self.halting_penalty = 0.0001

        
        
    def init_config(self, need_new_folder=True):
        if self.exp_name is None:
            time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.exp_name = '[%s_stage%d]<%s>' % (self.exp_note, self.training_stage, time_str)
            
        self.result_path = '%s/result/%s' % (self.root_dir, self.exp_name)
        self.log_path = '%s/result/%s/log.txt' % (self.root_dir, self.exp_name)
            
        if need_new_folder:
            os.mkdir(self.result_path)
