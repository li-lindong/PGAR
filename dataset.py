from volleyball import *
from collective import *

import pickle
import json


def return_dataset(cfg):
    if cfg.dataset_name=='voll':
        train_anns = volley_read_dataset(cfg.data_path, cfg.train_seqs)
        # 读取ff标签，并存入train_anns
        anno_ff_train = json.load(open(cfg.anno_ff_train, 'r', encoding='utf-8'))
        for key, value in train_anns.items():
            seq_id = str(key)
            for kk, vv in value.items():
                frame_id = str(kk)
                # 朝向转换为one-hot编码，位置坐标进行归一化
                temp_orientations = anno_ff_train['{}-{}'.format(seq_id, frame_id)]['orientation']
                temp_positions = anno_ff_train['{}-{}'.format(seq_id, frame_id)]['top_coordinate']
                ori_one_hot = []
                for ori, pos in zip(temp_orientations, temp_positions):
                    one_hot = [0] * 8
                    one_hot[ori] = 1
                    ori_one_hot.append(one_hot)
                    pos[0] = (pos[0] - cfg.top_coordinate_x[0]) / (cfg.top_coordinate_x[1] - cfg.top_coordinate_x[0])
                    pos[1] = (pos[1] - cfg.top_coordinate_y[0]) / (cfg.top_coordinate_y[1] - cfg.top_coordinate_y[0])
                train_anns[key][kk]['orientation'] = ori_one_hot
                train_anns[key][kk]['position'] = temp_positions
        train_frames = volley_all_frames(train_anns)

        test_anns = volley_read_dataset(cfg.data_path, cfg.test_seqs)
        # 读取ff标签，并存入test_anns
        anno_ff_test = json.load(open(cfg.anno_ff_test, 'r', encoding='utf-8'))
        for key, value in test_anns.items():
            seq_id = str(key)
            for kk, vv in value.items():
                frame_id = str(kk)
                # 朝向转换为one-hot编码，位置坐标进行归一化
                temp_orientations = anno_ff_test['{}-{}'.format(seq_id, frame_id)]['orientation']
                temp_positions = anno_ff_test['{}-{}'.format(seq_id, frame_id)]['top_coordinate']
                ori_one_hot = []
                for ori, pos in zip(temp_orientations, temp_positions):
                    one_hot = [0] * 8
                    one_hot[ori] = 1
                    ori_one_hot.append(one_hot)
                    pos[0] = (pos[0] - cfg.top_coordinate_x[0]) / (cfg.top_coordinate_x[1] - cfg.top_coordinate_x[0])
                    pos[1] = (pos[1] - cfg.top_coordinate_y[0]) / (cfg.top_coordinate_y[1] - cfg.top_coordinate_y[0])
                test_anns[key][kk]['orientation'] = ori_one_hot
                test_anns[key][kk]['position'] = temp_positions
        test_frames = volley_all_frames(test_anns)

        all_anns = {**train_anns, **test_anns}
        all_tracks = pickle.load(open(cfg.data_path + '/tracks_normalized.pkl', 'rb'))
        all_skeletons = volleyball_readpose(cfg.anno_skeleton)

        training_set = VolleyballDataset(all_anns, all_tracks, all_skeletons, train_frames,
                                      cfg.data_path, cfg.image_size, cfg.out_size, cfg.inference_module_name, num_before=cfg.num_before,
                                       num_after=cfg.num_after, is_training=True, is_finetune=(cfg.training_stage==1))

        validation_set = VolleyballDataset(all_anns, all_tracks, all_skeletons, test_frames,
                                      cfg.data_path, cfg.image_size, cfg.out_size, cfg.inference_module_name, num_before=cfg.num_before,
                                         num_after=cfg.num_after, is_training=False, is_finetune=(cfg.training_stage==1))
    
    elif cfg.dataset_name == 'cad' or 'cae':
        train_anns = collective_read_dataset(cfg.data_path, cfg.train_seqs)
        # 读取ff和skeleton标签，并存入train_anns
        anno_ff_skeleton_train = json.load(open(cfg.anno_ff_skeleton_train, 'r', encoding='utf-8'))
        for key, value in train_anns.items():
            seq_id = str(key).zfill(3)
            for kk, vv in value.items():
                frame_id = str(kk).zfill(4)
                # 朝向转换为one-hot编码，位置坐标进行归一化
                temp_orientations = anno_ff_skeleton_train['{}-{}'.format(seq_id, frame_id)]['pose']
                temp_positions = anno_ff_skeleton_train['{}-{}'.format(seq_id, frame_id)]['top_coordinates']
                ori_one_hot = []
                for ori, pos in zip(temp_orientations, temp_positions):
                    if ori == -1:   # -2表示NA
                        ori_one_hot.append([0] * 8)
                    else:
                        one_hot = [0] * 8
                        one_hot[ori] = 1
                        ori_one_hot.append(one_hot)
                    pos[0] = (pos[0] - cfg.top_coordinate_x[0]) / (cfg.top_coordinate_x[1] - cfg.top_coordinate_x[0])
                    pos[1] = (pos[1] - cfg.top_coordinate_y[0]) / (cfg.top_coordinate_y[1] - cfg.top_coordinate_y[0])
                train_anns[key][kk]['orientation'] = ori_one_hot
                train_anns[key][kk]['position'] = temp_positions
                train_anns[key][kk]['skeleton'] = anno_ff_skeleton_train['{}-{}'.format(seq_id, frame_id)]['keypoint']
        train_frames = collective_all_frames(train_anns)

        test_anns = collective_read_dataset(cfg.data_path, cfg.test_seqs)
        # 读取ff和skeleton标签，并存入test_anns
        anno_ff_skeleton_test = json.load(open(cfg.anno_ff_skeleton_test, 'r', encoding='utf-8'))
        for key, value in test_anns.items():
            seq_id = str(key).zfill(3)
            for kk, vv in value.items():
                frame_id = str(kk).zfill(4)
                # 朝向转换为one-hot编码，位置坐标进行归一化
                temp_orientations = anno_ff_skeleton_test['{}-{}'.format(seq_id, frame_id)]['pose']
                temp_positions = anno_ff_skeleton_test['{}-{}'.format(seq_id, frame_id)]['top_coordinates']
                ori_one_hot = []
                for ori, pos in zip(temp_orientations, temp_positions):
                    if ori == -1:   # -2表示NA
                        ori_one_hot.append([0] * 8)
                    else:
                        one_hot = [0] * 8
                        one_hot[ori] = 1
                        ori_one_hot.append(one_hot)
                    pos[0] = (pos[0] - cfg.top_coordinate_x[0]) / (cfg.top_coordinate_x[1] - cfg.top_coordinate_x[0])
                    pos[1] = (pos[1] - cfg.top_coordinate_y[0]) / (cfg.top_coordinate_y[1] - cfg.top_coordinate_y[0])
                test_anns[key][kk]['orientation'] = ori_one_hot
                test_anns[key][kk]['position'] = temp_positions
                test_anns[key][kk]['skeleton'] = anno_ff_skeleton_test['{}-{}'.format(seq_id, frame_id)]['keypoint']
        test_frames = collective_all_frames(test_anns)

        training_set = CollectiveDataset(train_anns, train_frames,
                                      cfg.data_path, cfg.image_size, cfg.out_size,
                                      num_frames = cfg.num_frames, is_training=True, is_finetune=(cfg.training_stage==1))

        validation_set = CollectiveDataset(test_anns, test_frames,
                                      cfg.data_path, cfg.image_size, cfg.out_size,
                                      num_frames = cfg.num_frames, is_training=False, is_finetune=(cfg.training_stage==1))
                              
    else:
        assert False
                                         
    
    print('Reading dataset finished...')
    print('%d train samples'%len(train_frames))
    print('%d test samples'%len(test_frames))
    
    return training_set, validation_set



def volleyball_readpose(data_path):
    f = open(data_path,'r')
    f = f.readlines()
    pose_ann = dict()
    for ann in f:
        ann = json.loads(ann)
        filename = ann['filename'].split('/')
        sid = filename[-3]
        src_id = filename[-2]
        fid = filename[-1][:-4]
        center = [ann['tmp_box'][0], ann['tmp_box'][1]]
        keypoint = []
        for i in range(0,51,3):
            keypoint.append(ann['keypoints'][i])
            keypoint.append(ann['keypoints'][i+1])
        pose_ann[sid + src_id + fid + str(center)] = keypoint
    return pose_ann