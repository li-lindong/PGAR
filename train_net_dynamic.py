import numpy as np
import torch
import torch.optim as optim

import time
import random
import os
import sys

from config import *
from volleyball import *
from collective import *
from dataset import *
from infer_model import *
from base_model import *
from utils import *

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
            
def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train_net(cfg):
    """
    training gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list
    
    # Show config parameters
    cfg.init_config()
    show_config(cfg)
    
    # Reading dataset
    training_set, validation_set = return_dataset(cfg)
    
    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': 8, # 4,
    }
    training_loader = data.DataLoader(training_set, **params)
    
    params['batch_size'] = cfg.test_batch_size
    validation_loader = data.DataLoader(validation_set, **params)
    
    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)
    torch.cuda.manual_seed_all(cfg.train_random_seed)
    torch.cuda.manual_seed(cfg.train_random_seed)


    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Build model and optimizer
    basenet_list={'volleyball': Basenet_volleyball, 'collective': Basenet_collective}
    gcnnet_list={'dynamic_volleyball': Dynamic_volleyball,
                 'dynamic_tce_volleyball': Dynamic_TCE_volleyball,
                 'pctdm_volleyball': PCTDM_volleyball,
                 'higcin_volleyball': HiGCIN_volleyball,
                 'at_volleyball': AT_volleyball,
                 'arg_volleyball': ARG_volleyball,
                 'sacrf_biute_volleyball': SACRF_BiUTE_volleyball,
                 'dynamic_collective': Dynamic_collective,
                 'bb_ggnn_edge_collective': bb_ggnn_edge_collective,
                 'bb_ff_skeleton_trans_collective': bb_ff_skeleton_trans_collective,
                 'bb_ggnn_ff_skeleton_trans_collective': bb_ggnn_ff_skeleton_trans_collective,
                 'bb_ggnn_edge_ff_skeleton_trans_collective': bb_ggnn_edge_ff_skeleton_trans_collective,
                 'bb_gat_ff_skeleton_trans_collective': bb_gat_ff_skeleton_trans_collective,
                 'bb_gcn_ff_skeleton_trans_collective': bb_gcn_ff_skeleton_trans_collective,
                 'bb_ggnn_edge_volleyball': bb_ggnn_edge_volleyball,
                 'bb_ggnn_edge_ff_skeleton_trans_volleyball': bb_ggnn_edge_ff_skeleton_trans_volleyball,
                 'bb_gcn_ff_skeleton_trans_volleyball': bb_gcn_ff_skeleton_trans_volleyball,
                 'bb_gat_ff_skeleton_trans_volleyball': bb_gat_ff_skeleton_trans_volleyball,
                 'bb_ggnn_ff_skeleton_trans_volleyball': bb_ggnn_ff_skeleton_trans_volleyball,}

    if cfg.training_stage == 1:
        Basenet = basenet_list[cfg.dataset_name]
        model = Basenet(cfg)
    elif cfg.training_stage == 2:
        GCNnet = gcnnet_list[cfg.inference_module_name]
        model = GCNnet(cfg)
        # Load backbone
        if cfg.load_backbone_stage2:
            model.loadmodel(cfg.stage1_model_path)
            print_log(cfg.log_path, "Loading stage1 backbone model: " + cfg.stage1_model_path)
        if cfg.load_ff_skeleton_trans_stage2:
            model.load_ff_skeleton_trans_model(cfg.ff_skeleton_trans_model_path)
            print_log(cfg.log_path, "Loading stage1 ff_skeleton_trans model: " + cfg.ff_skeleton_trans_model_path)
        if cfg.load_stage2model:
            # if cfg.use_multi_gpu:
            #     model = nn.DataParallel(model)
            state = torch.load(cfg.stage2model)
            state = {k.replace('module.', ''): v for k, v in state['state_dict'].items()}
            model.load_state_dict(state)
            print_log(cfg.log_path, 'Loading stage2 model: ' + cfg.stage2model)
    else:
        assert(False)
    
    if cfg.use_multi_gpu:
        model = nn.DataParallel(model)

    model = model.to(device=device)
    
    model.train()
    if cfg.set_bn_eval:
        model.apply(set_bn_eval)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=cfg.train_learning_rate,weight_decay=cfg.weight_decay)

    train_list = {'voll': train_volleyball, 'cad': train_collective, 'cae': train_collective}
    test_list = {'voll': test_volleyball, 'cad': test_collective, 'cae': test_collective}
    train = train_list[cfg.dataset_name]
    test = test_list[cfg.dataset_name]
    
    if cfg.test_before_train:
        test_info = test(validation_loader, model, device, 0, cfg)
        print(test_info)
    if cfg.only_test:
        print("测试完成！")
        sys.exit()

    # Training iteration
    best_MCA = {'epoch':0, 'activities_acc':0}
    best_MPCA = {'epoch':0, 'MPCA':0}
    start_epoch = 1
    for epoch in range(start_epoch, start_epoch+cfg.max_epoch):
        
        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch])
            
        # One epoch of forward and backward
        train_info = train(training_loader, model, device, optimizer, epoch, cfg)
        show_epoch_info('Train', cfg.log_path, train_info)

        # Test
        if epoch % cfg.test_interval_epoch == 0:
            test_info = test(validation_loader, model, device, epoch, cfg)
            show_epoch_info('Test', cfg.log_path, test_info)
            
            if test_info['activities_acc'] > best_MCA['activities_acc']:
                best_MCA = test_info
            if test_info['MPCA'] > best_MPCA['MPCA']:
                best_MPCA = test_info
            print_log(cfg.log_path, 
                      'Best MCA: %.2f%% at epoch #%d. Best MPCA: %.2f%% at epoch #%d.'%(best_MCA['activities_acc'], best_MCA['epoch'], best_MPCA['MPCA'], best_MPCA['epoch']))
            
            # Save model
            if cfg.training_stage == 2:
                # None
                # if test_info['activities_acc'] > 93.1:
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                filepath = cfg.result_path + '/stage%d_epoch%d_%.2f%%_%.2f%%.pth'%(cfg.training_stage, epoch, test_info['activities_acc'], test_info['MPCA'])
                torch.save(state, filepath)
                print('model saved to:',filepath)
            elif cfg.training_stage == 1:
                if test_info['activities_acc'] == best_MCA['activities_acc'] or test_info['MPCA'] == best_MPCA['MPCA']:
                    filepath = cfg.result_path + '/stage%d_epoch%d_%.2f%%_%.2f%%.pth' % (
                    cfg.training_stage, epoch, test_info['activities_acc'], test_info['MPCA'])
                    if isinstance(model, torch.nn.DataParallel):
                        model.module.savemodel(filepath)
                    else:
                        model.savemodel(filepath)
                    # for m in model.modules():
                    #     if isinstance(m, Basenet):
                    #         filepath=cfg.result_path+'/stage%d_epoch%d_%.2f%%.pth'%(cfg.training_stage,epoch,test_info['activities_acc'])
                    #         m.savemodel(filepath)
            #                         print('model saved to:',filepath)
            else:
                assert False
   
def train_volleyball(data_loader, model, device, optimizer, epoch, cfg):
    train_with_action = False
    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    activities_meter_lld = AccumulateMeter(num_class=cfg.num_activities)
    loss_meter = AverageMeter()
    epoch_timer = Timer()
    activities_conf = ConfusionMeter(cfg.num_activities)
    for batch_idx, batch_data in enumerate(data_loader):
        if batch_idx % 850 == 0 and batch_idx > 0:
            print('Training in processing {}/{}, group Activity Loss: {:.4f}'.format(batch_idx, len(data_loader), loss_meter.avg))

        model.train()
        if cfg.set_bn_eval:
            model.apply(set_bn_eval)
    
        # prepare batch data
        batch_data = [b.to(device = device) for b in batch_data]
        batch_size = batch_data[0].shape[0]
        num_frames = batch_data[0].shape[1]

        actions_in = batch_data[2].reshape((batch_size, num_frames, cfg.num_boxes))
        activities_in = batch_data[3].reshape((batch_size, num_frames))

        actions_in = actions_in[:, 0, :].reshape((batch_size*cfg.num_boxes,))
        activities_in = activities_in[:, 0].reshape((batch_size,))

        # forward
        # actions_scores,activities_scores=model((batch_data[0],batch_data[1]))
        ret = model((batch_data[0], batch_data[1], batch_data[4], batch_data[5], batch_data[6]))

        # Predict activities
        loss_list = []
        if 'activities' in list(ret.keys()):
            activities_scores = ret['activities']
            activities_loss = F.cross_entropy(activities_scores, activities_in)
            loss_list.append(activities_loss)
            activities_labels = torch.argmax(activities_scores,dim=1)
            activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())
            activities_accuracy = activities_correct.item() / activities_scores.shape[0]
            activities_meter.update(activities_accuracy, activities_scores.shape[0])
            activities_conf.add(activities_labels, activities_in)
            activities_meter_lld.accumulate(activities_labels.cpu().numpy(), activities_in.cpu().numpy())

        if 'actions' in list(ret.keys()):
            # Predict actions
            actions_scores = ret['actions']
            actions_weights = torch.tensor(cfg.actions_weights).to(device=device)
            actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights) * cfg.actions_loss_weight
            loss_list.append(actions_loss)
            actions_labels = torch.argmax(actions_scores, dim=1)
            actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
            actions_accuracy = actions_correct.item() / actions_scores.shape[0]
            actions_meter.update(actions_accuracy, actions_scores.shape[0])

        if 'halting' in list(ret.keys()):
            loss_list.append(ret['halting']*cfg.halting_penalty)

        if 'activities_ff' in ret.keys():
            activities_ff_loss = F.cross_entropy(ret['activities_ff'], activities_in)
            loss_list.append(activities_ff_loss)

        if 'edge_weight' in ret.keys():
            edge_weight = torch.cat(ret['edge_weight'], dim=1).flatten()
            relation = []
            for b in range(batch_size):
                for t in range(num_frames):
                    N = cfg.num_boxes
                    actions_t = batch_data[2][b, t, :N]
                    equality_matrix = torch.eq(actions_t.unsqueeze(0), actions_t.unsqueeze(1))
                    relation_matrix = equality_matrix.int()
                    relation.append(relation_matrix.flatten())
            relation = torch.cat(relation, dim=0).float()
            loss_weight = F.mse_loss(edge_weight, relation)
            loss_list.append(loss_weight)

        # print(loss_list)
        total_loss = sum(loss_list)
        loss_meter.update(total_loss.item(), batch_size)

        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        # Test max_clip_norm
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

    activities_meter_lld.compute()
    train_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_acc': activities_meter.avg*100,
        'MPCA': activities_meter_lld.MPCA * 100,
        'activities_conf': activities_conf.value(),
        'activities_MPCA': MPCA(activities_conf.value()),
        'activities_meter': activities_meter_lld
    } #'actions_acc':actions_meter.avg*100
    
    return train_info
        
    
def test_volleyball(data_loader, model, device, epoch, cfg):
    model.eval()
    train_with_action = False
    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    activities_meter_lld = AccumulateMeter(num_class=cfg.num_activities)
    loss_meter = AverageMeter()
    activities_conf = ConfusionMeter(cfg.num_activities)
    epoch_timer = Timer()

    with torch.no_grad():
        for batch_data_test in data_loader:
            # prepare batch data
            batch_data_test = [b.to(device=device) for b in batch_data_test]
            batch_size = batch_data_test[0].shape[0]
            num_frames = batch_data_test[0].shape[1]

            actions_in = batch_data_test[2].reshape((batch_size, num_frames, cfg.num_boxes))
            activities_in = batch_data_test[3].reshape((batch_size, num_frames))
            
            # forward
            # actions_scores,activities_scores=model((batch_data_test[0],batch_data_test[1]))
            # activities_scores = model((batch_data_test[0], batch_data_test[1]))
            ret = model((batch_data_test[0], batch_data_test[1], batch_data_test[4], batch_data_test[5], batch_data_test[6]))
            
            # Predict actions
            actions_in = actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))
            activities_in = activities_in[:,0].reshape((batch_size,))

            # Predict activities
            loss_list = []
            if 'activities' in list(ret.keys()):
                activities_scores = ret['activities']
                activities_loss = F.cross_entropy(activities_scores, activities_in)
                loss_list.append(activities_loss)
                activities_labels = torch.argmax(activities_scores, dim=1)
                # Save wrong samples
                # if torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float()) == 0:
                #     wrong.append(flag)
                # if flag == 1336: # 1336
                #     np.savetxt('vis/wrong_samples.txt', wrong)
                # flag += 1


                activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())
                activities_accuracy = activities_correct.item() / activities_scores.shape[0]
                activities_meter.update(activities_accuracy, activities_scores.shape[0])
                activities_conf.add(activities_labels, activities_in)
                activities_meter_lld.accumulate(activities_labels.cpu().numpy(), activities_in.cpu().numpy())

            if 'actions' in list(ret.keys()):
                actions_scores = ret['actions']
                actions_weights = torch.tensor(cfg.actions_weights).to(device=device)
                actions_loss = F.cross_entropy(actions_scores,actions_in, weight=actions_weights)
                loss_list.append(actions_loss)
                actions_labels = torch.argmax(actions_scores,dim=1)
                actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
                actions_accuracy = actions_correct.item() / actions_scores.shape[0]
                actions_meter.update(actions_accuracy, actions_scores.shape[0])

            if 'halting' in list(ret.keys()):
                loss_list.append(ret['halting'])

            # Total loss
            total_loss = sum(loss_list)
            loss_meter.update(total_loss.item(), batch_size)

    activities_meter_lld.compute()
    test_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'MPCA': activities_meter_lld.MPCA * 100,
        'activities_conf': activities_conf.value(),
        'activities_MPCA': MPCA(activities_conf.value()),
        'activities_meter': activities_meter_lld
    } #'actions_acc':actions_meter.avg*100
    
    return test_info


def train_collective(data_loader, model, device, optimizer, epoch, cfg):
    
    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    activities_meter_lld = AccumulateMeter(num_class=cfg.num_activities)
    loss_meter = AverageMeter()
    epoch_timer = Timer()
    activities_conf = ConfusionMeter(cfg.num_activities)
    for batch_data in data_loader:
        model.train()
        model.apply(set_bn_eval)
    
        # prepare batch data
        batch_data = [b.to(device=device) for b in batch_data]
        batch_size = batch_data[0].shape[0]
        num_frames = batch_data[0].shape[1]

        # forward
        # actions_scores,activities_scores=model((batch_data[0],batch_data[1],batch_data[4]))
        output = model((batch_data[0], batch_data[1], batch_data[4], batch_data[5], batch_data[6], batch_data[7]))
        activities_in = batch_data[3].reshape((batch_size,num_frames))
        bboxes_num = batch_data[4].reshape(batch_size,num_frames)

        # actions_in = batch_data[2].reshape((batch_size,num_frames,cfg.num_boxes))
        # actions_in_nopad=[]
        # if cfg.training_stage==1:
        #     # actions_in=actions_in.reshape((batch_size*num_frames,cfg.num_boxes,))
        #     bboxes_num = bboxes_num.reshape(batch_size*num_frames,)
        #     for bt in range(batch_size*num_frames):
        #         N=bboxes_num[bt]
        #         actions_in_nopad.append(actions_in[bt,:N])
        # else:
        #     for b in range(batch_size):
        #         N = bboxes_num[b][0]
        #         actions_in_nopad.append(actions_in[b][0][:N])
        # actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)  #ALL_N,
            
        if cfg.training_stage == 1:
            activities_in = activities_in.reshape(-1,)
        else:
            activities_in = activities_in[:,0].reshape(batch_size,)
        
        # Predict actions
        # actions_loss=F.cross_entropy(actions_scores,actions_in,weight=None)
        # actions_labels=torch.argmax(actions_scores,dim=1)  #B*T*N,
        # actions_correct=torch.sum(torch.eq(actions_labels.int(),actions_in.int()).float())
        # actions_accuracy = actions_correct.item() / actions_scores.shape[0]
        # actions_meter.update(actions_accuracy, actions_scores.shape[0])


        # 计算损失
        loss_list = []
        if 'activities' in output.keys():
            if cfg.activity_weights is not None:
                activities_loss = F.cross_entropy(output['activities'], activities_in, weight=torch.tensor(cfg.activity_weights, dtype=torch.float32).cuda())
            else: activities_loss = F.cross_entropy(output['activities'], activities_in)
            loss_list.append(activities_loss)
        if 'activities_ff' in output.keys():
            if cfg.activity_weights is not None:
                activities_ff_loss = F.cross_entropy(output['activities_ff'], activities_in, weight=torch.tensor(cfg.activity_weights, dtype=torch.float32).cuda())
            else: activities_ff_loss = F.cross_entropy(output['activities_ff'], activities_in)
            loss_list.append(activities_ff_loss)
        if 'edge_weight' in output.keys():
            edge_weight = torch.cat(output['edge_weight'], dim=1).flatten()
            relation = []
            for b in range(batch_size):
                for t in range(num_frames):
                    N = bboxes_num[b, t]
                    actions_t = batch_data[2][b, t, :N]
                    equality_matrix = torch.eq(actions_t.unsqueeze(0), actions_t.unsqueeze(1))
                    relation_matrix = equality_matrix.int()
                    relation.append(relation_matrix.flatten())
            relation = torch.cat(relation, dim=0).float()
            loss_weight = F.mse_loss(edge_weight, relation)
            loss_list.append(loss_weight)
        if cfg.contractive_loss_01:
            contractive_loss_01_fun = ContrastiveLoss().cuda()
            contractive_loss_01 = contractive_loss_01_fun(output['activities_features'], activities_in)
            contractive_loss_01_ff = contractive_loss_01_fun(output['activities_features_ff'], activities_in)
            loss_list.append(contractive_loss_01)
            loss_list.append(contractive_loss_01_ff)

        # Predict activities
        activities_labels = torch.argmax(output['activities'], dim=1)  #B*T,
        activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())
        activities_meter_lld.accumulate(activities_labels.cpu().numpy(), activities_in.cpu().numpy())
        activities_accuracy = activities_correct.item() / output['activities'].shape[0]
        activities_meter.update(activities_accuracy, output['activities'].shape[0])
        activities_conf.add(activities_labels, activities_in)

        # Total loss
        # total_loss = activities_loss # + cfg.actions_loss_weight*actions_loss
        total_loss = sum(loss_list)
        loss_meter.update(total_loss.item(), batch_size)

        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    activities_meter_lld.compute()
    train_info={
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_acc': activities_meter.avg*100,
        'MPCA': activities_meter_lld.MPCA * 100,
        'activities_conf': activities_conf.value(),
        'activities_MPCA': MPCA(activities_conf.value()),
        'activities_meter': activities_meter_lld
    } # 'actions_acc':actions_meter.avg*100
    
    return train_info
        
    
def test_collective(data_loader, model, device, epoch, cfg):
    model.eval()
    
    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    activities_meter_lld = AccumulateMeter(num_class=cfg.num_activities)
    loss_meter = AverageMeter()
    
    epoch_timer = Timer()
    activities_conf = ConfusionMeter(cfg.num_activities)
    # flag = 0
    # wrong = []
    with torch.no_grad():

        # 用于存储测试结果
        if cfg.only_test:
            result_txt_file = r"{}/result_samples.txt".format(cfg.result_path)
            feature_group, feature_confidence = [], []  # 分别存储群组特征和置信度特征，用于画散点图
            edge_weight = []    # 存储边的权重，用于画可视化图

        for batch_data in data_loader:
            # prepare batch data

            init_frame_id = batch_data[8]
            batch_data = [b.to(device=device) for b in batch_data[0:8]]
            batch_size = batch_data[0].shape[0]
            num_frames = batch_data[0].shape[1]
            
            actions_in = batch_data[2].reshape((batch_size,num_frames,cfg.num_boxes))
            activities_in = batch_data[3].reshape((batch_size,num_frames))
            bboxes_num = batch_data[4].reshape(batch_size,num_frames)

            # forward
            output = model((batch_data[0], batch_data[1], batch_data[4], batch_data[5], batch_data[6], batch_data[7]))

            # actions_scores,activities_scores=model((batch_data[0],batch_data[1],batch_data[4]))
            
            # actions_in_nopad=[]
            # if cfg.training_stage==1:
            #     actions_in=actions_in.reshape((batch_size*num_frames,cfg.num_boxes,))
            #     bboxes_num=bboxes_num.reshape(batch_size*num_frames,)
            #     for bt in range(batch_size*num_frames):
            #         N=bboxes_num[bt]
            #         actions_in_nopad.append(actions_in[bt,:N])
            # else:
            #     for b in range(batch_size):
            #         N=bboxes_num[b][0]
            #         actions_in_nopad.append(actions_in[b][0][:N])
            # actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)  #ALL_N,
            
            if cfg.training_stage==1:
                activities_in=activities_in.reshape(-1,)
            else:
                activities_in=activities_in[:, 0].reshape(batch_size,)

            # actions_loss=F.cross_entropy(actions_scores,actions_in)
            # actions_labels=torch.argmax(actions_scores,dim=1)  #ALL_N,
            # actions_correct=torch.sum(torch.eq(actions_labels.int(),actions_in.int()).float())
            # actions_accuracy = actions_correct.item() / actions_scores.shape[0]
            # actions_meter.update(actions_accuracy, actions_scores.shape[0])

            # Predict activities
            activities_loss = F.cross_entropy(output['activities'], activities_in)
            activities_labels = torch.argmax(output['activities'], dim=1)  #B,
            activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())
            activities_meter_lld.accumulate(activities_labels.cpu().numpy(), activities_in.cpu().numpy())
            activities_accuracy = activities_correct.item() / output['activities'].shape[0]
            activities_meter.update(activities_accuracy, output['activities'].shape[0])
            activities_conf.add(activities_labels, activities_in)

            if cfg.only_test:
                feature_group.append(output['activities_features'].cpu().numpy())
                feature_confidence.append(output['activities'].cpu().numpy())
                with open(result_txt_file, 'a+', encoding='utf-8') as f:
                    [f.writelines('{}\t{}\n'.format(id, yp)) for id, yp in zip(init_frame_id, activities_labels)]
                [edge_weight.append(torch.mean(ew, dim=0).cpu().numpy().tolist()) for ew in output['edge_weight']]

            # Save wrong samples
            # if torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float()) == 0:
            #     wrong.append(flag)
            # if flag == 764:
            #     np.savetxt('vis/Collective/wrong_samples.txt', wrong)
            # flag += 1

            # Total loss
            total_loss = activities_loss # + cfg.actions_loss_weight*actions_loss
            loss_meter.update(total_loss.item(), batch_size)

    if cfg.only_test:
        np.save(r"{}/feature_group.npy".format(cfg.result_path), np.concatenate(feature_group, axis=0))
        np.save(r"{}/feature_confidence.npy".format(cfg.result_path), np.concatenate(feature_confidence, axis=0))
        json.dump(edge_weight, open(r"{}/edge_weight.json".format(cfg.result_path), "w", encoding="utf-8"))

    activities_meter_lld.compute()
    test_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'MPCA': activities_meter_lld.MPCA * 100,
        'activities_conf': activities_conf.value(),
        'activities_MPCA': MPCA(activities_conf.value()),
        'activities_meter': activities_meter_lld
    } #'actions_acc':actions_meter.avg*100
    
    return test_info
