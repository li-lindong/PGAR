import torch
import time
import numpy as np
import torchvision.transforms as transforms
from thop import profile, clever_format
import torch.nn as nn
import torch.nn.functional as F

def prep_images(images):
    """
    preprocess images
    Args:
        images: pytorch tensor
    """
    images = images.div(255.0)

    images = torch.sub(images,0.5)
    images = torch.mul(images,2.0)

    return images

# def prep_images(images):
#     """
#     preprocess images
#     Args:
#         images: pytorch tensor
#     """
#     # Reference: pytorch.org/docs/stable/torchvision/models.html
#
#     images = images.div(255.0)
#     normalizer = transforms.Normalize(
#         mean = [0.485, 0.456, 0.406],
#         std = [0.229, 0.224, 0.225]
#     )
#
#     for i in range(images.shape[0]):
#         images[i] = normalizer(images[i])
#     #images = torch.sub(images,0.5)
#     #images = torch.mul(images,2.0)
#
#     return images

def calc_pairwise_distance(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [N,D]
        Y: [M,D]
    Returns:
        dist: [N,M] matrix of euclidean distances
    """
    rx=X.pow(2).sum(dim=1).reshape((-1,1))
    ry=Y.pow(2).sum(dim=1).reshape((-1,1))
    dist=rx-2.0*X.matmul(Y.t())+ry.t()
    return torch.sqrt(dist)

def calc_pairwise_distance_3d(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [B,N,D]
        Y: [B,M,D]
    Returns:
        dist: [B,N,M] matrix of euclidean distances
    """
    B=X.shape[0]
    
    rx=X.pow(2).sum(dim=2).reshape((B,-1,1))
    ry=Y.pow(2).sum(dim=2).reshape((B,-1,1))
    
    dist=rx-2.0*X.matmul(Y.transpose(1,2))+ry.transpose(1,2)
    
    return torch.sqrt(dist)

def sincos_encoding_2d(positions,d_emb):
    """
    Args:
        positions: [N,2]
    Returns:
        positions high-dimensional representation: [N,d_emb]
    """

    N=positions.shape[0]
    
    d=d_emb//2
    
    idxs = [np.power(1000,2*(idx//2)/d) for idx in range(d)]
    idxs = torch.FloatTensor(idxs).to(device=positions.device)
    
    idxs = idxs.repeat(N,2)  #N, d_emb
    
    pos = torch.cat([ positions[:,0].reshape(-1,1).repeat(1,d),positions[:,1].reshape(-1,1).repeat(1,d) ],dim=1)

    embeddings=pos/idxs
    
    embeddings[:,0::2]=torch.sin(embeddings[:,0::2])  # dim 2i
    embeddings[:,1::2]=torch.cos(embeddings[:,1::2])  # dim 2i+1
    
    return embeddings


def print_log(file_path,*args):
    print(*args)
    if file_path is not None:
        with open(file_path, 'a') as f:
            print(*args,file=f)

def show_config(cfg):
    print_log(cfg.log_path, '=====================Config=====================')
    for k,v in cfg.__dict__.items():
        print_log(cfg.log_path, k,': ',v)
    print_log(cfg.log_path, '======================End=======================')
    
def show_epoch_info(phase, log_path, info):
    print_log(log_path, '')
    if phase=='Test':
        print_log(log_path, '====> %s at epoch #%d'%(phase, info['epoch']))
    else:
        print_log(log_path, '%s at epoch #%d'%(phase, info['epoch']))
        
    print_log(log_path, 'Group Activity Accuracy: %.2f%%, Loss: %.5f, Using %.1f seconds'%(
                info['activities_acc'], info['loss'], info['time']))
    print_log(log_path, "MCA: {:.3f} \t MPCA: {:.3f}".format(info['activities_acc'], info['activities_meter'].MPCA*100))
    recall_txt = ""
    for r in info['activities_meter'].recall.values():
        recall_txt += "{:.2f}\t".format(r*100)
    print_log(log_path, "召回率： {}".format(recall_txt))

    if 'activities_conf' in info.keys():
        print_log(log_path, info['activities_conf'])
    if 'activities_MPCA' in info.keys():
        print_log(log_path, 'Activities MPCA:{:.2f}%'.format(info['activities_MPCA']))
    if 'MAD' in info.keys():
        print_log(log_path, 'MAD:{:.4f}'.format(info['MAD']))
    print_log(log_path, '\n')
        
def log_final_exp_result(log_path, data_path, exp_result):
    no_display_cfg=['num_workers', 'use_gpu', 'use_multi_gpu', 'device_list',
                   'batch_size_test', 'test_interval_epoch', 'train_random_seed',
                   'result_path', 'log_path', 'device']
    
    with open(log_path, 'a') as f:
        print('', file=f)
        print('', file=f)
        print('', file=f)
        print('=====================Config=====================', file=f)
        
        for k,v in exp_result['cfg'].__dict__.items():
            if k not in no_display_cfg:
                print( k,': ',v, file=f)
            
        print('=====================Result======================', file=f)
        
        print('Best result:', file=f)
        print(exp_result['best_result'], file=f)
            
        print('Cost total %.4f hours.'%(exp_result['total_time']), file=f)
        
        print('======================End=======================', file=f)
    
    
    data_dict=pickle.load(open(data_path, 'rb'))
    data_dict[exp_result['cfg'].exp_name]=exp_result
    pickle.dump(data_dict, open(data_path, 'wb'))
        
    
class AverageMeter(object):
    """
    Computes the average value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

class Timer(object):
    """
    class to do timekeeping
    """
    def __init__(self):
        self.last_time=time.time()
        
    def timeit(self):
        old_time=self.last_time
        self.last_time=time.time()
        return self.last_time-old_time

class ConfusionMeter(object):
    """Maintains a confusion matrix for a given calssification problem.

    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.

    Args:
        k (int): number of classes in the classification problem
        normalized (boolean): Determines whether or not the confusion matrix
            is normalized or not

    """

    def __init__(self, k, normalized=False):
        super(ConfusionMeter, self).__init__()
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix of K x K size where K is no of classes

        Args:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of
                integer values between 0 and K-1.
            target (tensor): Can be a N-tensor of integer values assumed to be integer
                values between 0 and K-1 or N x K tensor, where targets are
                assumed to be provided as one-hot vectors

        """
        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.k ** 2)

        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf

def MPCA(conf_mat):
    '''

    :param conf_mat: np.ndarray((k, k), dtype=np.int32)
    :return:
    '''
    class_sum = np.sum(conf_mat, axis = 1, dtype = np.float32)
    for i in range(len(class_sum)):
        class_sum[i] = np.float32(conf_mat[i][i])/np.float32(class_sum[i])
    mpca = np.mean(class_sum)*100
    return mpca

def MAC2FLOP(macs, params, module_name = ''):
    macs, params = clever_format([macs, params], "%.3f")
    print('{} MACs: {}  #Params: {}'.format(module_name, macs, params))
    if 'M' in macs:
        flops = float(macs.replace('M', '')) * 2
        flops = str(flops/1000) + 'G'
    elif 'G' in macs:
        flops = str(float(macs.replace('G', '')) * 2) + 'G'
    print('{} GFLOPs: {}  #Params: {}'.format(module_name, flops, params))


class MADmeter(object):
    def __init__(self, T, N):
        super(MADmeter, self).__init__()
        self.T = T
        self.N = N
        self.B = 0
        self.MAD = 0.

    def generate_mask(self, features, field, field_shape = 'rect'):
        if field_shape == 'rect':
            B, T, N, NFB = features.shape
            TN = T*N
            if len(field) == 1: # shape like [3, 3] [5, 5] [7, 7] [9, 9]...
                assert field[0]%2 == 1
                mask = torch.zeros((TN, TN), dtype = torch.bool, device = features.device)
                for i in range(TN):
                    x, y = i//N, i%N
                    for j in range(field[0]):
                        jx = j - field[0] // 2
                        if jx + x >=0:
                            for k in range(field[0]):
                                ky = k - field[0]//2
                                if ky + y >=0:
                                    mask[i][(jx + x)*T + (ky + y)] = True
            elif len(field) == 2 and field[0] == T and field[1] == N: # fully-connected
                mask = torch.ones((TN, TN), dtype = torch.bool, device = features.device)

        elif field_shape == 'dynamic':  # [B, TN, k2+1, NFB]
            B, TN, k2, NFB = features.shape
            mask = torch.zeros((TN*k2, TN*k2), dtype=torch.bool, device=features.device)
            for i in range(TN):
                for j in range(k2 - 1):
                    mask[i*k2, i*k2 + j + 1] = True

            # if len(field) == 2 and field[0] == T and field[1] == N:  # fully-connected
            #     mask = torch.ones((TN, TN), dtype=torch.bool, device=features.device)
            # else: # shape like [3, 3] [5, 5] [7, 7] [9, 9]...
            #     assert field[0]%2 == 1 and field[1]%2 == 1
            #     mask = torch.zeros((TN, TN), dtype = torch.bool, device = features.device)
            #     for i in range(TN):
            #         x, y = i//N, i%N
            #         for j in range(field[0]):
            #             jx = j - field[0] // 2
            #             if jx + x >=0:
            #                 for k in range(field[0]):
            #                     ky = k - field[0]//2
            #                     if ky + y >=0:
            #                         mask[i][(jx + x)*T + (ky + y)] = True

        return mask

    def node_select(self, features, field_shape = 'dynamic'):
        if field_shape == 'dynamic':
            B, TN, k2, NFB = features.shape
            node_selector = torch.zeros((TN*k2, ), dtype=torch.bool, device=features.device)
            for i in range(TN):
                node_selector[i*k2] = True
            return node_selector

    def cal_MAD(self, features, field, field_shape = 'rect'):
        B, T, N, NFB = features.shape

        mask = self.generate_mask(features = features,
                                  field = field,
                                  field_shape = field_shape)
        features = features.view(B, T * N, NFB)
        norm = torch.norm(features, dim = 2).unsqueeze(dim = 2)
        divisor = torch.bmm(norm, norm.transpose(1, 2))
        dist_array = 1. - torch.bmm(features, features.transpose(1, 2)) / (divisor + 1e-8)
        dist_array = dist_array * mask.float()
        # print(mask)
        MAD = (torch.sum(dist_array, dim = 2) / (torch.sum(mask.float(), dim = 1) + 1e-8) ).detach() # B, TN
        # print(MAD)
        if field_shape == 'dynamic':
            node_selector = self.node_select(features.view(B, T, N, NFB), field_shape = 'dynamic')
            batch_MAD = MAD[node_selector[None,:].repeat((B, 1))].view(B, -1)
            self.MAD += torch.sum(torch.mean(batch_MAD, dim = 1), dim = 0)
        else:
            self.MAD += torch.sum(torch.mean(MAD, dim = 1))
        self.B += B

    def output_MAD(self):
        return self.MAD / self.B

# 累计指标类 -- lld
class AccumulateMeter(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.TP_class = {}  # 各类别的TP数量
        self.pred_class = {}    # 各类别的预测数量
        self.true_class = {}    # 各类别标签的数量
        self.TP_total = 0   # 所有的TP数量
        self.precision = {}     # 各类别的准确率
        self.recall = {}    # 各类别的召回率
        self.mAP = 0    # mean average precision
        self.accuracy = 0   # 准确率
        self.MPCA = 0      # Mean per Class Accuracy
        self.count = 0      # 已预测的样本数

        for i in range(num_class):
            self.TP_class[i] = 0
            self.pred_class[i] = 0
            self.true_class[i] = 0
            self.precision[i] = 0
            self.recall[i] = 0

    def accumulate(self, pred, true):
        self.count += len(pred)
        for p, t in zip(pred, true):
            self.pred_class[p] += 1
            self.true_class[t] += 1
            if p == t:
                self.TP_class[p] += 1
                self.TP_total += 1

    def compute(self):
        for cate in self.TP_class:
            if self.pred_class[cate] == 0:
                self.precision[cate] = 0
            else:
                self.precision[cate] = self.TP_class[cate] / self.pred_class[cate]
            if self.true_class[cate] == 0:
                self.recall[cate] = 0
            else:
                self.recall[cate] = self.TP_class[cate] / self.true_class[cate]

        self.accuracy = self.TP_total / self.count

        self.MPCA = sum(self.recall.values()) / self.num_class

# 基于预先相似度的对比损失，只关注0类和1类 - lls
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def get_upper_triangular_elements(self, matrix):
        N = matrix.shape[0]
        mask = torch.triu(torch.ones(N, N), diagonal=1).bool()  # Create a mask for the upper triangular elements
        upper_triangular_elements = matrix[mask]
        return upper_triangular_elements.view(-1)  # Flatten the result into a 1D tensor

    def forward(self, embeddings, labels):
        """
        Computes the contrastive loss.

        Args:
            embeddings: A tensor of shape (N, embedding_dim) containing the embeddings of N samples.
            labels: A tensor of shape (N,) containing the labels of the N samples (0, 1, 2, or 3).

        Returns:
            The contrastive loss.
        """
        # Filter out samples that are not from class 0 or 1
        mask = (labels == 0) | (labels == 1)
        embeddings = embeddings[mask]
        labels = labels[mask]

        N = embeddings.shape[0]
        if N < 2:  # Handle cases where there are fewer than 2 samples from class 0 or 1
            return torch.tensor(0.0)

        # Compute pairwise distances
        distances = F.cosine_similarity(embeddings.unsqueeze(dim=1), embeddings.unsqueeze(dim=0), dim=2)
        distances = (self.get_upper_triangular_elements(distances) + 1.0) / 2.0

        # Create a mask for pairs with the same label
        same_label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        same_label_mask = self.get_upper_triangular_elements(same_label_mask)

        # Compute the loss
        # loss = torch.sum((1 - distances) * same_label_mask) + torch.sum(distances * (~same_label_mask))
        loss = torch.sum(distances * (~same_label_mask))

        # return loss / (N * (N - 1) / 2)  # Average loss
        return loss / sum(~same_label_mask)  # Average loss

if __name__=='__main__':
    # Test MPCA
    # x = np.ndarray((3,3), dtype = np.int32)
    # x[0] = [1, 2, 3]
    # x[1] = [1, 2, 3]
    # x[2] = [2, 2, 3]
    # print(MPCA(x))

    # Test dynamic MAD
    # Mad = MADmeter(10, 12)
    # f = torch.randn((1, 120, 26, 1024))
    # Mad.cal_MAD(f, [1, 25], field_shape = 'dynamic')
    # print(Mad.output_MAD())

    Mad = MADmeter(10, 12)
    f = torch.randn((1, 10, 12, 1024))
    Mad.cal_MAD(f, [10, 12], field_shape='rect')
    print(Mad.output_MAD())



