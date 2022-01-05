import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
from tools.data import S3DIS
from torch.utils.data import DataLoader
from tools.data import Toronto3D
from tools.PointGCN import PointGCN
from tools.RGA import RGA


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.abspath(BASE_DIR))


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('../checkpoints'):
        os.makedirs('../checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system(
        'copy ' + BASE_DIR + r'\PointGCN.py ' + ROOT_DIR + r'\checkpoints' + '\\' + args.exp_name + r'\\PointGCN.py')
    os.system('copy ' + BASE_DIR + r'\conv.py ' + ROOT_DIR + r'\checkpoints' + '\\' + args.exp_name + r'\\conv.py')
    os.system('copy ' + BASE_DIR + r'\data.py ' + ROOT_DIR + r'\checkpoints' + '\\' + args.exp_name + r'\\data.py')
    os.system(
        'copy ' + BASE_DIR + r'\get_graph.py ' + ROOT_DIR + r'\checkpoints' + '\\' + args.exp_name + r'\\get_graph.py')
    os.system(
        'copy ' + BASE_DIR + r'\get_feature.py ' + ROOT_DIR + r'\checkpoints' + '\\' + args.exp_name + r'\\get_feature.py')
    os.system('copy ' + ROOT_DIR + r'\train.py ' + ROOT_DIR + r'\checkpoints' + '\\' + args.exp_name + r'\\train.py')


def calculate_sem_IoU(pred_np, seg_np, clas):
    I_all = np.zeros(clas)
    U_all = np.zeros(clas)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(clas):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all


class focal_loss(nn.Module):
    def __init__(self, alpha, gamma=2, num_classes=13, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(focal_loss, self).__init__()
        self.size_average = size_average
        assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
        print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
        self.alpha = torch.Tensor(alpha)
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)  # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def Load_data(args):
    if args.dataset == 'S3DIS':
        clas = 13
        if args.exp_method == 'normal':
            train_data = S3DIS(partition='train', num_points=args.num_points, test_area=args.test_area)
            test_data = S3DIS(partition='test', num_points=args.num_points, test_area=args.test_area)
        elif args.exp_method == 'less':
            train_data = S3DIS(partition='test', num_points=args.num_points, test_area=args.test_area)
            test_data = S3DIS(partition='train', num_points=args.num_points, test_area=args.test_area)
        else:
            print('please choose true exp_method')

    elif args.dataset == 'Toronto3D':
        clas = 8
        if args.exp_method == 'normal':
            train_data = Toronto3D(partition='train', num_points=args.num_points, test_area=args.test_area)
            test_data = Toronto3D(partition='test', num_points=args.num_points, test_area=args.test_area)
        elif args.exp_method == 'less':
            train_data = Toronto3D(partition='test', num_points=args.num_points, test_area=args.test_area)
            test_data = Toronto3D(partition='train', num_points=args.num_points, test_area=args.test_area)
        else:
            print('please choose true exp_method')

    else:
        print('please choose true dataset')

    return train_data, test_data, len(train_data.get_alphalist())


def Load_model(args, device):
    if args.model_name == 'PointGCN':
        model = PointGCN(args).to(device)
    elif args.model_name == 'RGA':
        model = RGA(9, args.k, args.random_rate)
    else:
        print('please choose true model')

    return model

