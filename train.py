from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tools.data import S3DIS
import numpy as np
from torch.utils.data import DataLoader
from tools.util import cal_loss, IOStream, _init_, calculate_sem_IoU, Load_data, Load_model, focal_loss
import sklearn.metrics as metrics
from tools.PointGCN import PointGCN
from tqdm import tqdm

# from model import DGCNN_semseg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train(args, io):
    train_data, test_data, clas = Load_data(args)
    train_loader = DataLoader(train_data,
                              num_workers=0,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(test_data,
                             num_workers=0,
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=True)

    device = torch.device("cuda")
    model = Load_model(args, device)
    model = nn.DataParallel(model)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    # opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    alphalist = train_data.get_alphalist()
    alphalist = alphalist/np.sum(alphalist)
    alphalist = (0.5-alphalist)**2
    print(alphalist)
    # alphalist = 1 / (1 + np.log(alphalist))
    criterion = focal_loss(alphalist)
    best_test_iou = 0

    print('---------------now training---------------')
    for epoch in range(args.epochs):
        ###################
        ###### Train ######
        ###################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True, ncols=100)
        for index, (data, seg) in loop:
            loop.set_description('Train epoch: {}/{}'.format(epoch + 1, args.epochs))
            data, seg = data.to(device), seg.to(device)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, clas), seg.view(-1, 1).squeeze())
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]  # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()  # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))  # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))  # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
        scheduler.step()
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg, clas)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch + 1,
                                                                                                  train_loss * 1.0 / count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious))
        io.cprint(outstr)
        torch.save(model.state_dict(), 'checkpoints/%s/models/model_epoch_%s.t7' % (args.exp_name, epoch + 1))
        ####################
        # Test
        ####################
        # test_loss = 0.0
        # count = 0.0
        # model.eval()
        # test_true_cls = []
        # test_pred_cls = []
        # test_true_seg = []
        # test_pred_seg = []
        # all13 = [0 for _ in range(clas)]
        # pre13 = [0 for _ in range(clas)]
        # acc13 = [0 for _ in range(clas)]
        # with torch.no_grad():
        #     loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=True, ncols=100)
        #     for index, (data, seg) in loop:
        #         loop.set_description('Test epoch:  {}/{}'.format(epoch + 1, args.epochs))
        #         data, seg = data.to(device), seg.to(device)
        #         batch_size = data.size()[0]
        #         opt.zero_grad()
        #         seg_pred = model(data, model='test')
        #         # seg_pred = model(data)
        #         seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        #         loss = criterion(seg_pred.view(-1, clas), seg.view(-1, 1).squeeze())
        #         pred = seg_pred.max(dim=2)[1]
        #         count += batch_size
        #         test_loss += loss.item() * batch_size
        #         seg_np = seg.cpu().numpy()
        #         pred_np = pred.detach().cpu().numpy()
        #         test_true_cls.append(seg_np.reshape(-1))
        #         test_pred_cls.append(pred_np.reshape(-1))
        #         test_true_seg.append(seg_np)
        #         test_pred_seg.append(pred_np)
        #     test_true_cls = np.concatenate(test_true_cls)
        #     test_pred_cls = np.concatenate(test_pred_cls)
        #     test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        #     avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        #     test_true_seg = np.concatenate(test_true_seg, axis=0)
        #     test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        #     test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg, clas)
        #     outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch + 1,
        #                                                                                           test_loss * 1.0 / count,
        #                                                                                           test_acc,
        #                                                                                           avg_per_class_acc,
        #                                                                                           np.mean(test_ious))
        #     io.cprint(outstr)
        #     for l in range(clas):
        #         all13[l] = np.sum((test_true_cls == l))
        #         pre13[l] = np.sum((test_pred_cls == l) & (test_true_cls == l))
        #         acc13[l] = pre13[l] / all13[l]
        #     io.cprint(str(acc13))
        #
        #     print(test_ious)
        #
        #     if np.mean(test_ious) >= best_test_iou:
        #         best_test_iou = np.mean(test_ious)
        #         torch.save(model.state_dict(),
        #                    'checkpoints/%s/models/model_best_%s.t7' % (args.exp_name, args.test_area))
        #

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='semseg000', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='S3DIS', metavar='N',
                        choices=['S3DIS', 'Toronto3D'])
    parser.add_argument('--exp_method', type=str, default='less', metavar='N',
                        choices=['normal', 'less'])
    parser.add_argument('--model_name', type=str, default='PointGCN', metavar='N',
                        choices=['PointGCN', 'RGA'])
    parser.add_argument('--test_area', type=str, default='3', metavar='N')
    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=4, metavar='test_batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--num_points', type=int, default=96,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--k', type=int, default=25, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--random_rate', type=float, default=0.8, metavar='N',
                        help='000')
    args = parser.parse_args()

    # torch.cuda.set_device(1)
    _init_(args)
    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    io.cprint('Using GPU: ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')

    train(args, io)
    print('over all')
