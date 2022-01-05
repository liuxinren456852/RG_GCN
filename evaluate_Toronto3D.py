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
from tools.util import cal_loss, IOStream, _init_, calculate_sem_IoU
import sklearn.metrics as metrics
from tools.PointGCN import PointGCN
from tqdm import tqdm
import data_Toronto3D
from plyfile import PlyData, PlyElement

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate(args, io):
    all_true_cls = []
    all_pred_cls = []
    all_true_seg = []
    all_pred_seg = []

    for test_area in range(1, 7):
        test_area = str(test_area)

        if (args.test_area == 'all') or (test_area == args.test_area):
            test_loader = DataLoader(S3DIS(partition='test', num_points=args.num_points, test_area=test_area),
                                     batch_size=args.test_batch_size, shuffle=False, drop_last=False)

            test_dataname = 'L00' + test_area
            out_dir = os.path.join('checkpoints', args.exp_name, 'vis')
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            fout = open(os.path.join(out_dir, test_dataname + '_pred.obj'), 'w')
            fout_gt = open(os.path.join(out_dir, test_dataname + '_gt.obj'), 'w')
            fout_data_label = open(os.path.join(out_dir, test_dataname + '_pred.txt'), 'w')
            fout_gt_label = open(os.path.join(out_dir, test_dataname + '_gt.txt'), 'w')

            # Try to load models
            device = torch.device("cuda" if args.cuda else "cpu")
            model = PointGCN(args).to(device)
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_best_%s.t7' % test_area)))
            model = model.eval()
            total = sum([param.nelement() for param in model.parameters()])
            print("Number of parameter: %.2fM" % (total / 1e6))

            test_acc = 0.0
            count = 0.0
            test_true_cls = []
            test_pred_cls = []
            test_true_seg = []
            test_pred_seg = []
            with torch.no_grad():
                loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=True, ncols=100)
                for index, (data, seg) in loop:
                    start_idx = index * args.test_batch_size

                    data, seg = data.to(device), seg.to(device)
                    BATCH_SIZE = data.size()[0]
                    seg_pred = model(data, model='test')
                    seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                    pred = seg_pred.max(dim=2)[1]
                    seg_np = seg.cpu().numpy()
                    pred_np = pred.detach().cpu().numpy()

                    for b in range(BATCH_SIZE):
                        pts = data[start_idx + b, :, :]
                        label = seg[start_idx + b, :]
                        pts[:, 3:6] *= 255.0
                        pred_label = pred[b, :]
                        for i in range(args.num_points):
                            color = data_Toronto3D.g_label2color[pred_label[i]]
                            color_gt = data_Toronto3D.g_label2color[label[i]]
                            fout.write('v %f %f %f %d %d %d\n' % (pts[i, 0], pts[i, 1], pts[i, 2], color[0], color[1], color[2]))
                            fout_gt.write('v %f %f %f %d %d %d\n' % (pts[i, 0], pts[i, 1], pts[i, 2], color_gt[0], color_gt[1], color_gt[2]))
                            fout_data_label.write('%f %f %f %d %d %d %f %d\n' % (pts[i, 0], pts[i, 1], pts[i, 2], pts[i, 3], pts[i, 4], pts[i, 5], pred[b, i, pred[i]], pred[i]))
                            fout_gt_label.write('%d\n' % (label[i]))

                    test_true_cls.append(seg_np.reshape(-1))
                    test_pred_cls.append(pred_np.reshape(-1))
                    test_true_seg.append(seg_np)
                    test_pred_seg.append(pred_np)

                fout_data_label.close()
                fout_gt_label.close()
                fout.close()
                fout_gt.close()

                test_true_cls = np.concatenate(test_true_cls)
                test_pred_cls = np.concatenate(test_pred_cls)
                test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
                avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
                test_true_seg = np.concatenate(test_true_seg, axis=0)
                test_pred_seg = np.concatenate(test_pred_seg, axis=0)
                test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
                outstr = 'Test :: test area: %s, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (test_area,
                                                                                                        test_acc,
                                                                                                        avg_per_class_acc,
                                                                                                        np.mean(test_ious))
                io.cprint(outstr)
                io.cprint(str(test_ious))
                all_true_cls.append(test_true_cls)
                all_pred_cls.append(test_pred_cls)
                all_true_seg.append(test_true_seg)
                all_pred_seg.append(test_pred_seg)

    if args.test_area == 'all':
        all_true_cls = np.concatenate(all_true_cls)
        all_pred_cls = np.concatenate(all_pred_cls)
        all_acc = metrics.accuracy_score(all_true_cls, all_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(all_true_cls, all_pred_cls)
        all_true_seg = np.concatenate(all_true_seg, axis=0)
        all_pred_seg = np.concatenate(all_pred_seg, axis=0)
        all_ious = calculate_sem_IoU(all_pred_seg, all_true_seg)
        outstr = 'Overall Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (all_acc,
                                                                                         avg_per_class_acc,
                                                                                         np.mean(all_ious))
        io.cprint(outstr)
        io.cprint(str(all_ious))


if __name__ == "__main__":
    # eval settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='semseg', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='S3DIS', metavar='N',
                        choices=['S3DIS'])
    parser.add_argument('--test_area', type=str, default='5', metavar='N',
                        choices=['1', '2', '3', '4', '5', '6', 'all'])
    parser.add_argument('--test_batch_size', type=int, default=4, metavar='test_batch_size',
                        help='Size of batch)')
    parser.add_argument('--model_root', type=str, default='', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--num_points', type=int, default=128,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--k', type=int, default=25, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--random_rate', type=float, default='0.8', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--visu', type=str, default='',
                        help='visualize the model')
    parser.add_argument('--visu_format', type=str, default='ply',
                        help='file format of visualization')
    args = parser.parse_args()

    _init_(args)
    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    evaluate(args, io)
    print('over all')
