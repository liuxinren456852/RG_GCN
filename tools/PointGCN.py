import torch
import torch.nn as nn
from tools.conv import GraphConv, SpGAT
from tools.get_graph import GetKnnGraph
from tools.get_feature import FeatureExtractionModule, spFeatureExtractionModule


class PointGCN(nn.Module):
    def __init__(self, args):
        super(PointGCN, self).__init__()
        self.args = args
        self.k = args.k
        self.random_rate = args.random_rate
        self.device = torch.device("cuda")

        self.knn = GetKnnGraph(k=self.k, random_rate=self.random_rate)
        self.knn_test = GetKnnGraph(k=self.k, random_rate=self.random_rate, isTrain=False)

        self.bn01 = nn.Sequential(nn.BatchNorm2d(64),
                                  nn.ReLU())
        self.bn02 = nn.Sequential(nn.BatchNorm2d(64),
                                  nn.ReLU())
        self.bn03 = nn.Sequential(nn.BatchNorm2d(64),
                                  nn.ReLU())
        self.bn04 = nn.Sequential(nn.BatchNorm2d(64),
                                  nn.ReLU())
        self.bn05 = nn.Sequential(nn.BatchNorm2d(64),
                                  nn.ReLU())
        self.bn06 = nn.Sequential(nn.BatchNorm2d(64),
                                  nn.ReLU())
        self.bn07 = nn.Sequential(nn.BatchNorm2d(64),
                                  nn.ReLU())
        self.bn08 = nn.Sequential(nn.BatchNorm2d(64),
                                  nn.ReLU())

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)

        self.rl = nn.ELU()

        self.fc_start = nn.Linear(9, 64)
        self.bn_start = nn.Sequential(nn.BatchNorm2d(64),
                                      nn.ReLU())

        # self.conv00 = PositionMaxPoolingModule(64, int(self.k * self.random_rate))
        self.conv0 = FeatureExtractionModule(16, 64, int(self.k * self.random_rate), self.device)

        self.conv1 = spFeatureExtractionModule(64, 256, int(self.k * self.random_rate), self.device)
        self.conv2 = GraphConv(256, 64)
        # self.conv2 = SpGAT(256, 64, 64, 0.5, 1, 2)
        self.conv3 = GraphConv(64, 64)

        self.conv4 = FeatureExtractionModule(64, 256, int(self.k * self.random_rate), self.device)
        # self.conv4 = FeatureMaxPoolingModule(64, 256, int(self.k * self.random_rate))
        self.conv5 = GraphConv(256, 64)
        self.conv6 = GraphConv(64, 64)

        self.conv7 = FeatureExtractionModule(64, 256, int(self.k * self.random_rate), self.device)
        # self.conv7 = FeatureMaxPoolingModule(64, 256, int(self.k * self.random_rate))
        self.conv8 = GraphConv(256, 64)
        self.conv9 = GraphConv(64, 64)

        self.conv70 = FeatureExtractionModule(64, 256, int(self.k * self.random_rate), self.device)
        # self.conv7 = FeatureMaxPoolingModule(64, 256, int(self.k * self.random_rate))
        self.conv80 = GraphConv(256, 64)
        self.conv90 = GraphConv(64, 64)

        self.conv10 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                    self.bn1,
                                    nn.ReLU())
        self.conv11 = nn.Sequential(nn.Conv1d(1152, 512, kernel_size=1, bias=False),
                                    self.bn3,
                                    nn.ReLU())
        self.conv12 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                    self.bn4,
                                    nn.ReLU())
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv13 = nn.Conv1d(256, 13, kernel_size=1, bias=False)

    def forward(self, data, model='train'):
        num_points = data.size(1)
        points = data[..., :3].clone()
        if model == 'train':
            edge_index = self.knn(points.transpose(1, 2))
        elif model == 'test':
            edge_index = self.knn_test(points.transpose(1, 2))
        # print(edge_index.shape)

        # x0 = self.conv00(points).squeeze(-1)

        x = self.fc_start(data).transpose(-2, -1).unsqueeze(-1)
        x = self.bn_start(x)  # shape (B, 64, N, 1)

        # x = self.conv0(points, x)
        # x0 = x.squeeze(-1)

        x = self.conv1(points, x)  # shape (B, 64, N, 1) --> (B, 256, N, 1)
        # print(0, x.shape)
        # x = x.squeeze(-1).transpose(1, 2)
        x = self.conv2(x, edge_index)  # shape (B, 256, N, 1) --> (B, 64, N, 1)
        # x = self.bn01(x)
        # x = self.rl(x)
        # print(x.shape)
        # print(edge_index.shape)
        x = self.conv3(x, edge_index)  # shape (B, 256, N, 1) --> (B, 64, N, 1)
        # x = self.bn02(x)
        x1 = x.squeeze(-1)  # shape (B, 64, N, 1)

        x = self.conv4(points, x)  # shape (B, 64, N, 1) --> (B, 256, N, 1)
        x = self.conv5(x, edge_index)  # shape (B, 256, N, 1) --> (B, 64, N, 1)
        # x = self.bn03(x)
        # x = self.rl(x)
        x = self.conv6(x, edge_index)  # shape (B, 64, N, 1) --> (B, 64, N, 1)
        # x = self.bn04(x)
        x2 = x.squeeze(-1)  # shape (B, 64, N, 1)

        x = self.conv7(points, x)  # shape (B, 64, N, 1) --> (B, 256, N, 1)
        x = self.conv8(x, edge_index)  # shape (B, 256, N, 1) --> (B, 64, N, 1)
        # x = self.bn05(x)
        # x = self.rl(x)
        x = self.conv9(x, edge_index)  # shape (B, 64, N, 1) --> (B, 64, N, 1)
        # print(x[0,0,0,0])
        # x = self.bn06(x)
        x3 = x.squeeze(-1)  # shape (B, 64, N, 1)

        x = self.conv70(points, x)  # shape (B, 64, N, 1) --> (B, 256, N, 1)
        x = self.conv80(x, edge_index)  # shape (B, 256, N, 1) --> (B, 64, N, 1)
        # x = self.bn05(x)
        # x = self.rl(x)
        x = self.conv90(x, edge_index)  # shape (B, 64, N, 1) --> (B, 64, N, 1)
        # x = self.bn06(x)
        x4 = x.squeeze(-1)  # shape (B, 64, N, 1)

        x = torch.cat((x1, x2), dim=1)  # shape (B, 256, N)
        x = self.conv10(x)
        x = x.max(dim=-1, keepdim=True)[0]
        x = x.repeat(1, 1, num_points)
        x = torch.cat((x, x1, x2), dim=1)

        x = self.conv11(x)  # shape (B, 1024, N) --> (B, 512, N)
        x = self.conv12(x)  # shape (B, 512, N) --> (B, 256, N)
        # x = self.dp1(x)
        x = self.conv13(x)  # shape (B, 256, N) --> (B, 13, N)

        return x
