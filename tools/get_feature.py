import torch
import torch.nn as nn
from tools.conv import GraphConv
from tools.get_graph import GetKnnGraph


def spknn(x, y, k):
    """ knn serach
    Arguments:
        pos_support - [B,N,3] support points
        pos - [B,M,3] centre of queries
        k - number of neighboors, needs to be > N
    Returns:
        idx - [B,M,k]
        dist2 - [B,M,k] squared distances
    """
    B = x.size(0)
    m, n = x.size(1), y.size(1)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(2, keepdim=True).expand(B, m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(2, keepdim=True).expand(B, n, m).transpose(1, 2)
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    # dist.addmm_(1, -2, x, y.transpose(1, 2))
    distances = dist - 2 * torch.matmul(x, y.transpose(1, 2))
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵

    dist = distances.topk(k=k, dim=-1)[0]
    idx = distances.topk(k=k, dim=-1)[1]

    return idx


def knn(x, k):
    device = torch.device("cuda")
    x = x.transpose(1, 2).to(device)
    inner = -2 * torch.matmul(x.transpose(2, 1), x).to(device)
    xx = torch.sum(x ** 2, dim=1, keepdim=True).to(device)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).to(device)
    dist = pairwise_distance.topk(k=k, dim=-1)[0]
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    knn_out = (idx, dist)

    return knn_out  # idx, dist


def knn_feature(x, k=20):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    idx = knn(x.transpose(1, 2), k=k)[0]  # (batch_size, num_points, k)
    dist = knn(x.transpose(1, 2), k=k)[1]
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    dist = dist.view(batch_size, num_points, k, 1).repeat(1, 1, 1, num_dims)

    # feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    feature = torch.cat((x, feature, feature - x, dist), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, transpose=False, padding_mode='zeros',
                 bn=False, activation_fn=None):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding_mode=padding_mode)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class KnnFeature(nn.Module):
    def __init__(self, d_in, d_out, k):
        super(KnnFeature, self).__init__()

        self.k = k
        self.conv = nn.Sequential(nn.Conv2d(d_in * 4, d_out, kernel_size=1, bias=False),
                                  nn.BatchNorm2d(d_out),
                                  nn.LeakyReLU(negative_slope=0.2))

    def forward(self, features):
        features = knn_feature(features.squeeze(-1), k=self.k)

        return self.conv(features)


class PointFeatureFusion(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors, device):
        super(PointFeatureFusion, self).__init__()

        self.num_neighbors = num_neighbors
        self.knnfeature = KnnFeature(d_in, d_out, num_neighbors)
        # self.mlp = SharedMLP(10, d_out, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.mlp = nn.Sequential(nn.Conv2d(10, d_out, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(d_out),
                                 nn.ReLU())

        self.device = device

    def forward(self, coords, features, knn_output):
        features = self.knnfeature(features)

        # finding neighboring points
        idx, dist = knn_output
        idx, dist = idx.to(self.device), dist.to(self.device)
        B, N, K = idx.size()
        # features = features.squeeze(-1)
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx)  # shape (B, 3, N, K)

        concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)
        ), dim=-3).to(self.device)

        return torch.cat((
            self.mlp(concat),
            features
        ), dim=-3)


class PointCrossFeature(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors, device):
        super(PointCrossFeature, self).__init__()

        self.num_neighbors = num_neighbors
        self.knnfeature = KnnFeature(d_in, d_out, num_neighbors)
        # self.mlp = SharedMLP(10, d_out, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.mlp_featureToPoint = nn.Sequential(nn.Conv2d(d_out, 3, kernel_size=1, bias=False),
                                                nn.BatchNorm2d(3),
                                                nn.ELU())
        self.mlp_PointToFeature = nn.Sequential(nn.Conv2d(13, d_in, kernel_size=1, bias=False),
                                                nn.BatchNorm2d(d_in),
                                                nn.ELU())
        self.mlp_out_p = nn.Sequential(nn.Conv2d(13, int(d_out/2), kernel_size=1, bias=False),
                                       nn.BatchNorm2d(int(d_out/2)),
                                       nn.ELU())
        self.mlp_out_f = nn.Sequential(nn.Conv2d(d_out+d_in, int(d_out/2), kernel_size=1, bias=False),
                                       nn.BatchNorm2d(int(d_out/2)),
                                       nn.ELU())

        self.device = device

    def forward(self, coords, features, knn_output):
        features_knn = self.knnfeature(features)
        features_offset = self.mlp_featureToPoint(features_knn)

        # finding neighboring points
        idx, dist = knn_output
        idx, dist = idx.to(self.device), dist.to(self.device)
        B, N, K = idx.size()
        # features = features.squeeze(-1)
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)
        point_e = features_offset + extended_coords
        neighbors = torch.gather(extended_coords, 2, extended_idx)  # shape (B, 3, N, K)

        concat_point = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3),
            point_e
        ), dim=-3).to(self.device)

        point_offset = self.mlp_PointToFeature(concat_point)
        features_e = point_offset + features
        concat_feature = torch.cat((
            features_knn,
            features_e
        ), dim=-3).to(self.device)

        out_point = self.mlp_out_p(concat_point)
        out_feature = self.mlp_out_f(concat_feature)

        return torch.cat((
            out_point,
            out_feature
        ), dim=-3)


class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )
        # self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.mlp = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ELU())

    def forward(self, x):
        scores = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        att_features = torch.sum(scores * x, dim=-1, keepdim=True)  # shape (B, d_in, N, 1)
        max_features = x.max(dim=-1, keepdim=True)[0]
        mean_features = x.mean(dim=-1, keepdim=True)

        return self.mlp(max_features)


class CrossPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrossPooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )
        # self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.mlp = nn.Sequential(nn.Conv2d(3 * in_channels, out_channels, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ELU())

    def forward(self, x):
        scores = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        att_features = torch.sum(scores * x, dim=-1, keepdim=True)  # shape (B, d_in, N, 1)
        # print(att_features.shape)
        max_features = x.max(dim=-1, keepdim=True)[0]
        # print(max_features.shape)
        mean_features = x.mean(dim=-1, keepdim=True)
        # print(mean_features.shape)
        features = torch.cat((att_features, max_features, mean_features), dim=1)

        return self.mlp(features)


class FeatureExtractionModule(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors, device):
        super(FeatureExtractionModule, self).__init__()

        self.num_neighbors = num_neighbors
        self.lse1 = PointFeatureFusion(d_in, d_out // 2, num_neighbors, device)
        self.pool2 = CrossPooling(d_out, d_out)

    def forward(self, coords, features):
        knn_output = knn(coords.contiguous(), self.num_neighbors)
        x = self.lse1(coords, features, knn_output)
        x = self.pool2(x)
        return x


class spFeatureExtractionModule(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors, device):
        super(spFeatureExtractionModule, self).__init__()

        self.num_neighbors = num_neighbors
        self.lse = PointCrossFeature(d_in, d_out, num_neighbors, device)
        self.pool = AttentivePooling(d_out, d_out)

    def forward(self, coords, features):
        knn_output = knn(coords.contiguous(), self.num_neighbors)
        x = self.lse(coords, features, knn_output)
        x = self.pool(x)
        return x


class PGCN(nn.Module):
    def __init__(self, d_in, d_out, k, random_rate, device):
        super(PGCN, self).__init__()
        self.k = k
        self.random_rate = random_rate
        self.device = device

        self.conv1 = FeatureExtractionModule(d_in, d_out, int(self.k * self.random_rate), self.device)
        self.conv2 = GraphConv(d_out, d_out)
        self.knn = GetKnnGraph(k=self.k, random_rate=self.random_rate)
        self.knn_test = GetKnnGraph(k=self.k, random_rate=self.random_rate, isTrain=False)

    def forward(self, points, features, model='train'):
        if model == 'train':
            edge_index = self.knn(points.transpose(1, 2))
        elif model == 'test':
            edge_index = self.knn_test(points.transpose(1, 2))
        x = self.conv1(points, features)
        x = self.conv2(x, edge_index)
        return x
