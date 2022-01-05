import torch
import torch.nn as nn
from tools.conv import GraphConv, SpGAT
from tools.get_graph import GetKnnGraph
from tools.get_feature import FeatureExtractionModule, SharedMLP, knn, spknn, PGCN


class RGA(nn.Module):
    def __init__(self, d_in, k, random_rate, decimation=2, device=torch.device('cuda')):
        super(RGA, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decimation = decimation
        self.knn = GetKnnGraph(k, random_rate)

        self.fc_start = nn.Linear(d_in, 64)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(64, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers

        self.encoder = nn.ModuleList([
            PGCN(64, 64, k, random_rate, device),
            PGCN(64, 128, k, random_rate, device),
            PGCN(128, 256, k, random_rate, device),
            PGCN(256, 512, k, random_rate, device)
        ])

        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())

        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.ReLU()
        )
        self.decoder = nn.ModuleList([
            SharedMLP(1024, 256, **decoder_kwargs),
            SharedMLP(512, 128, **decoder_kwargs),
            SharedMLP(256, 64, **decoder_kwargs),
            SharedMLP(128, 32, **decoder_kwargs)
        ])

        # final semantic prediction
        self.fc_end = nn.Sequential(
            SharedMLP(32, 64, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
            nn.Dropout(),
            SharedMLP(32, 13)
        )
        self.device = device

        self = self.to(device)

    def forward(self, input, model='train'):
        r"""
            Forward pass

            Parameters
            ----------
            input: torch.Tensor, shape (B, N, d_in)
                input points

            Returns
            -------
            torch.Tensor, shape (B, num_classes, N)
                segmentation scores for each point
        """
        N = input.size(1)
        d = self.decimation

        coords = input[..., :3].clone()
        x = self.fc_start(input).transpose(-2, -1).unsqueeze(-1)
        x = self.bn_start(x)  # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []

        permutation = torch.randperm(N)
        coords = coords[:, permutation]
        x = x[:, :, permutation]

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:, :N // decimation_ratio], x, model)
            x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:, :, :N // decimation_ratio].clone()

        # # >>>>>>>>>> ENCODER

        x = self.mlp(x)

        # <<<<<<<<<< DECODER
        for mlp in self.decoder:
            neighbors = spknn(coords[:, :d * N // decimation_ratio].contiguous(),
                              coords[:, :N // decimation_ratio].contiguous(),
                              1)

            neighbors = neighbors.to(self.device)

            extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)

            x_neighbors = torch.gather(x, -2, extended_neighbors)

            x = torch.cat((x_neighbors, x_stack.pop()), dim=1)

            x = mlp(x)

            decimation_ratio //= d

        # >>>>>>>>>> DECODER
        # inverse permutation
        x = x[:, :, torch.argsort(permutation)]

        scores = self.fc_end(x)

        return scores.squeeze(-1)
