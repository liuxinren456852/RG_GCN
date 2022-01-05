import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Conv2d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def batched_index_select(x, idx):
    idx = idx.clone()
    batch_size, num_dims, num_vertices = x.shape[:3]
    k = idx.shape[-1]
    idx_base = torch.arange(0, batch_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')).view(-1, 1, 1) * num_vertices
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.reshape(batch_size * num_vertices, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def norm_layer(norm, nc):
    norm = norm.lower()
    if norm == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


class BasicConv(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=bias))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
            if drop > 0:
                m.append(nn.Dropout2d(drop))

        super(BasicConv, self).__init__(*m)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', norm='batch', bias=True):
        super(GraphConv, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index):
        edge_index = edge_index.clone()
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index):
        k = x.shape[-1]
        maxvalue = []
        for i in range(k):
            xx = x[..., i]
            x_i = batched_index_select(xx, edge_index[1])
            x_j = batched_index_select(xx, edge_index[0])
            max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
            maxvalue.append(max_value)
        max_value = torch.stack(maxvalue, dim=-1)
        return max_value


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = []
        for i in range(indices.shape[0]):
            aa = torch.sparse_coo_tensor(indices[i], values[i], shape)
            a.append(aa)
        a = torch.stack(a)
        assert not torch.isnan(a).any()
        assert not torch.isnan(b).any()

        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        ctx.B = b.shape[0]

        output = []
        for i in range(indices.shape[0]):
            re = torch.matmul(a[i], b[i])
            output.append(re)
        output = torch.stack(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # print(grad_output.shape)
        # print(a, b)
        # print(a.shape, b.shape)
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.transpose(1, 2))
            edge_idx = a._indices()[0, :] * ctx.B + a._indices()[1, :] * ctx.N + a._indices()[2, :] * ctx.N
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = []
            a = a.transpose(1, 2)
            # print(a)
            # print(a[1].shape)
            # print(grad_output[1].shape)
            for i in range(a.shape[0]):
                # print(i)
                grad_bb = torch.matmul(a[i].t(), grad_output[i])
                # print('bb', grad_bb.shape)
                # grad_bb = a[i].matmul(grad_output[i])
                grad_b.append(grad_bb)
            grad_b = torch.stack(grad_b)
        # print(grad_values.shape, grad_b.shape)
        grad_values = grad_values.view(ctx.B, -1)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'
        N = input.size()[1]
        B = input.size()[0]
        edge = adj.reshape(2, B, -1).transpose(0, 1)

        h = torch.matmul(input, self.W)  # B N Cin x Cin Cout -> B N Cout
        # h: B x N x Cout
        if torch.isnan(h).any():
            print(h)
            print(self.W)

        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = []
        for i in range(h.shape[0]):
            edge_h.append(torch.cat((h[i, edge[i, 0, :], :], h[i, edge[i, 1, :], :]), dim=-1))  # concat B NxK Cout
        edge_h = torch.stack(edge_h).transpose(1, 2)  # B Nxk 2Cout -> B 2Cout Nxk
        # edge_h: B x 2Cout x Nxk

        edge_e = torch.exp(-self.leakyrelu(self.a.matmul(edge_h).squeeze()))  # self.a == (1 x 2Cout)
        # edge_e: B x Nxk
        assert not torch.isnan(edge_e).any()

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(B, N, 1), device=dv))
        e_rowsum += 0.001  # 防止出现nan
        # e_rowsum: B x N x 1
        if torch.isnan(e_rowsum).any():
            print(e_rowsum)

        assert not torch.isnan(e_rowsum).any()

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime0 = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        #  h_prime.shape: B x N x Cout
        if torch.isnan(h_prime0).any():
            print(h_prime0)

        assert not torch.isnan(h_prime0).any()

        h_prime = h_prime0.div(e_rowsum)
        if torch.isnan(h_prime).any():
            print(e_rowsum)
            print(h_prime0)

        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            # print(F.elu(h_prime).shape)
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=True)

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))

        x = x.transpose(1, 2).unsqueeze(-1)
        # return -F.log_softmax(x, dim=-1)
        return x
