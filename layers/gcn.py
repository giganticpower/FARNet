###################################################################
# File Name: gcn.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Fri 07 Sep 2018 01:16:31 PM CST
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from layers.layers1 import GraphAttentionLayer
from layers.aggcn import GraphConvLayer

class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, A):
        x = torch.bmm(A, features)
        return x


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim * 2, out_dim))

        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.agg = agg()

    def forward(self, features, A):
        b, n, d = features.shape
        assert (d == self.in_dim)
        agg_feats = self.agg(features, A)
        cat_feats = torch.cat([features, agg_feats], dim=2)
        out = torch.einsum('bnd,df->bnf', (cat_feats, self.weight))
        out = F.relu(out + self.bias)
        return out


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

        self.classifier1 = nn.Sequential(
            nn.Linear(32, nclass),
            nn.PReLU(nclass),
            nn.Linear(nclass, 2))

    def forward(self, x, adj, one_hop_idcs):
        B, N, D = x.shape

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))

        k1 = one_hop_idcs.size(-1)
        dout = x.size(-1)
        edge_feat = torch.zeros(B, k1, dout).cuda()
        for b in range(B):
            edge_feat[b, :, :] = x[b, one_hop_idcs[b]]
        edge_feat = edge_feat.view(-1, dout)
        pred = self.classifier1(edge_feat)

        return pred


class GCN(nn.Module):
    def __init__(self, input, output):
        super(GCN, self).__init__()
        # self.bn0 = nn.BatchNorm1d(input, affine=False)
        # self.conv1 = GraphConv(input, 512, MeanAggregator)
        # self.conv2 = GraphConv(512, 256, MeanAggregator)
        # self.conv3 = GraphConv(256, 128, MeanAggregator)
        # self.conv4 = GraphConv(128, 64, MeanAggregator)

        # Dgat
        self.bn01 = nn.BatchNorm1d(input, affine=False)
        self.conv11 = GraphConv(input, 512, MeanAggregator)
        self.conv21 = GraphConv(512, 256, MeanAggregator)
        self.conv31 = GraphConv(256, 128, MeanAggregator)
        self.conv41 = GraphConv(128, 64, MeanAggregator)

        # self.conv1 = GraphConvLayer(input, 512)
        # self.conv2 = GraphConvLayer(512, 256)
        # self.conv3 = GraphConvLayer(256, 128)
        # self.conv4 = GraphConvLayer(128, 64)

        # self.W = nn.Parameter(torch.empty(size=(600, 12)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.classifier = nn.Sequential(
            nn.Linear(64, output),
            nn.PReLU(output),
            nn.Linear(output, 2))

    def forward(self, x, A, one_hop_idcs, train=True):
        # data normalization l2 -> bn
        B, N, D = x.shape

        x = x.view(-1, D)
        x = self.bn01(x)
        x = x.view(B, N, D)

        x = self.conv11(x, A)
        x = self.conv21(x, A)
        x = self.conv31(x, A)
        x = self.conv41(x, A)
        k1 = one_hop_idcs.size(-1)
        dout = x.size(-1)
        edge_feat = torch.zeros(B, k1, dout).cuda()
        for b in range(B):
            edge_feat[b, :, :] = x[b, one_hop_idcs[b]]
        edge_feat = edge_feat.view(-1, dout)
        pred = self.classifier(edge_feat)

        # shape: (B*k1)x2
        return pred
