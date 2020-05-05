import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import csv
from tqdm import tqdm
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GatedGraphConv, GATConv
from numpy.random import default_rng
from pytorchtools import EarlyStopping

### This file contains graph models that we used ###


############# Graph Convolution ##############
class GCNInferenceNetwork(nn.Module):
    def __init__(self, d=1):
        super(GCNInferenceNetwork, self).__init__()
        self.conv1 = GCNConv(d, 16)
        self.conv2 = GCNConv(16, 16)
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, graph):
        x, edges, y, gene_idx = graph.x, graph.edge_index, graph.y, graph.gene_idx_

        # 2 layer convolution on the whole graph
        x = F.relu(self.conv1(x, edges))
        x = F.dropout(x, p=0.4)
        x = F.relu(self.conv2(x, edges))

        # isolate gene of interest, and feed into MLP for regression
        temp_x = x[gene_idx[0]].view(1, 16)
        for i in range(y.size(0) - 1):
            temp_x = torch.cat((temp_x, x[gene_idx[0] + (i + 1) *
                                          int(x.size()[0] / graph.num_graphs)].view(1, 16)))

        temp_x = F.relu(self.fc1(temp_x))
        temp_x = self.fc2(temp_x)

        return temp_x


class MultiGCNInferenceNetwork2(nn.Module):
    def __init__(self, d=1):
        super(MultiGCNInferenceNetwork2, self).__init__()
        self.conv1 = GCNConv(d, 16)
        self.conv2 = GCNConv(16, 16)
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, graph):
        x, edges, y, idx = graph.x, graph.edge_index, graph.y, graph.gene_idx_

        # 2 layer convolution on the whole graph
        x = F.relu(self.conv1(x, edges))
        x = F.dropout(x, p=0.4)
        x = F.relu(self.conv2(x, edges))

        # isolate genes of interest, and feed into MLP for regression
        temp_x = x[idx[0]]

        for i in range(int(y.size(0) / len(idx[0]) - 1)):
            new_idx = idx[0] + (i + 1) * int(x.size()[0] / graph.num_graphs)
            temp_x = torch.cat((temp_x, x[new_idx]))

        temp_x = F.relu(self.fc1(temp_x))
        temp_x = self.fc2(temp_x)

        return temp_x


class MultiGCNInferenceNetwork(nn.Module):
    def __init__(self, d=1):
        super(MultiGCNInferenceNetwork, self).__init__()
        self.conv1 = GCNConv(d, 16)
        self.conv2 = GCNConv(16, 16)

        # create ten heads for ten regression tasks
        self.fc1_1, self.fc2_1 = nn.Linear(16, 8), nn.Linear(8, 1)
        self.fc1_2, self.fc2_2 = nn.Linear(16, 8), nn.Linear(8, 1)
        self.fc1_3, self.fc2_3 = nn.Linear(16, 8), nn.Linear(8, 1)
        self.fc1_4, self.fc2_4 = nn.Linear(16, 8), nn.Linear(8, 1)
        self.fc1_5, self.fc2_5 = nn.Linear(16, 8), nn.Linear(8, 1)
        self.fc1_6, self.fc2_6 = nn.Linear(16, 8), nn.Linear(8, 1)
        self.fc1_7, self.fc2_7 = nn.Linear(16, 8), nn.Linear(8, 1)
        self.fc1_8, self.fc2_8 = nn.Linear(16, 8), nn.Linear(8, 1)
        self.fc1_9, self.fc2_9 = nn.Linear(16, 8), nn.Linear(8, 1)

    def forward(self, graph):
        x, edges, y = graph.x, graph.edge_index, graph.y

        # 2 layer convolution on the whole graph
        x = F.relu(self.conv1(x, edges))
        x = F.dropout(x, p=0.4)
        x = F.relu(self.conv2(x, edges))

        # temp_lis will have the shape[n_genes, n_batchs, n_embedding]
        temp_lis = []
        for i in range(9):
            # idx 1421 is the first target gene
            current_gene = 1421 + i
            temp_x = x[current_gene].view(1, 16)

            # y.size() has the form [n_targets*n_batch, 1]
            for j in range(int(y.size(0) / 9) - 1):
                temp_x = torch.cat((temp_x, x[current_gene + (j + 1) * 1430].view(1, 16)))
            temp_lis.append(temp_x)

        # build ten heads for ten different regression
        pred_1 = self.fc2_1(F.relu(self.fc1_1(temp_lis[0])))
        pred_2 = self.fc2_2(F.relu(self.fc1_2(temp_lis[1])))
        pred_3 = self.fc2_3(F.relu(self.fc1_3(temp_lis[2])))
        pred_4 = self.fc2_4(F.relu(self.fc1_4(temp_lis[3])))
        pred_5 = self.fc2_5(F.relu(self.fc1_5(temp_lis[4])))
        pred_6 = self.fc2_6(F.relu(self.fc1_6(temp_lis[5])))
        pred_7 = self.fc2_7(F.relu(self.fc1_7(temp_lis[6])))
        pred_8 = self.fc2_8(F.relu(self.fc1_8(temp_lis[7])))
        pred_9 = self.fc2_9(F.relu(self.fc1_9(temp_lis[8])))

        return pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, pred_8, pred_9
############# Graph Convolution ##############


############# Gated Graph Convolution ##############
class MultiGGCN(nn.Module):
    def __init__(self, d=1):
        super(MultiGGCN, self).__init__()
        self.conv1 = GatedGraphConv(16, 2)
        self.conv2 = GatedGraphConv(16, 2)
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, graph):
        x, edges, y, idx = graph.x, graph.edge_index, graph.y, graph.gene_idx_

        # 2 layer convolution on the whole graph
        x = F.relu(self.conv1(x, edges))
        x = F.dropout(x, p=0.4)
        x = F.relu(self.conv2(x, edges))

        # isolate genes of interest, and feed into MLP for regression
        temp_x = x[idx[0]]

        for i in range(int(y.size(0) / len(idx[0]) - 1)):
            new_idx = idx[0] + (i + 1) * int(x.size()[0] / graph.num_graphs)
            temp_x = torch.cat((temp_x, x[new_idx]))

        temp_x = F.relu(self.fc1(temp_x))
        temp_x = F.dropout(temp_x, p=0.4)
        temp_x = self.fc2(temp_x)

        return temp_x


class GGCN(nn.Module):
    def __init__(self, d=1):
        super(GGCN, self).__init__()
        self.conv1 = GatedGraphConv(16, 2)
        self.conv2 = GatedGraphConv(16, 2)
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, graph):
        x, edges, y, gene_idx = graph.x, graph.edge_index, graph.y, graph.gene_idx_

        # 2 layer convolution on the whole graph
        x = F.relu(self.conv1(x, edges))
        x = F.dropout(x, p=0.4)
        x = F.relu(self.conv2(x, edges))

        # isolate gene of interest, and feed into MLP for regression
        temp_x = x[gene_idx[0]].view(1, 16)
        for i in range(y.size(0) - 1):
            temp_x = torch.cat((temp_x, x[gene_idx[0] + (i + 1) *
                                          int(x.size()[0] / graph.num_graphs)].view(1, 16)))

        temp_x = F.relu(self.fc1(temp_x))
        temp_x = F.dropout(temp_x, p=0.4)
        temp_x = self.fc2(temp_x)

        return temp_x
############# Gated Graph Convolution ##############


############# GAT ##############
class GATInferenceNetwork(nn.Module):
    def __init__(self, d=1):
        super(GATInferenceNetwork, self).__init__()
        self.conv1 = GATConv(d, 16)
        self.conv2 = GATConv(16, 16)
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, graph):
        x, edges, y, gene_idx = graph.x, graph.edge_index, graph.y, graph.gene_idx_

        # 2 layer convolution on the whole graph
        x = F.relu(self.conv1(x, edges))
        x = F.dropout(x, p=0.4)
        x = F.relu(self.conv2(x, edges))

        # isolate gene of interest, and feed into MLP for regression
        temp_x = x[gene_idx[0]].view(1, 16)
        for i in range(y.size(0) - 1):
            temp_x = torch.cat((temp_x, x[gene_idx[0] + (i + 1) *
                                          int(x.size()[0] / graph.num_graphs)].view(1, 16)))

        temp_x = F.relu(self.fc1(temp_x))
        temp_x = F.dropout(temp_x, p=0.4)
        temp_x = self.fc2(temp_x)
        
        return temp_x

class MultiGATInferenceNetwork(nn.Module):
    def __init__(self, d=1):
        super(MultiGATInferenceNetwork, self).__init__()
        self.conv1 = GatedGraphConv(d, 16)
        self.conv2 = GatedGraphConv(16, 16)
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, graph):
        x, edges, y, idx = graph.x, graph.edge_index, graph.y, graph.gene_idx_

        # 2 layer convolution on the whole graph
        x = F.relu(self.conv1(x, edges))
        x = F.dropout(x, p=0.4)
        x = F.relu(self.conv2(x, edges))

        # isolate genes of interest, and feed into MLP for regression
        temp_x = x[idx[0]]

        for i in range(int(y.size(0) / len(idx[0]) - 1)):
            new_idx = idx[0] + (i + 1) * int(x.size()[0] / graph.num_graphs)
            temp_x = torch.cat((temp_x, x[new_idx]))

        temp_x = F.relu(self.fc1(temp_x))
        temp_x = F.dropout(temp_x, p=0.4)
        temp_x = self.fc2(temp_x)

        return temp_x
############# GAT ##############