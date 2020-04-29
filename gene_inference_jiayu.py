import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import csv
from tqdm import tqdm
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from numpy.random import default_rng
from pytorchtools import EarlyStopping

# Assume user has gpu ready server
cuda = torch.device('cuda')

# initialize RNG
RNG = default_rng()


# create mapping table from genes to nodes number
def node_mapping(gene_name):
    node_map = {}
    for i, j in enumerate(gene_name):
        node_map.update({j: i})
    return node_map


gene_set = set(["App", "Apoe", "Gusb", "Lamp5", "Mbp", "Pvalb", "S100b", "Slc30a3", "Snca", "Mapt"])


# returns an edge list
# execute filter when filtered is set to True
# no genes in gene_set are allowed to be in the returned edge_List except the filterGene
def filterEdgeListByGene(filterGene, node_map, gene_edge, filtered=True):
    edge_list = []
    unrecog_gene = 0
    filtered_gene_set = set(gene_set)
    filtered_gene_set.remove(filterGene)
    for k, i in enumerate(gene_edge):

        letter_count = 0
        while i[letter_count] is not '|':
            letter_count += 1

        first_gene = i[:letter_count]
        second_gene = i[letter_count + 1:]

        try:
            node_map[first_gene]
            node_map[second_gene]

            if (first_gene in filtered_gene_set or second_gene in filtered_gene_set) and filtered is True:
                # do not add to edge list
                continue

            edge_list.append([node_map[first_gene], node_map[second_gene]])
            # edge_list.append([node_map[second_gene], node_map[first_gene]])
        except:
            unrecog_gene += 1
            pass

    print('%d edges not recoginze; the size of edge list is [%d, %d]'
          % (unrecog_gene, np.shape(edge_list)[0], np.shape(edge_list)[1]))
    return torch.tensor(np.array(edge_list).T, dtype=torch.long)


def data_loader(train_file, test_file, node_map, edge_list, target_name):
    gene_idx = []
    train_loader = []
    validate_loader = []
    target_gene = target_name

    for i in range(len(target_name)):
        gene_idx.append(node_map[target_gene[i]])
    gene_idx = np.array(gene_idx)

    with open(train_file) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            # index 1420 onwards are downstream gene
            new_row = np.array(row)
            target_expression_level = [i for i in row[1421:]]
            target_expression_level = torch.tensor(target_expression_level).view(len(target_name), 1)
            new_row[1421:] = 0

            # create graph
            data = Data(x=torch.tensor(new_row, dtype=torch.float).view(len(row), 1),
                        y=target_expression_level, edge_index=edge_list, gene_idx_=gene_idx)
            train_loader.append(data)

    with open('input_test_cat.csv') as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:

            new_row = np.array(row)
            target_expression_level = [i for i in row[1421:]]
            target_expression_level = torch.tensor(target_expression_level).view(len(target_name), 1)
            new_row[1421:] = 0

            # create graph
            data = Data(x=torch.tensor(row, dtype=torch.float).view(len(row), 1),
                        y=target_expression_level, edge_index=edge_list, gene_idx_=gene_idx)
            validate_loader.append(data)
     
    print('train set contains %d graphs, validate set contains %d graphs' %(len(train_loader), len(validate_loader)))
    return train_loader, validate_loader


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


# for total train
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


def getLabel(y, gene_idx, n_target):
    # y is a batch of graph
    # y has the format [gene1, gene2, gene3..., gene1, gene2, gene3...]
    # 1,2,3 repercents n_target, and '...' repercents n_batch
    # return selected lables form y according to gene_idx
    selected_label = []
    bat_size = int(y.size()[0] / n_target)
    for i in range(bat_size):
        selected_label.append(y[gene_idx + i*n_target])
    return torch.tensor(selected_label, device=cuda).view(bat_size, 1)


def train(model, train_loader, batch_size, lr):
    model.train()
    target_size = train_loader[0].y.size()[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loss_all = 0

    # create progess bar for training
    count = 0
    loop_length = np.zeros((len(train_loader) // batch_size,)).tolist()
    loop = tqdm(loop_length)

    # order of samples for stochastic mini batch
    sample_order = RNG.permutation(len(train_loader))

    for i in loop:

        # conserve gpu memory
        try:
            del total_pred, batch, x, edges, y, loss_
        except:
            pass

        # stochastic mini batch
        batch = [train_loader[j] for j in sample_order[count * batch_size:(count + 1) * batch_size]]
        batch = Batch.from_data_list(batch).to(device=cuda)
        x, edges, y = batch.x, batch.edge_index, batch.y

        total_pred = model(batch)

        try:
            # the model contains one head
            dummy = total_pred.size()
            loss_ = loss_fn(total_pred, y)
        except:
            # The model contains nine heads
            # calculate losses for each heads, and then sum them up
            # backprop on the total loss to update weights for all nine heads
            n_pred = len(total_pred)

            # calculate the first loss
            label_ = getLabel(y, 0, target_size)
            loss_ = loss_fn(total_pred[0], label_)

            # sum up the rest losses for backprop
            for i in range(n_pred - 1):
                label_ = getLabel(y, i + 1, target_size)
                loss_ = loss_ + loss_fn(total_pred[i + 1], label_)

        loss_all += loss_.item()

        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()

        # output number of epoches, train loss and mse
        loop.set_postfix(loss=loss_all / (count + 1))
        count += 1

    return loss_all / (len(train_loader) // batch_size)


def validate(model, validate_loader, batch_size):
    model.eval()
    target_size = validate_loader[0].y.size()[0]
    loss_fn = nn.MSELoss()
    loss_tot = np.zeros((target_size, ))
    loss_all = 0

    for i in range(len(validate_loader) // batch_size):
        # ordered mini-batch
        batch = [validate_loader[j] for j in range(i * batch_size, (i + 1) * batch_size)]
        batch = Batch.from_data_list(batch).to(device=cuda)
        x, edges, y = batch.x, batch.edge_index, batch.y

        total_pred = model(batch)

        try:
            dummy = total_pred.size()
            loss_ = loss_fn(total_pred, y)

            for k in range(batch_size):
                loss_all_ind = []
                for j in range(target_size):
                    loss_ = loss_fn(total_pred[k * target_size + j], y[k * target_size + j])
                    loss_all_ind.append(loss_.item())
                loss_all_ind = np.array(loss_all_ind)
                loss_tot += loss_all_ind
        except:
            total_pred = len(total_pred)

            # calculate the first loss
            label_ = getLabel(y, 0, target_size)
            loss_ = loss_fn(total_pred[0], label_)
            indivdual_loss = [loss_.item()]

            # sum up the rest loss
            for i in range(len(total_pred) - 1):
                label_ = getLabel(y, i + 1, target_size)
                next_loss = loss_fn(total_pred[i + 1], label_)
                indivdual_loss.append(next_loss.item())
                loss_ = loss_ + next_loss
            loss_tot += np.array(indivdual_loss)

        loss_all += loss_.item()

    return loss_all / (len(validate_loader) // batch_size), loss_tot


def multi_train(model, train_loader, validate_loader, train_batch_size=16, validate_batch_size=32, patience=3, epoches=30, lr=1e-5):
    # for storing individual gene mse loss
    temp_lis = []
    # for storing total mse loss during training
    train_loss_lis = []
    # for storing total mse loss during validate
    validate_loss_lis = []
    # for storing individual mse loss during validation
    val_loss_ind_lis = []

    # define early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=False)

    for i in range(epoches):

        train_loss = train(model, train_loader, train_batch_size, lr=lr)
        train_loss_lis.append(train_loss)

        val_loss, val_loss_ind = validate(model, validate_loader, validate_batch_size)
        validate_loss_lis.append(val_loss)
        val_loss_ind_lis.append(val_loss_ind)

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return train_loss_lis, validate_loss_lis, val_loss_ind_lis