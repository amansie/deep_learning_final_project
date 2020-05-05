import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import csv
import gene_inference_model as graph_model
from tqdm import tqdm
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GatedGraphConv
from numpy.random import default_rng
from pytorchtools import EarlyStopping

### This file contains all untility functions such as load data, trainig and validation ###


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

#     print('%d edges not recoginze; the size of edge list is [%d, %d]'
#           % (unrecog_gene, np.shape(edge_list)[0], np.shape(edge_list)[1]))
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

    with open(test_file) as csvfile:
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


def data_loader_st(train_file, test_file, node_map, edge_list, target_name):
    train_loader = []
    validate_loader = []
    target_gene = target_name

    with open(train_file) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            new_row = np.array(row)
            target_expression_level = row[node_map[target_gene]]
            target_expression_level = torch.tensor(target_expression_level).view(1, 1)
            new_row[node_map[target_gene]] = 0

            # create graph
            data = Data(x=torch.tensor(new_row, dtype=torch.float).view(len(row), 1),
                        y=target_expression_level, edge_index=edge_list, gene_idx_=node_map[target_gene])
            train_loader.append(data)

    with open(test_file) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:

            new_row = np.array(row)
            target_expression_level = row[node_map[target_gene]]
            target_expression_level = torch.tensor(target_expression_level).view(1, 1)
            new_row[node_map[target_gene]] = 0

            # create graph
            data = Data(x=torch.tensor(row, dtype=torch.float).view(len(row), 1),
                        y=target_expression_level, edge_index=edge_list, gene_idx_=node_map[target_gene])
            validate_loader.append(data)
     
    # print('train set contains %d graphs, validate set contains %d graphs' %(len(train_loader), len(validate_loader)))
    return train_loader, validate_loader


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


def train(model, train_loader, batch_size, lr, weight):
    model.train()
    target_size = train_loader[0].y.size()[0]
    if weight is None: 
        weight = np.ones((target_size, ))
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

            # calculate the losses for each prediciton
            loss_1 = loss_fn(total_pred[0], getLabel(y, 0, target_size)) * weight[0]
            loss_2 = loss_fn(total_pred[1], getLabel(y, 1, target_size)) * weight[1]
            loss_3 = loss_fn(total_pred[2], getLabel(y, 2, target_size)) * weight[2]
            loss_4 = loss_fn(total_pred[3], getLabel(y, 3, target_size)) * weight[3]
            loss_5 = loss_fn(total_pred[4], getLabel(y, 4, target_size)) * weight[4]
            loss_6 = loss_fn(total_pred[5], getLabel(y, 5, target_size)) * weight[5]
            loss_7 = loss_fn(total_pred[6], getLabel(y, 6, target_size)) * weight[6]
            loss_8 = loss_fn(total_pred[7], getLabel(y, 7, target_size)) * weight[7]
            loss_9 = loss_fn(total_pred[8], getLabel(y, 8, target_size)) * weight[8]
            
            loss_ = loss_1+loss_2+loss_3+loss_4+loss_5+loss_6+loss_7+loss_8+loss_9
            
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


def validate_st(model, validate_loader, batch_size):
    model.eval()
    target_size = validate_loader[0].y.size()[0]
    loss_fn = nn.MSELoss()
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
        except:
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

    return loss_all / (len(validate_loader) // batch_size)


def multi_train(model, train_loader, validate_loader, train_batch_size=16, 
                validate_batch_size=32, patience=3, epoches=30, lr=1e-5, weight=None):
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

        train_loss = train(model, train_loader, train_batch_size, lr=lr, weight=weight)
        train_loss_lis.append(train_loss)

        val_loss, val_loss_ind = validate(model, validate_loader, validate_batch_size)
        validate_loss_lis.append(val_loss)
        val_loss_ind_lis.append(val_loss_ind)

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return train_loss_lis, validate_loss_lis, val_loss_ind_lis


def single_train(model, train_loader, validate_loader, train_batch_size=32,
                validate_batch_size=32, patience=3, epoches=70, lr=1e-5, weight=None):
    # for storing individual gene mse loss
    temp_lis = []
    # for storing total mse loss during training
    train_loss_lis = []
    # for storing total mse loss during validate
    validate_loss_lis = []

    # define early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=False)

    for i in range(epoches):

        train_loss = train(model, train_loader, train_batch_size, lr=lr, weight=weight)
        train_loss_lis.append(train_loss)

        val_loss = validate_st(model, validate_loader, validate_batch_size)
        validate_loss_lis.append(val_loss)

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return train_loss_lis, validate_loss_lis


def multiSingleTrain(model_name, train_file, test_file, node_map, gene_edge, target_genes):
    master_validate_loss = []
    for gene in target_genes:
        # create edge list for each gene
        edge_list = filterEdgeListByGene(gene, node_map, gene_edge, True)
        
        # create graph data for each target gene
        train_loader, validate_loader = data_loader_st(train_file, test_file, node_map, edge_list, gene)
        
        # create GNN model
        if model_name == 'GCN': 
            model = graph_model.GCNInferenceNetwork().to(torch.device('cuda'))
        elif model_name == 'GGCN': 
            model = graph_model.GGCN().to(torch.device('cuda'))
        elif model_name == 'GAT': 
            model = graph_model.GATInferenceNetwork().to(torch.device('cuda'))
        else: 
            print('model not implmented')
            return None

        # train on single target gene
        train_loss, val_loss = single_train(model, train_loader, validate_loader)
        master_validate_loss.append(val_loss)

        # print results
        print(gene+' training completed, final val_loss is: %f' %(val_loss[-1]))

        # empty cpu
        del train_loader, validate_loader, edge_list, model
    return master_validate_loss