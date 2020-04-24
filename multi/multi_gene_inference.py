from copy import deepcopy
import numpy as np
from numpy.random import default_rng
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import csv
from tqdm import tqdm
from time import sleep
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


CUDA_DEVICE = torch.device('cuda')

# initialize RNG
RNG = default_rng()


#returns an edge list
#no genes in gene_set are allowed to be in the returned edge_List except the filterGene
#TODO: test when we have the google credits running
def filterEdgeListByGene(filterGene, gene_edge, gene_set, node_map):
    gene_set = deepcopy(gene_set)
    edge_list = []
    gene_set.remove(filterGene)
#     for k, i in enumerate(gene_edge): 
        
#         letter_count = 0
#         while i[letter_count] is not '|': 
#             letter_count += 1
            
#         first_gene = i[:letter_count]
#         second_gene = i[letter_count+1:]

    for first_gene, second_gene in gene_edge:  # use the edge list uploaded to GitHub
        try: 
            node_map[first_gene]
            node_map[second_gene]
            if(first_gene in gene_set or second_gene in gene_set):
                #do not add to edge list
                continue
              
            edge_list.append([node_map[first_gene], node_map[second_gene]])
            edge_list.append([node_map[second_gene], node_map[first_gene]])
        except: 
#             print('could not find genes in edge {}/{}'.format(first_gene, second_gene))
            pass
    return edge_list


# load training data as pytorch_geometric Data objects
# unlike data_loader, can generate graphs to predict one of several genes
# each graph in the dataset will have one gene's expression as its expected output
# filename: filename for gene expression data to load (.csv file)
# cat: filename of additional (eg "output gene") expression data
# genes: names of the genes to predict as output
# gene_edge: list of edges, as a list of gene name pairs (2-element iterable)
# node_map: dictionary mapping gene name to node number
# edge_list: list of gene name pairs, representing edges
# returns a list of pytorch_geometric Data objects
def multi_data_loader(filename, genes, node_map, gene_edge, cat=None):
    # get order of genes, based on node_map
    gene_order = [k for k, _ in sorted(node_map.items(), key=lambda item: item[1])]

    # read expression data
    reader = pd.read_csv(filename, header=0, index_col=0)
    if not cat is None:
        reader_cat = pd.read_csv(cat, header=0, index_col=0)
        reader = pd.concat([reader, reader_cat], axis=1)
    reader = reader[gene_order]
    
    # convert edge list into pytorch_geometric Data format
    edge_lists = dict()
    for gene in genes:
        edge_list = filterEdgeListByGene(gene, gene_edge, genes, node_map)
        edge_lists[gene] = torch.tensor(edge_list, dtype=torch.long).t()

    loader = []
    # iterate through examples (cells)
    for _, row in reader.iterrows():
        # iterate through output genes
        for gene in genes:
            row_copy = deepcopy(row)
            target_expression_level = torch.tensor(row[gene]).view(1, 1)
            row_copy.loc[gene] = 0.0
            # create graph
            data = Data(x=torch.tensor(row, dtype=torch.float).view(-1, 1),
                        y=target_expression_level,
                        edge_index=edge_lists[gene],
                        gene_node=node_map[gene])
            loader.append(data)
    return loader


class MultiGCNInferenceNetwork(nn.Module): 
    def __init__(self, d=1): 
        super(MultiGCNInferenceNetwork, self).__init__()
        self.conv1 = GCNConv(d, 16)
        self.conv2 = GCNConv(16, 16)
        self.fc1   = nn.Linear(16, 8)
        self.fc2   = nn.Linear(8, 1)
    
    def forward(self, graph): 
        x, edges, y, gene_node = graph.x, graph.edge_index, graph.y, graph.gene_node
        
        # 2 layer convolution on the whole graph
        x = F.relu(self.conv1(x, edges))
        x = F.dropout(x, p=0.4)
        x = F.relu(self.conv2(x, edges))

        # isolate gene of interest, and feed into MLP for regression
        temp_x = x[gene_node.view(-1)[0]].view(1, 16)
        for i in range(y.size(0)-1):
            temp_x = torch.cat((temp_x, x[gene_node.view(-1)[i+1]+(i+1)*1431].view(1, 16)))
            
        temp_x = F.relu(self.fc1(temp_x))
        temp_x = self.fc2(temp_x)
            
        return temp_x


def multi_train(model, train_loader, batch_size): 
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.MSELoss()
    
    loss_all = 0
    
    # create progess bar for training
    count = 0
    loop_length = np.zeros((len(train_loader)//batch_size, )).tolist()
    loop = tqdm(loop_length)
    
    # order of samples for stochastic mini batch
    sample_order = RNG.permutation(len(train_loader))
    
    for i in loop:
        
        # conserve gpu memory 
        try:
            del pred_, batch, x, edges, y, loss
        except:
            pass
        
        # stochastic mini batch
        batch = [train_loader[j] for j in sample_order[count*batch_size:(count+1)*batch_size]]
        batch = Batch.from_data_list(batch).to(device=CUDA_DEVICE)
        
        x, edges, y = batch.x, batch.edge_index, batch.y
        pred_ = model(batch)
        
        loss = loss_fn(pred_, y)
        loss_all += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # output number of epoches, train loss and mse
        loop.set_postfix(loss=loss_all/(count+1))
        count += 1
    
    return loss_all / (len(train_loader)//batch_size)


def validate(model, validate_loader, batch_size): 
    model.eval()
    loss_fn = nn.MSELoss()
    loss_all = 0
    
    for i in range(len(validate_loader)//batch_size): 
        
        # conserve gpu memory 
        try: 
            del pred_, batch, x, edges, y, loss
        except: 
            pass
        
        # ordered mini-batch
        batch = [validate_loader[j] for j in range(i*batch_size, (i+1)*batch_size)]
        batch = Batch.from_data_list(batch).to(device=CUDA_DEVICE)
        
        x, edges, y = batch.x, batch.edge_index, batch.y
        pred_ = model(batch)
        
        loss = loss_fn(pred_, y)
        loss_all += loss.item()
    
    return loss_all / (len(validate_loader)//batch_size)