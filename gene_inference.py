from copy import deepcopy
import numpy as np
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


#returns an edge list
#no genes in gene_set are allowed to be in the returned edge_List except the filterGene
#TODO: test when we have the google credits running
def filterEdgeListByGene(filterGene, gene_edge, gene_set):
    edge_list = []
    filtered_gene_set = set(gene_set)
    filtered_gene_set.remove(filterGene)
    for k, i in enumerate(gene_edge): 
        
        letter_count = 0
        while i[letter_count] is not '|': 
            letter_count += 1
            
        first_gene = i[:letter_count]
        second_gene = i[letter_count+1:]
        
        try: 
            node_map[first_gene]
            node_map[second_gene]
            if(first_gene in filtered_gene_set or second_gene in filtered_gene_set ):
                #do not add to edge list
                continue
              
            edge_list.append([node_map[first_gene], node_map[second_gene]])
            edge_list.append([node_map[second_gene], node_map[first_gene]])
        except: 
            print('could not fine gene name at %dth line of gene edge' %k)
    return edge_list


# load training data, return list of cell graphs
# filename: filename for gene expression data to load (.csv file)
# gene: gene name to predict as output
# edge_list: list of edges, in the format that torch_geometric.Data.data takes
def data_loader(filename, gene, node_map, edge_list, multiplier=1e5):
    # get order of genes, based on node_map
    gene_order = [k for k, _ in sorted(node_map.items(), key=lambda item: item[1])]

    loader = []

    reader = pd.read_csv(filename, header=0, index_col=0)[gene_order]
    target_expression_levels = deepcopy(reader[gene])
    reader.loc[:,gene] = 0.0
    reader = reader * multiplier
    for (_, row), target_expression_level in zip(reader.iterrows(), target_expression_levels):
        # iterate through examples (cells)        
        # create graph
        data = Data(x=torch.tensor(row, dtype=torch.float).view(-1, 1), 
                    y=target_expression_level, edge_index=edge_list)
        loader.append(data)
    return loader


class GCNInferenceNetwork(nn.Module): 
    def __init__(self, d=1): 
        super(GCNInferenceNetwork, self).__init__()
        self.conv1 = GCNConv(d, 16)
        self.conv2 = GCNConv(16, 16)
        self.fc1   = nn.Linear(16, 8)
        self.fc2   = nn.Linear(8, 1)
    
    def forward(self, graph): 
        x, edges, y = graph.x, graph.edge_index, graph.y
        
        # 2 layer convolution on the whole graph
        x = F.relu(self.conv1(x, edges))
        x = F.dropout(x, p=0.4)
        x = F.relu(self.conv2(x, edges))

        # isolate gene of interest, and feed into MLP for regression
        temp_x = x[1423].view(1, 16)
        for i in range(y.size(0)-1): 
            temp_x = torch.cat((temp_x, x[1423+(i+1)*1431].view(1, 16)))
            
        temp_x = F.relu(self.fc1(temp_x))
        temp_x = self.fc2(temp_x)
            
        return temp_x


def MSError(pred_, y): 
    mse_sum = 0
    for i in range(len(pred_)): 
        mse_sum += (y[i]-pred_[i])**2
    mse_sum = mse_sum / len(pred_)
    return mse_sum


def train(model, train_loader, batch_size): 
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    loss_fn = nn.MSELoss()
    
    loss_all = 0
    
    # create progess bar for training
    count = 0
    loop_length = np.zeros((len(train_loader)//batch_size, )).tolist()
    loop = tqdm(loop_length)
    
    for i in loop:  
        
        # conserve gpu memory 
        try: 
            del pred_, batch, x, edges, y, loss
        except: 
            pass
        
        # ordered mini batch
        batch = [train_loader[j] for j in range(count*batch_size, (count+1)*batch_size)]
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