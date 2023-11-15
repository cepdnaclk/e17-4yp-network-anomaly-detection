import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import torch.nn as nn
import time
from util.time import *
from util.env import *
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
import torch.nn.functional as F

from .graph_layer import GraphLayer


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()
    
    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()


class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num = 512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num-1:
                modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear( layer_in_num, inter_num ))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out


class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):

        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
  
        out = self.bn(out)
        
        return self.relu(out)


class CAUSAL_GAT(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, significance_level=0.05):

        super(CAUSAL_GAT, self).__init__()

        self.edge_index_sets = edge_index_sets

        device = get_device()

        edge_index = edge_index_sets[0]

        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        edge_set_num = len(edge_index_sets)

        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1) for i in range(edge_set_num)
        ])

        self.node_embedding = None
        self.significance_level = significance_level
        self.learned_graph = None

        self.out_layer = OutLayer(dim*edge_set_num, node_num, out_layer_num, inter_num = out_layer_inter_dim)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        self.init_params()
    

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))


    def forward(self, data, labels, org_edge_index):
        
        x = data.clone().detach()
        y = labels.clone().detach()
        edge_index_sets = self.edge_index_sets

        device = data.device

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num*batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)
            
            batch_edge_index = self.cache_edge_index_sets[i]
            
            all_embeddings = self.embedding(torch.arange(node_num).to(device))
            all_embeddings = all_embeddings.repeat(batch_num, 1)

            # ============ Casual Connection Learning ==============
            causal_graph = self.get_causal_edges(edge_index, batch_num, node_num, device, x, y, all_embeddings, self.significance_level)
            self.learned_graph = causal_graph
            # ==========================================================
            
            batch_gated_edge_index = get_batch_edge_index(causal_graph, batch_num, node_num).to(device)
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)
            gcn_outs.append(gcn_out)
        
        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)

        indexes = torch.arange(0,node_num).to(device)
        out = torch.mul(x, self.embedding(indexes))
        
        out = out.permute(0,2,1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)
        
        return out
    
    # ============= Function for learning causal connections ============
    def get_causal_edges(self, org_edge_index, batch_num, node_num, device, org_x, org_y, all_embeddings, significance_level):
        edge_index = org_edge_index.clone().detach()
        train_mode = self.training
        self.eval()
        for module in self.children():
            module.eval()

        # 1. Get the sub models =========================
        # Add a self-connection to each p_index
        for p_index in torch.unique(org_edge_index[1]):
            # Find the indices where the value appears in the second dimension
            indices = torch.nonzero((edge_index[1] == p_index) & (edge_index[0] > p_index), as_tuple=True)[0]

            # Get the index to insert before (the first occurrence)
            insert_index = indices[0].item() if len(indices) > 0 else len(edge_index[0])

            # Insert the value in both dimensions at the specified index
            edge_index = torch.cat((edge_index[:, :insert_index], torch.tensor([[p_index], [p_index]]), edge_index[:, insert_index:]), dim=1)
        
        edge_num = edge_index.shape[1]
        sub_models = []
        # Add the fully connected graph as well
        sub_models.append(edge_index)

        # Create a list to keep track of removed edges per p_index
        removed_edges_per_target = [[] for _ in range(edge_index.max().item() + 1)]

        # Iterate through each submodel
        for i in range(node_num-1):

            # Initialize a mask to select edges for the current submodel
            mask = torch.ones(edge_num, dtype=torch.bool)

            # Iterate through each p_index and find the first edge that hasn't been removed yet
            for p_index in torch.unique(edge_index[1]):

                # Get the c_indexes
                c_indexes = edge_index[0][edge_index[1] == p_index]

                # Remove the first edge which that hasn't been removed yet (Do not remove the self edge)
                for c_index in c_indexes:
                    if (c_index.item() != p_index) and (c_index.item() not in removed_edges_per_target[p_index.item()]):

                        # Mark this edge as removed
                        removed_edges_per_target[p_index.item()].append(c_index.item())

                        # Exclude this edge from the mask
                        indices = torch.nonzero((edge_index[0] == c_index) & (edge_index[1] == p_index)).squeeze()
                        mask[indices] = 0
                        break

            # Create a submodel by excluding the selected edges
            sub_edge_tensor = edge_index[:, mask]

            # Append the submodel to the list
            sub_models.append(sub_edge_tensor)

        
        # 2. Make the predictions =========================
        submodel_predictions = []
        for sub_model in sub_models:
            x = org_x.clone().detach()

            batch_edge_index = get_batch_edge_index(sub_model, batch_num, node_num).to(device)
            
            with torch.no_grad():
                gcn_out = self.gnn_layers[0](x, batch_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)
            
            x = gcn_out
            x = x.view(batch_num, node_num, -1)

            indexes = torch.arange(0,node_num).to(device)
            out = torch.mul(x, self.embedding(indexes))
            
            out = out.permute(0,2,1)
            out = F.relu(self.bn_outlayer_in(out))
            out = out.permute(0,2,1)

            out = self.dp(out)
            out = self.out_layer(out)
            out = out.view(-1, node_num)

            # Save the predictions
            submodel_predictions.append(out)


        # 3. Perform an F-test with the predictions 
        # of the fully connected graph =========================
        causally_learned_graph = sub_models[0].clone().detach()
        true_y = org_y.clone().detach()
        for submodel_num in range(1,len(sub_models)):
            for p_index in range(node_num):
                pred_y_full_graph = submodel_predictions[0][:,p_index]
                pred_y_sub_model = submodel_predictions[submodel_num][:,p_index]
                true_y_for_p_index = true_y[:,p_index]

                if "cuda" in get_device():
                    pred_y_full_graph = pred_y_full_graph.cpu()
                    pred_y_sub_model = pred_y_sub_model.cpu()
                    true_y_for_p_index = true_y_for_p_index.cpu()

                pred_y_full_graph = pred_y_full_graph.detach().numpy()
                pred_y_sub_model = pred_y_sub_model.detach().numpy()

                # Calculated the residuals
                R_full_graph = true_y_for_p_index - pred_y_full_graph
                R_sub_model = true_y_for_p_index - pred_y_sub_model

                # Perform the F-test (ANOVA test) on residuals
                f_statistic, p_value = f_oneway(R_sub_model, R_full_graph)

                # If p_value <= alpha --> reject the hypothesis(There is significant difference between the groups)
                # If there is no significant difference => remove the edge
                if p_value > significance_level:
                    # Find the edge to remove
                    submodel = sub_models[submodel_num]
                    c_indexes = submodel[0][submodel[1] == p_index]
                    missing_c_index = torch.tensor([x for x in torch.arange(0, node_num) if x not in c_indexes]).item()
                    
                    # Remove that edge from the causally_learned_graph
                    mask = torch.ones(causally_learned_graph.shape[1], dtype=torch.bool)
                    indices = torch.nonzero((causally_learned_graph[0] == missing_c_index) & (causally_learned_graph[1] == p_index)).squeeze()
                    mask[indices] = 0
                    causally_learned_graph = causally_learned_graph[:, mask]


        if train_mode:
            self.train()
            for module in self.children():
                module.train()

        return causally_learned_graph
        