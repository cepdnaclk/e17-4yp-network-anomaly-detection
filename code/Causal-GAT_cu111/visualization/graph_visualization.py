import os
import sys

import numpy as np
import networkx as nx

from matplotlib import pyplot as plt


anomaly_node_size = 80
default_node_size = 20

central_node_color = "yellow"
anomaly_node_color = "red"
default_node_color = "black"

anomaly_edge_color = "red"
default_edge_color = (0.35686275, 0.20392157, 0.34901961, 0.1)


def visualize_graph(model=None, n_samples=None, feature_num=None , feature_map=None, dataset=None):

    
    # Compute the graph structure from train data
    coeff_weights = model.gnn_layers[0].att_weight_1.cpu().detach().numpy()
    edge_index = model.gnn_layers[0].edge_index_1.cpu().detach().numpy()
    weight_mat = np.zeros((feature_num, feature_num))

    for i in range(len(coeff_weights)):
        edge_i, edge_j = edge_index[:, i]
        edge_i, edge_j = edge_i % feature_num, edge_j % feature_num
        weight_mat[edge_i][edge_j] += coeff_weights[i]

    weight_mat /= n_samples

    adj_mat = weight_mat

    # Define the central nodes
    for central_node_id in range (feature_num):
        central_node = int(central_node_id)

        G = nx.from_numpy_array(adj_mat)
        G.remove_edges_from(nx.selfloop_edges(G))
        pos = nx.spring_layout(G)

        # Find the neighboring nodes and selected the edges with highest value
        scores = np.stack([adj_mat[central_node], adj_mat[:, central_node]], axis=1)
        scores = np.max(scores, axis=1)

        # Define red nodes as the nodes with edge weight > 0.1
        red_nodes = list(np.where(scores > 0.05)[0])

        edges = [set(edge) for edge in G.edges()]
        edge_colors = [default_edge_color for edge in edges]

        node_colors = [default_node_color for i in range(feature_num)]
        node_sizes = [default_node_size for i in range(feature_num)]

        node_colors[central_node] = central_node_color
        node_sizes[central_node] = anomaly_node_size

        for node in red_nodes:

            if node == central_node:
                continue

            node_colors[node] = anomaly_node_color
            node_sizes[node] = anomaly_node_size

            edge_pos = edges.index(set((node, central_node)))
            edge_colors[edge_pos] = anomaly_edge_color

        x, y = pos[central_node]
        plt.text(x,y + 0.15,
                s=feature_map[central_node],
                bbox=dict(facecolor=central_node_color, alpha=0.5), horizontalalignment='center')

        # print("Central Node:", feature_map[central_node])

        for node in red_nodes:
            x, y = pos[node]
            plt.text(x,y + 0.15,
                    s=feature_map[node],
                    bbox=dict(facecolor=anomaly_node_color, alpha=0.5), horizontalalignment='center')

            # print("Red Node:", feature_map[node])

        # for node in range (feature_num):
        #     if (node == central_node) or (node in red_nodes):
        #         continue
        #     x, y = pos[node]
        #     plt.text(x,y + 0.15,
        #             s=feature_map[node],
        #             bbox=dict(facecolor=default_node_color, alpha=0.5), horizontalalignment='center')

        nx.draw(G, pos,
                edge_color=edge_colors,
                node_color=node_colors,
                node_size=node_sizes)

        output_file_for_central_node = os.path.join(f'./visualization/{dataset}/{feature_map[central_node]}_graph.png')
        plt.savefig(output_file_for_central_node, format="PNG")
        plt.clf()