# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.preprocessing import MinMaxScaler

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.iostream import printsep,save_attack_infos

from visualization.graph_visualization import visualize_graph

from datasets.TimeDataset import TimeDataset


from models.CAUSAL_GAT import CAUSAL_GAT

from train import train
from test  import test
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random


class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset'] 
        train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
        test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)
       
        train, test = train_orig, test_orig

        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        # [[c_index],[p_index]]
        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())

        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)

        self.train_dataloader_list = []
        self.val_dataloader_list = []

        for fold in range(train_config['num_folds']):
            # Generate loaders for the current fold
            train_loader, val_loader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], train_config['num_folds'], fold)
            self.train_dataloader_list.append(train_loader)
            self.val_dataloader_list.append(val_loader)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'], shuffle=False, num_workers=0)

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        self.model = CAUSAL_GAT(edge_index_sets, len(feature_map), 
                dim=train_config['dim'], 
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                significance_level=train_config['significance_level']
            ).to(self.device)


    def run(self):

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
            print(model_save_path)
        else:
            model_save_path = self.get_save_path()[0]
            print(model_save_path)

            self.train_log = train(self.model, model_save_path, 
                config = train_config,
                train_dataloader_list=self.train_dataloader_list,
                val_dataloader_list=self.val_dataloader_list
                )

        # test    
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        _, self.test_result = test(best_model, self.test_dataloader)
        _, self.val_result = test(best_model, self.val_dataloader)

        self.get_score(self.test_result, self.val_result)


    def get_loaders(self, train_dataset, seed, batch, num_folds=5, current_fold=0):
        
        dataset_len = len(train_dataset)        
        fold_size = dataset_len // num_folds
        
        val_start_index = current_fold * fold_size
        val_end_index = val_start_index + fold_size
        
        indices = torch.arange(dataset_len)

        val_sub_indices = indices[val_start_index:val_end_index]
        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_end_index:]])

        train_subset = Subset(train_dataset, train_sub_indices)
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=True)

        return train_dataloader, val_dataloader


    def get_score(self, test_result, val_result):

        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)

        test_labels = np_test_result[2, :, 0].tolist()
    
        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1) 
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)

        print('=========================** Result **============================\n')

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info

        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}\n')
        print(f'confusion matrix: {info[4]}\n')

        if self.env_config['report'] == 'best':
            save_path = self.get_save_path()[1]
            save_attack_infos(info, test_scores, test_labels,  self.feature_map, save_path, self.env_config['dataset'], self.train_config)
        
        if self.env_config['visualize_graph'] is True:
            X_train = main.train_dataset.x
            n_samples, feature_num, _ = X_train.shape
            visualize_graph(model=self.model, n_samples=n_samples, feature_num=feature_num , 
                            feature_map=self.feature_map, dataset=env_config['dataset'])
        

    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            # date-time format for linux 
            # self.datestr = now.strftime('%m|%d-%H:%M:%S')
            self.datestr = now.strftime('%m_%d-%H-%M-%S')
        datestr = self.datestr          

        paths = [
            f'./pretrained/{dir_path}/best_{datestr}.pt',
            f'./results/{dir_path}/{datestr}.json',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type = int, default=128)
    parser.add_argument('-epoch', help='train epoch', type = int, default=100)
    parser.add_argument('-slide_win', help='slide_win', type = int, default=15)
    parser.add_argument('-dim', help='dimension', type = int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=5)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type = str, default='wadi')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type = int, default=0)
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-num_folds', help='num folds', type = int, default=5)
    parser.add_argument('-significance_level', help='significance_level', type = float, default=0.05)
    parser.add_argument('-report', help='best / val', type = str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')
    parser.add_argument('-visualize_graph', help='visualize graph', type = bool, default=False)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)


    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'num_folds': args.num_folds,
        'significance_level': args.significance_level,
    }

    env_config={
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path,
        'visualize_graph' : args.visualize_graph
    }

    main = Main(train_config, env_config, debug=False)
    main.run()
    