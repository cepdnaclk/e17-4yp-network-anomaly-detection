#!/usr/bin/env python

import json
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader
from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from datasets.TimeDataset import TimeDataset
from datetime import datetime
from argparse import ArgumentParser, FileType
from configparser import ConfigParser
from confluent_kafka import Producer

if __name__ == '__main__':
    # Parse the command line.
    parser = ArgumentParser()
    parser.add_argument('config_file', type=FileType('r'))
    args = parser.parse_args()

    # Parse the configuration.
    # See https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
    config_parser = ConfigParser()
    config_parser.read_file(args.config_file)
    config = dict(config_parser['default'])

    # Create Producer instance
    producer = Producer(config)

    # Create Dataloader
    dataset = "swat"
    train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
    test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)

    train, test = train_orig, test_orig

    if 'attack' in train.columns:
        train = train.drop(columns=['attack'])

    feature_map = get_feature_map(dataset)
    fc_struc = get_fc_graph_struc(dataset)

    # [[c_index],[p_index]]
    fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
    fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)

    # train_dataset_indata = construct_data(train, feature_map, labels=0)
    test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())

    cfg = {
        'slide_win': 5,
        'slide_stride': 1,
    }

    test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    topic = "sensordata"

    # Iterate through your dataloader
    # type(x,y,labels,edge_index), <class 'torch.Tensor'>
    # x torch.Size([1, 51, 5])
    # y torch.Size([1, 51])
    # labels torch.Size([1])
    # edge_index torch.Size([1, 2, 2550])
    for x, y, labels, edge_index in test_dataloader:
        # Convert tensors to lists or numpy arrays if needed
        x, y, labels, edge_index = x.cpu().numpy(), y.cpu().numpy(), labels.cpu().numpy(), edge_index.cpu().numpy()


        # Create a dictionary to hold your data
        data = {
            'x': x.tolist(),
            'y': y.tolist(),
            'labels': labels.tolist(),
            'edge_index': edge_index.tolist(),
            'fc_edge_index': fc_edge_index.cpu().numpy().tolist(),
            'feature_map': feature_map
        }

        # Serialize the data to JSON
        data_json = json.dumps(data)
        # Published time
        timestamp = time.time()
        current_time = datetime.fromtimestamp(timestamp)
        # Send the data to the Kafka topic
        producer.produce(topic, key=None, value=data_json)
        print("sensor data successfully published: ", current_time)
        time.sleep(4)

    # Block until the messages are sent.
    producer.poll(10000)
    producer.flush()
