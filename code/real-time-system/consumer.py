#!/usr/bin/env python
import numpy as np
from dotenv import load_dotenv
from influxdb_client.client import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

from util.env import get_device, set_device

from util.iostream import save_attack_infos

from models.CAUSAL_GAT import CAUSAL_GAT

from test import test
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores

import os
from argparse import ArgumentParser, FileType
from configparser import ConfigParser

import torch
from confluent_kafka import Consumer, OFFSET_BEGINNING
from rest_framework.utils import json


def get_score(test_result, feature_map):
    print('=========================** Result **============================\n')
    np_test_result = np.array(test_result)
    test_labels = np_test_result[2, :, 0].tolist()
    test_scores = get_full_err_scores(test_result)
    top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)
    attacks = save_attack_infos(top1_best_info, test_scores, test_labels, feature_map, "swat")
    return attacks


if __name__ == '__main__':
    # Parse the command line.
    parser = ArgumentParser()
    parser.add_argument('config_file', type=FileType('r'))
    parser.add_argument('--reset', action='store_true')
    args = parser.parse_args()

    # Parse the configuration.
    # See https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
    config_parser = ConfigParser()
    config_parser.read_file(args.config_file)
    config = dict(config_parser['default'])
    config.update(config_parser['consumer'])

    # Create Consumer instance
    consumer = Consumer(config)

    # Set up a callback to handle the '--reset' flag.
    def reset_offset(consumer, partitions):
        if args.reset:
            for p in partitions:
                p.offset = OFFSET_BEGINNING
            consumer.assign(partitions)

    # Subscribe to topic
    topic = "sensordata"
    consumer.subscribe([topic], on_assign=reset_offset)

    # InfluxDB configuration
    load_dotenv()
    token = os.environ.get("INFLUXDB_TOKEN")
    org = "UOP"
    url = "http://localhost:8086"

    write_client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
    bucket = "fypg21_alert"
    write_api = write_client.write_api(write_options=SYNCHRONOUS)

    set_device("cpu")
    device = get_device()

    s_test_predicted_list = []
    s_test_ground_list = []
    s_test_labels_list = []
    count = 0
    alert = []

    # Poll for new messages from Kafka and print them.
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                # Initial message consumption may take up to
                # `session.timeout.ms` for the consumer group to
                # balance and start consuming
                print("Waiting...")
            elif msg.error():
                print("ERROR: %s".format(msg.error()))
            else:
                # Deserialize the message from JSON
                data_json = msg.value()
                data = json.loads(data_json)

                # Process the data as needed
                x = torch.tensor(data['x'])
                y = torch.tensor(data['y'])
                labels = torch.tensor(data['labels'])
                edge_index = torch.tensor(data['edge_index'])
                fc_edge_index = torch.tensor(data['fc_edge_index'])
                feature_map = data['feature_map']

                edge_index_sets = [fc_edge_index]
                model = CAUSAL_GAT(edge_index_sets, len(feature_map),
                                   dim=64,
                                   input_dim=5,
                                   out_layer_num=1,
                                   out_layer_inter_dim=128,
                                   significance_level=0.05
                                   ).to("cpu")

                model.load_state_dict(torch.load('./pretrained/swat/best_09_23-11-00-17.pt'))
                best_model = model.to(device)

                t_test_predicted_list, t_test_ground_list, t_test_labels_list = test(best_model, x, y, labels,
                                                                                     edge_index)
                if len(s_test_predicted_list) <= 0:
                    s_test_predicted_list = t_test_predicted_list
                    s_test_ground_list = t_test_ground_list
                    s_test_labels_list = t_test_labels_list
                else:
                    s_test_predicted_list = torch.cat((s_test_predicted_list, t_test_predicted_list), dim=0)
                    s_test_ground_list = torch.cat((s_test_ground_list, t_test_ground_list), dim=0)
                    s_test_labels_list = torch.cat((s_test_labels_list, t_test_labels_list), dim=0)

                test_predicted_list = s_test_predicted_list.tolist()
                test_ground_list = s_test_ground_list.tolist()
                test_labels_list = s_test_labels_list.tolist()
                if count > 20:
                    attacks = get_score([test_predicted_list, test_ground_list, test_labels_list], feature_map)
                    if len(alert) <= 0:
                        alert = attacks

                    elif len(alert) < len(attacks):
                        alert = attacks

                count += 1

    except KeyboardInterrupt:
        pass
    finally:
        # Leave group and commit final offsets
        consumer.close()
