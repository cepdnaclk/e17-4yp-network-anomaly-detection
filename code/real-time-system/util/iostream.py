import os

from dotenv import load_dotenv
from influxdb_client import Point
from influxdb_client.client import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

from util.data import get_attack_interval
import time
from datetime import datetime
from pytz import timezone

from util.net_struct import get_feature_map
from util.time import timestamp2str
import numpy as np


def printsep():
    print('=' * 40 + '\n')


def save_attack_infos(top1_best_info, total_err_scores, labels, names, dataset):
    load_dotenv()
    token = os.environ.get("INFLUXDB_TOKEN")
    org = "UOP"
    url = "http://localhost:8086"

    write_client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
    bucket = "fypg21"
    write_api = write_client.write_api(write_options=SYNCHRONOUS)

    slide_win = 5
    down_len = 10

    if dataset == 'wadi' or dataset == 'msl':
        s = '09/10/2017 18:00:00'
    elif dataset == 'swat':
        s = '28/12/2015 10:00:00'
    start_s = int(time.mktime(datetime.strptime(s, "%d/%m/%Y %H:%M:%S").timetuple()))
    cst8 = timezone('Asia/Shanghai')
    fmt = '%m/%d %H:%M:%S'

    features = get_feature_map("swat")

    # Saving test scores in InfluxDB
    #######################################################
    latest_test_scores = total_err_scores[:, -1]
    for index, score in enumerate(latest_test_scores):
        point = (
            Point(features[index]).field("threshold", top1_best_info[1]).field("sensor_value", score)
        )
        write_api.write(bucket=bucket, org="UOP", record=point)
    #######################################################

    attack_inters = get_attack_interval(labels)
    indices_map = names

    indices = np.argmax(total_err_scores, axis=0).tolist()
    anomaly_sensors = [indices_map[index] for index in indices]

    topk = 5
    topk_indices = np.argpartition(total_err_scores, -topk, axis=0)[-topk:]
    topk_indices = np.transpose(topk_indices)

    topk_anomaly_sensors = []
    topk_err_score_map = []

    for i, indexs in enumerate(topk_indices):
        # print(indexs)
        topk_anomaly_sensors.append([indices_map[index] for index in indexs])

        item = {}
        for sensor, index in zip(topk_anomaly_sensors[i], indexs):
            if sensor not in item:
                item[sensor] = total_err_scores[index, i]

        topk_err_score_map.append(item)

    attacks = []
    for head, end in attack_inters:
        attack_infos = {}
        topk_attack_infos = {}

        head_t = timestamp2str(start_s + (head + slide_win) * down_len, fmt, cst8)
        end_t = timestamp2str(start_s + (end + slide_win) * down_len, fmt, cst8)

        for i in range(head, end):
            max_sensor = anomaly_sensors[i]
            topk_sensors = topk_anomaly_sensors[i]

            if max_sensor not in attack_infos:
                attack_infos[max_sensor] = 0
            attack_infos[max_sensor] += 1

            for anomaly_sensor in topk_sensors:
                if anomaly_sensor not in topk_attack_infos:
                    topk_attack_infos[anomaly_sensor] = 0
                topk_attack_infos[anomaly_sensor] += topk_err_score_map[i][anomaly_sensor]

        sorted_attack_infos = {k: v for k, v in sorted(attack_infos.items(), reverse=True, key=lambda item: item[1])}
        sorted_topk_attack_infos = {k: v for k, v in
                                    sorted(topk_attack_infos.items(), reverse=True, key=lambda item: item[1])}

        save_infos = {
            'threshold': top1_best_info[1],
            'start': head_t,
            'end': end_t,
            # 'sensors': list(sorted_attack_infos),
            # 'sensors_values': list(sorted_attack_infos.values()),
            'topk_sensors': list(sorted_topk_attack_infos),
            'topk_scores': list(sorted_topk_attack_infos.values())
        }
        attacks.append(save_infos)
    return attacks
