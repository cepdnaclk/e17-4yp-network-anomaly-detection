from util.data import get_attack_interval
import time
from datetime import datetime
from pytz import utc, timezone
from util.time import timestamp2str
import json
import argparse
import numpy as np


def printsep():
    print('='*40+'\n')


def save_attack_infos(top1_best_info, total_err_scores, labels, names, save_path, dataset, config):
    slide_win=config['slide_win']
    down_len=10
    
    if dataset == 'wadi' or dataset == 'msl':
        s = '09/10/2017 18:00:00'
    elif dataset == 'swat':
        s = '28/12/2015 10:00:00'
    start_s = int(time.mktime(datetime.strptime(s, "%d/%m/%Y %H:%M:%S").timetuple()))
    cst8 = timezone('Asia/Shanghai')
    fmt = '%m/%d %H:%M:%S'

    attack_inters = get_attack_interval(labels)

    save_infos = {
        'f1_score': top1_best_info[0],
        'precision': top1_best_info[1],
        'recall' : top1_best_info[2],
        'confusion_matrix_TN' : int(top1_best_info[4][0][0]),
        'confusion_matrix_FP' : int(top1_best_info[4][0][1]),
        'confusion_matrix_FN' : int(top1_best_info[4][1][0]),
        'confusion_matrix_TP' : int(top1_best_info[4][1][1]),
        'threshold' : top1_best_info[5],
        'attacks': []
    }

    indices_map = names

    indices = np.argmax(total_err_scores, axis=0).tolist()
    anomaly_sensors = [ indices_map[index] for index in indices ]

    topk = 5
    topk_indices = np.argpartition(total_err_scores, -topk, axis=0)[-topk:]
    topk_indices = np.transpose(topk_indices)

    topk_anomaly_sensors = []
    topk_err_score_map=[]
    for i, indexs in enumerate(topk_indices):
        # print(indexs)
        topk_anomaly_sensors.append([indices_map[index] for index in indexs])

        item = {}
        for sensor, index in zip(topk_anomaly_sensors[i],indexs):
            if sensor not in item:
                item[sensor] = total_err_scores[index, i]

        topk_err_score_map.append(item)

    for head, end in attack_inters:
        attack_infos = {}
        topk_attack_infos = {}

        head_t = timestamp2str(start_s+(head+slide_win)*down_len, fmt, cst8)
        end_t = timestamp2str(start_s+(end+slide_win)*down_len, fmt, cst8)
        # head_t = datetime.fromtimestamp(start_s+head).astimezone(cst8).strftime(fmt)
        # end_t = datetime.fromtimestamp(start_s+end).astimezone(cst8).strftime(fmt)

        # print(f'\nattack from {head_t} to {end_t}:')
        
        for i in range(head, end):
            # t = datetime.fromtimestamp(start_s+i).astimezone(cst8).strftime(fmt)
            t = timestamp2str(start_s+(i+slide_win)*down_len, fmt, cst8)
            max_sensor = anomaly_sensors[i]
            topk_sensors = topk_anomaly_sensors[i]

            if max_sensor not in attack_infos:
                attack_infos[max_sensor] = 0
            attack_infos[max_sensor] += 1

            # for anomaly_sensor in topk_sensors:
            #     if anomaly_sensor not in topk_attack_infos:
            #         topk_attack_infos[anomaly_sensor] = 0
            #     topk_attack_infos[anomaly_sensor] += 1
            
            for anomaly_sensor in topk_sensors:
                if anomaly_sensor not in topk_attack_infos:
                    topk_attack_infos[anomaly_sensor] = 0
                topk_attack_infos[anomaly_sensor] += topk_err_score_map[i][anomaly_sensor] 

        # print('-------------------------------')
        # print(f'total top 5 attack sensors from {head_t} to {end_t}:')
        sorted_attack_infos = {k: v for k, v in sorted(attack_infos.items(), reverse=True, key=lambda item: item[1])}
        sorted_topk_attack_infos = {k: v for k, v in sorted(topk_attack_infos.items(), reverse=True, key=lambda item: item[1])}
        # for key, count in sorted_attack_infos.items()[:5]:
        #     print(key, count)

        save_infos['attacks'].append({
            'start': head_t,
            'end': end_t,
            'sensors': list(sorted_attack_infos),
            'sensors_values': list(sorted_attack_infos.values()),
            'topk_sensors': list(sorted_topk_attack_infos),
            'topk_scores': list(sorted_topk_attack_infos.values())
        })

    with open(save_path, 'w+') as outfile:
        json.dump(save_infos, outfile, indent=4)  
