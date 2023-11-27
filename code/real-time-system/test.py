import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *

import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F

from util.data import *
from util.preprocess import *


def test(model, x, y, labels, edge_index):
    # test
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()
    model.eval()

    # for x, y, labels, edge_index in dataloader:
    x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]

    with torch.no_grad():
        predicted = model(x, y).float().to(device)
        loss = loss_func(predicted, y)

        labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

        t_test_predicted_list = predicted
        t_test_ground_list = y
        t_test_labels_list = labels

    return t_test_predicted_list, t_test_ground_list, t_test_labels_list
