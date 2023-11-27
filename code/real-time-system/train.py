import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from sklearn.metrics import mean_squared_error
from test import *
import torch.nn.functional as F
import numpy as np


def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')
    
    return loss


def train(model=None, save_path='', config={}, train_dataloader_list=None, val_dataloader_list=None):

    seed = config['seed']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])

    train_loss_lists = []  # List to store training loss lists for each fold

    device = get_device()

    early_stop_win = 15

    model.train()

    for fold, (train_dataloader, val_dataloader) in enumerate(zip(train_dataloader_list, val_dataloader_list)):
        train_loss_list = []  # Training loss list for the current fold

        min_loss = 1e+8
        stop_improve_count = 0

        for i_epoch in range(config['epoch']):
            acu_loss = 0
            model.train()

            for x, labels, attack_labels, edge_index in train_dataloader:
                x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

                optimizer.zero_grad()
                out = model(x, labels, edge_index).float().to(device)
                loss = loss_func(out, labels)

                loss.backward()
                optimizer.step()

                train_loss_list.append(loss.item())
                acu_loss += loss.item()

            # Print the loss for the current fold and epoch
            print(f'Fold {fold + 1}, epoch ({i_epoch + 1}/{config["epoch"]}) '
                  f'(Loss:{acu_loss / len(train_dataloader):.8f}, ACU_loss:{acu_loss})', flush=True)

            # Early stopping based on validation loss
            if val_dataloader is not None:
                val_loss, val_result = test(model, val_dataloader)

                if val_loss < min_loss:
                    torch.save(model.state_dict(), save_path)

                    min_loss = val_loss
                    stop_improve_count = 0
                else:
                    stop_improve_count += 1

                if stop_improve_count >= early_stop_win:
                    print(f'Stop improve count (Fold {fold + 1}): {stop_improve_count}')
                    break

        train_loss_lists.append(train_loss_list)  # Store training loss list for the current fold

    # Calculate and print the average training loss over all folds
    avg_train_loss = np.mean([np.min(losses) for losses in train_loss_lists])
    print(f"Average Training Loss over {len(train_dataloader_list)} Folds: {avg_train_loss}")

    return train_loss_lists  # Return a list of training loss lists for all folds
