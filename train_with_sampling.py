from model import STGAMAM
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from DataLoader import SensorDataset
import logging
import time  # debugging
from plot import *
from helpers import *
from joblib import load
from icecream import ic
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math, random

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
                    datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)


def flip_from_probability(p):
    return True if random.random() < p else False


def train(dataloader, EPOCH, path_to_save_model, path_to_save_loss, path_to_save_predictions, device,
          feature_size, adj, train_station_number):

    global src, target, all_predictions, sensor_number, index_in, index_tar
    device = torch.device(device)
    print("---device---", device)

    model = STGAMAM(feature_size=feature_size, adj=adj).double().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    best_model = ""
    min_train_loss = float('inf')
    nNode = len(adj)
    k = 3

    for epoch in range(EPOCH + 1):
        train_loss = 0
        val_loss = 0

        model.train()
        for index_in, index_tar, _input, target, sensor_number in dataloader:

            optimizer.zero_grad()

            _input = _input.permute(1, 0, 2).double().to(device)
            target = target.permute(1, 0, 2).double().to(device)

            src = _input

            next_input_model = src
            all_predictions = []

            for i in range(24 - 1):

                st_fusion_adj = construct_st_adj_dtw(adj, k, next_input_model).to(device)
                prediction = model(next_input_model, st_fusion_adj, device)
                prediction = torch.unsqueeze(prediction, dim=1)

                if all_predictions == []:
                    all_predictions = prediction
                else:
                    all_predictions = torch.cat((all_predictions, prediction), dim=0)

                new_timestep = target[i * nNode: i * nNode + nNode, :, :]
                next_input_model = torch.cat((next_input_model[nNode:, :, :], new_timestep))

            st_fusion_adj = construct_st_adj_dtw(adj, k, next_input_model).to(device)
            prediction = model(next_input_model, st_fusion_adj, device)
            all_predictions = torch.cat((all_predictions, torch.unsqueeze(prediction, dim=1)))

            loss = criterion(target[train_station_number - 1::nNode, :, 0].unsqueeze(-1),
                             all_predictions[train_station_number - 1::nNode])
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

        if train_loss < min_train_loss:
            clean_save_model()
            torch.save(model.state_dict(), path_to_save_model + f"best_train_{epoch}.pth")
            torch.save(optimizer.state_dict(), path_to_save_model + f"optimizer_{epoch}.pth")
            best_model = f"best_train_{epoch}.pth"
            min_train_loss = train_loss

        if epoch % 1 == 0:

            logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")
            scaler = load('scalar_item.joblib')

            src_humidity = scaler.inverse_transform(src[:, :, 0].cpu())
            target_humidity = scaler.inverse_transform(target[:, :, 0].cpu())
            prediction_humidity = scaler.inverse_transform(
                all_predictions[:, :, 0].detach().cpu().numpy())
            plot_training_station(epoch, path_to_save_predictions, src_humidity, target_humidity, prediction_humidity,
                                  sensor_number, train_station_number, index_in, index_tar)

        train_loss = train_loss / len(dataloader)
        log_loss(train_loss, path_to_save_loss, train=True)

    plot_loss(path_to_save_loss, train=True)
    return best_model
