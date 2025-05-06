import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from train_with_sampling import *
from DataLoader import *
import torch.nn as nn
import torch
from helpers import *
from inference import *


def main(
        epoch: int = 200,
        k: int = 60,
        batch_size: int = 1,
        frequency: int = 100,
        training_length=48,
        forecast_window=24,
        train_csv="Data/temperature_train_data.csv",
        test_csv="Data/temperature_test_data.csv",
        # train_csv="Data/humidity_train_data.csv",
        # test_csv="Data/humidity_train_data.csv",

        path_to_save_model="save_model/",
        path_to_save_loss="save_loss/",
        path_to_save_predictions="save_predictions/",
        device="cuda:0"
):
    train_dataset = SensorDataset(csv_name=train_csv, root_dir="", training_length=training_length,
                                  forecast_window=forecast_window)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = SensorDataset(csv_name=test_csv, root_dir="", training_length=training_length,
                                 forecast_window=forecast_window)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    feature_size = 7
    adj = torch.tensor(get_mic_adj(train_csv, 3)).to(device)
    train_station_number = 8

    best_model = train(train_dataloader, epoch, path_to_save_model, path_to_save_loss,
                       path_to_save_predictions, device, feature_size, adj, train_station_number)

    inference(path_to_save_predictions, forecast_window, test_dataloader, device, path_to_save_model, best_model,
              feature_size, adj, train_station_number)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--k", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frequency", type=int, default=100)
    parser.add_argument("--path_to_save_model", type=str, default="save_model/")
    parser.add_argument("--path_to_save_loss", type=str, default="save_loss/")
    parser.add_argument("--path_to_save_predictions", type=str, default="save_predictions/")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    main(
        epoch=args.epoch,
        k=args.k,
        batch_size=args.batch_size,
        frequency=args.frequency,
        path_to_save_model=args.path_to_save_model,
        path_to_save_loss=args.path_to_save_loss,
        path_to_save_predictions=args.path_to_save_predictions,
        device=args.device,
    )
