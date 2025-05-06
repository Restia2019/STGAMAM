import matplotlib.pyplot as plt
from helpers import EMA
from icecream import ic
import numpy as np
import torch


def plot_loss(path_to_save, train=True):
    plt.rcParams.update({'font.size': 10})
    with open(path_to_save + "/train_loss.txt", 'r') as f:
        loss_list = [float(line) for line in f.readlines()]
    if train:
        title = "Train"
    else:
        title = "Validation"
    EMA_loss = EMA(loss_list)
    plt.plot(loss_list, label="loss")
    plt.plot(EMA_loss, label="EMA loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(title + "_loss")
    plt.savefig(path_to_save + f"/{title}.png")
    plt.close()


def plot_prediction(title, path_to_save, src, tgt, prediction, sensor_number, index_in, index_tar):
    idx_scr = index_in[0, :].tolist()
    idx_tgt = index_tar[0].tolist()
    idx_pred = idx_tgt

    for i in range(len(sensor_number)):
        new_src = src[i::len(sensor_number), :]
        new_tgt = tgt[i::len(sensor_number), :]
        new_prediction = prediction[i::len(sensor_number), :]
        new_title = "{}_{}".format(title, i + 1)

        plt.figure(figsize=(15, 6))
        plt.rcParams.update({"font.size": 16})

        plt.plot(idx_scr, new_src, '-', color='blue', label='Input', linewidth=2)
        plt.plot(idx_tgt, new_tgt, '-', color='indigo', label='Target', linewidth=2)
        plt.plot(idx_pred, new_prediction, '--', color='limegreen', label='Forecast', linewidth=2)

        plt.grid(b=True, which='major', linestyle='solid')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', linestyle='dashed', alpha=0.5)
        plt.xlabel("Time Elapsed")
        plt.ylabel("Humidity (%)")
        plt.legend()
        plt.title("Forecast from Sensor " + str(sensor_number[i]))

        plt.savefig(path_to_save + f"Prediction_{new_title}.png")
        plt.close()


def plot_prediction_station(title, path_to_save, src, tgt, prediction, sensor_list, train_station_number, index_in, index_tar):
    idx_scr = index_in[0, :].tolist()
    idx_tgt = index_tar[0].tolist()
    idx_pred = idx_tgt

    new_src = src[train_station_number - 1::len(sensor_list), :]
    new_tgt = tgt[train_station_number - 1::len(sensor_list), :]
    new_prediction = prediction[train_station_number - 1::len(sensor_list), :]

    plt.figure(figsize=(15, 6))
    plt.rcParams.update({"font.size": 16})

    plt.plot(idx_scr, new_src, 'o-.', color='blue', label='Input', linewidth=1)
    plt.plot(idx_tgt, new_tgt, 'o-.', color='red', label='Target', linewidth=1)
    plt.plot(idx_pred, new_prediction, 'o-.', color='limegreen', label='Forecast', linewidth=1)

    plt.grid(b=True, which='major', linestyle='solid')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', linestyle='dashed', alpha=0.5)
    plt.xlabel("Time Slot")
    plt.ylabel("Temperature (K)")
    plt.legend()
    plt.title("Forecast from Station{}".format(train_station_number))

    plt.savefig(path_to_save + f"Prediction_{title}.png")
    plt.close()


def plot_training(epoch, path_to_save, src, prediction, sensor_number, index_in, index_tar):

    idx_scr = [i for i in range(len(src))]
    idx_pred = [i for i in range(1, len(prediction) + 1)]

    plt.figure(figsize=(15, 6))
    plt.rcParams.update({"font.size": 18})
    plt.grid(b=True, which='major', linestyle='-')
    plt.grid(b=True, which='minor', linestyle='--', alpha=0.5)
    plt.minorticks_on()

    plt.plot(idx_scr, src, 'o-.', color='blue', label='input sequence', linewidth=1)
    plt.plot(idx_pred, prediction, 'o-.', color='limegreen', label='prediction sequence', linewidth=1)

    plt.title("Teaching Forcing from Sensor " + str(sensor_number[0]) + ", Epoch " + str(epoch))
    plt.xlabel("Time Elapsed")
    plt.ylabel("Humidity (%)")
    plt.legend()
    plt.savefig(path_to_save + f"/Epoch_{str(epoch)}.png")
    plt.close()


def plot_training_3_old(epoch, path_to_save, src, sampled_src, prediction, sensor_number, index_in, index_tar):
    idx_scr = index_in.tolist()[0][:-1]
    idx_sampled_src = index_in.tolist()[0][:-1]
    idx_pred = index_in.tolist()[0][1:]

    for i in range(len(sensor_number)):
        plt.figure(figsize=(15, 6))
        plt.rcParams.update({"font.size": 18})
        plt.grid(b=True, which='major', linestyle='-')
        plt.grid(b=True, which='minor', linestyle='--', alpha=0.5)
        plt.minorticks_on()

        new_sampled_src = sampled_src[i::len(sensor_number), :]
        new_src = src[i::len(sensor_number), :]
        new_prediction = prediction[i::len(sensor_number), :]

        plt.plot(idx_sampled_src, new_sampled_src, 'o-.', color='red', label='sampled source', linewidth=1,
                 markersize=10)
        plt.plot(idx_scr, new_src, 'o-.', color='blue', label='input sequence', linewidth=1)
        plt.plot(idx_pred, new_prediction, 'o-.', color='limegreen', label='prediction sequence', linewidth=1)

        plt.title("Teaching Forcing from Sensor " + str(sensor_number[i]) + ", Epoch " + str(epoch))
        plt.xlabel("Time Elapsed")
        plt.ylabel("Humidity (%)")
        plt.legend()
        plt.savefig(path_to_save + f"/Epoch_{str(epoch)}_{i + 1}.png")
        plt.close()


def plot_training_3(epoch, path_to_save, src, tgt, prediction, sensor_number, index_in, index_tar):
    idx_scr = index_in.tolist()[0]
    idx_tgt = index_tar.tolist()[0]
    idx_pred = index_tar.tolist()[0]

    for i in range(len(sensor_number)):
        plt.figure(figsize=(15, 6))
        plt.rcParams.update({"font.size": 18})
        plt.grid(b=True, which='major', linestyle='-')
        plt.grid(b=True, which='minor', linestyle='--', alpha=0.5)
        plt.minorticks_on()

        new_src = src[i::len(sensor_number), :]
        new_tgt = tgt[i::len(sensor_number), :]
        new_prediction = prediction[i::len(sensor_number), :]

        plt.plot(idx_scr, new_src, 'o-.', color='blue', label='input sequence', linewidth=1)
        plt.plot(idx_tgt, new_tgt, 'o-.', color='red', label='ground truth sequence', linewidth=1,
                 markersize=10)
        plt.plot(idx_pred, new_prediction, 'o-.', color='limegreen', label='prediction sequence', linewidth=1)

        plt.title("Train with sampling, Sensor " + str(sensor_number[i]) + ", Epoch " + str(epoch))
        plt.xlabel("Time Elapsed")
        plt.ylabel("Humidity (%)")
        plt.legend()
        plt.savefig(path_to_save + f"/Epoch_{str(epoch)}_{i + 1}.png")
        plt.close()


def plot_training_station(epoch, path_to_save, src, tgt, prediction, sensor_list, train_station_number, index_in, index_tar):
    idx_scr = index_in.tolist()[0]
    idx_tgt = index_tar.tolist()[0]
    idx_pred = index_tar.tolist()[0]

    plt.figure(figsize=(15, 6))
    plt.rcParams.update({"font.size": 18})
    plt.grid(b=True, which='major', linestyle='-')
    plt.grid(b=True, which='minor', linestyle='--', alpha=0.5)
    plt.minorticks_on()

    new_src = src[train_station_number - 1::len(sensor_list), :]
    new_tgt = tgt[train_station_number - 1::len(sensor_list), :]
    new_prediction = prediction[train_station_number - 1::len(sensor_list), :]

    plt.plot(idx_scr, new_src, 'o-.', color='blue', label='input sequence', linewidth=1)
    plt.plot(idx_tgt, new_tgt, 'o-.', color='red', label='ground truth sequence', linewidth=1,
             markersize=10)
    plt.plot(idx_pred, new_prediction, 'o-.', color='limegreen', label='prediction sequence', linewidth=1)

    plt.title("Train with sampling, Station " + str(train_station_number) + ", Epoch " + str(epoch))
    plt.xlabel("Time Elapsed")
    plt.ylabel("Humidity (%)")
    plt.legend()
    plt.savefig(path_to_save + f"/Epoch_{str(epoch)}.png")
    plt.close()


def plot_param(epoch, model_params):
    path_to_save = 'save_model/'
    param_values = [param_value.cpu().numpy() for param_value in model_params.values()]
    param_names = list(model_params.keys())

    plt.figure(figsize=(10, 6))
    for param_name, param_value in zip(param_names, param_values):
        plt.plot(param_value, label=param_name)
    plt.xlabel('Iterations')
    plt.ylabel('Parameter Value')
    plt.legend()
    plt.savefig(path_to_save + f"/Epoch_{str(epoch)}.png")
