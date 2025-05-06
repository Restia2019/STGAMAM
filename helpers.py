import os, shutil
import pandas as pd
import math
import numpy as np
import torch
from minepy import MINE
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def log_loss(loss_val: float, path_to_save_loss: str, train: bool = True):
    if train:
        file_name = "train_loss.txt"
    else:
        file_name = "val_loss.txt"

    path_to_file = path_to_save_loss + file_name
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "a") as f:
        f.write(str(loss_val) + "\n")
        f.close()


def EMA(values, alpha=0.1):
    ema_values = [values[0]]
    for idx, item in enumerate(values[1:]):
        ema_values.append(alpha * item + (1 - alpha) * ema_values[idx])
    return ema_values


def clean_directory():
    if os.path.exists('save_loss'):
        shutil.rmtree('save_loss')
    if os.path.exists('save_model'):
        shutil.rmtree('save_model')
    if os.path.exists('save_predictions'):
        shutil.rmtree('save_predictions')
    os.mkdir("save_loss")
    os.mkdir("save_model")
    os.mkdir("save_predictions")


def clean_save_model():
    if os.path.exists('save_model'):
        shutil.rmtree('save_model')
    os.mkdir("save_model")


def get_adj():
    data = pd.read_csv('Data/city_attributes.csv')

    cities = data['City'].tolist()

    dist_matrix = [[0] * len(cities) for _ in range(len(cities))]

    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            city1 = (data.loc[i, 'Latitude'], data.loc[i, 'Longitude'])
            city2 = (data.loc[j, 'Latitude'], data.loc[j, 'Longitude'])
            distance = calculate_distance(city1[0], city1[1], city2[0], city2[1])
            if distance < 150:
                dist_matrix[i][j] = 1
                dist_matrix[j][i] = 1
    return dist_matrix


def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_station_k_biggest_mic(humidity_by_station, station_ids, k, station_id):
    mine = MINE(alpha=0.6, c=15)

    correlations = np.zeros((len(station_ids) + 1))
    x = humidity_by_station[station_id]

    for i in range(1, len(station_ids) + 1):
        y = humidity_by_station[i]
        mine.compute_score(x, y)
        correlations[i] = mine.mic()

    non_zero_indices = np.nonzero(correlations)

    biggest_k_indices = np.argsort(correlations[non_zero_indices])[-k - 1:]
    biggest_k_indices = biggest_k_indices + 1

    return biggest_k_indices[:k]


def get_mic_adj(dataDir, k):
    data = np.genfromtxt(dataDir, delimiter=',', skip_header=1, usecols=(0, 2), dtype=None)

    station_ids = np.unique(data['f0'])

    humidity_by_station = {}

    for station_id in station_ids:
        humidity = data[data['f0'] == station_id]['f1']
        humidity_by_station[station_id] = humidity

    mic_adj = [[0] * len(station_ids) for _ in range(len(station_ids))]
    for i in range(1, len(station_ids) + 1):
        biggest_k_station_ids = compute_station_k_biggest_mic(humidity_by_station, station_ids, k, i)
        for station_id in biggest_k_station_ids:
            mic_adj[i - 1][station_id - 1] = 1

    d_adj = get_adj()
    for i in range(len(station_ids)):
        for j in range(len(station_ids)):
            if mic_adj[i][j] == 1:
                d_adj[i][j] = 1
    return d_adj


def construct_st_adj(A, steps):
    N = len(A)
    adj = torch.zeros(N * steps, N * steps)

    for i in range(steps):
        adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A

    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1

    return adj


def construct_st_adj_dtw(A, steps, next_input_model):
    N = len(A)
    adj = torch.zeros(N * steps, N * steps)

    for i in range(steps):
        adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A

    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1

    distances = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            dtw_distance, path = fastdtw(next_input_model[i::48, -1, -1].cpu(),
                                         next_input_model[j::48, -1, -1].cpu())
            distances[i, j] = dtw_distance
            distances[j, i] = dtw_distance

    adj[0:N, 0:N] = 0
    for i in range(N):
        row_distances = distances[i]
        k_min_indices = np.argsort(row_distances)[1: steps + 1]
        adj[i, k_min_indices] = 1

    adj[0:N, -N:] = adj[0:N, 0:N]
    adj[-N:, 0:N] = adj[0:N, 0:N]
    adj[-N:, -N:] = adj[0:N, 0:N]

    return adj
