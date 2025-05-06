from model import STGAMAM
import logging
from plot import *
from joblib import load
from helpers import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
                    datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)


def inference(path_to_save_predictions, forecast_window, data_loader, device, path_to_save_model, best_model,
              feature_size, adj, train_station_number):
    device = torch.device(device)
    nNode = len(adj)

    model = STGAMAM(feature_size=feature_size, adj=adj).double().to(device)
    model.load_state_dict(torch.load(path_to_save_model + best_model))
    scaler = load('scalar_item.joblib')

    criterion_MAE = torch.nn.L1Loss()
    criterion_MSE = torch.nn.MSELoss()

    val_MAE_loss = 0
    val_MAPE_loss = 0
    val_RMSE_loss = 0

    k = 3
    with torch.no_grad():

        model.eval()
        for plot in range(25):

            for index_in, index_tar, _input, target, sensor_number in data_loader:

                _input = _input.permute(1, 0, 2).double().to(device)
                target = target.permute(1, 0, 2).double().to(device)

                src = _input

                next_input_model = src
                all_predictions = []

                for i in range(forecast_window - 1):

                    st_fusion_adj = construct_st_adj_dtw(adj, k, next_input_model).to(device)
                    prediction = model(next_input_model, st_fusion_adj, device)
                    prediction = torch.unsqueeze(prediction, dim=1)

                    if all_predictions == []:
                        all_predictions = prediction
                    else:
                        all_predictions = torch.cat((all_predictions, prediction), dim=0)

                    pos_encodings = target[i * nNode: i * nNode + nNode, :, 1:]
                    next_input_model = torch.cat(
                        (next_input_model[nNode:, :, :], torch.cat((prediction, pos_encodings), dim=2)))

                st_fusion_adj = construct_st_adj_dtw(adj, k, next_input_model).to(device)
                prediction = model(next_input_model, st_fusion_adj, device)
                all_predictions = torch.cat((all_predictions, torch.unsqueeze(prediction, dim=1)))

                true = torch.Tensor(scaler.inverse_transform(target[train_station_number - 1::nNode, :, 0].cpu()))
                val_predictions = torch.Tensor(scaler.inverse_transform(all_predictions[train_station_number - 1::nNode, :, 0].cpu()))

                MAE_loss = criterion_MAE(true, val_predictions)
                MAPE_loss = calculate_MAPE(true, val_predictions)
                MSE_loss = criterion_MSE(true, val_predictions)

                val_MAE_loss += MAE_loss
                val_MAPE_loss += MAPE_loss
                val_RMSE_loss += MSE_loss.sqrt()

                logger.info(f"plot: {plot}, MSE_loss: {MSE_loss}")

            src_humidity = scaler.inverse_transform(src[:, :, 0].cpu())
            target_humidity = scaler.inverse_transform(target[:, :, 0].cpu())
            prediction_humidity = scaler.inverse_transform(all_predictions[:, :, 0].detach().cpu().numpy())
            plot_prediction_station(plot, path_to_save_predictions, src_humidity, target_humidity, prediction_humidity,
                                    sensor_number, train_station_number, index_in, index_tar)

        val_MAE_loss = val_MAE_loss / 25
        val_MAPE_loss = val_MAPE_loss / 25
        val_RMSE_loss = val_RMSE_loss / 25

        logger.info(f"Average MAE On Unseen Dataset: {val_MAE_loss.item()}")
        logger.info(f"Average MAPE On Unseen Dataset: {val_MAPE_loss.item()}")
        logger.info(f"Average RMSE On Unseen Dataset: {val_RMSE_loss.item()}")


def calculate_MAPE(actual_values, predicted_values):
    if len(actual_values) != len(predicted_values):
        raise ValueError("The lengths of actual_values and predicted_values should be the same.")

    absolute_errors = []
    for actual, predicted in zip(actual_values, predicted_values):
        absolute_errors.append(abs((actual - predicted) / actual))

    MAPE = (sum(absolute_errors) / len(actual_values)) * 100
    return MAPE