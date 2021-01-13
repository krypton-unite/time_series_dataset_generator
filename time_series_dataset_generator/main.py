import pytest
import numpy as np
from sklearn.metrics import r2_score  # mean_squared_error
from skorch.callbacks import EarlyStopping
from skorch.dataset import CVSplit
from time_series_models import BenchmarkLSTM
from time_series_predictor import TimeSeriesPredictor
from torch.optim import Adam
from tests.helpers import FlightSeriesDataset

if __name__ == "__main__":
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(
            initial_forget_gate_bias=1,
            hidden_dim=7,
            num_layers=1,
        ),
        lr = 5e-3,
        lambda1=1e-8,
        optimizer__weight_decay=1e-8,
        iterator_train__shuffle=True,
        early_stopping=EarlyStopping(patience=100),
        max_epochs=500,
        train_split=CVSplit(10),
        optimizer=Adam,
    )
    past_pattern_length = 24
    future_pattern_length = 12
    pattern_length = past_pattern_length + future_pattern_length
    fsd = FlightSeriesDataset(pattern_length, future_pattern_length, 1, stride=1)
    tsp.fit(fsd)

    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > 0

    netout = tsp.predict(fsd.X_test)

    idx = np.random.randint(0, len(fsd.X_test))

    y_true = fsd.y_test[idx, :, :]
    y_hat = netout[idx, :, :]
    r2s = r2_score(y_true, y_hat)
    print("Final R2 score: {}".format(r2s))
    assert r2s > -1

# if __name__ == "__main__":
#     tsp = TimeSeriesPredictor(
#         BenchmarkLSTM(
#             initial_forget_gate_bias=1,
#             hidden_dim=7,
#             num_layers=1,
#         ),
#         lr = 5e-3,
#         lambda1=1e-8,
#         optimizer__weight_decay=1e-8,
#         iterator_train__shuffle=True,
#         early_stopping=EarlyStopping(patience=100),
#         max_epochs=500,
#         train_split=CVSplit(2),
#         optimizer=Adam,
#     )
#     past_pattern_length = 24
#     future_pattern_length = 12
#     pattern_length = past_pattern_length + future_pattern_length
#     fsd = FlightSeriesDataset(pattern_length, future_pattern_length, 1)
#     tsp.fit(fsd)

#     mean_r2_score = tsp.score(tsp.dataset)
#     assert mean_r2_score > -5

#     netout = tsp.predict(fsd.X_test)

#     idx = np.random.randint(0, len(fsd.X_test))

#     y_true = fsd.y_test[idx, :, :]
#     y_hat = netout[idx, :, :]
#     r2s = r2_score(y_true, y_hat)
#     assert r2s > -1
#     print("Final R2 score: {}".format(r2s))
#     assert True