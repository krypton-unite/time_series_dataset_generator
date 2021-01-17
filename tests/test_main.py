import pytest
import numpy as np
from sklearn.metrics import r2_score  # mean_squared_error
from skorch.callbacks import EarlyStopping
from skorch.dataset import CVSplit
from time_series_models import BenchmarkLSTM
from time_series_predictor import TimeSeriesPredictor
from torch.optim import Adam

from .helpers import FlightSeriesDataset
from .fixtures import expected_stride_result, test_main_context

# @pytest.mark.skip
@pytest.mark.usefixtures('expected_stride_result')
@pytest.mark.parametrize('stride', ['auto', 1])
def test_size_stride(stride, expected_stride_result):
    expected_result = expected_stride_result(stride)
    past_pattern_length = 24
    future_pattern_length = 12
    pattern_length = past_pattern_length + future_pattern_length
    fsd = FlightSeriesDataset(pattern_length, future_pattern_length, 0, stride = stride)
    assert fsd.test == None
    assert (fsd.x == expected_result['x']).all()
    assert (fsd.y == expected_result['y']).all()

# @pytest.mark.skip
@pytest.mark.usefixtures('test_main_context')
@pytest.mark.parametrize('stride', ['auto', 1])
def test_main(stride, test_main_context):
    context = test_main_context(stride)
    past_pattern_length = context['past_pattern_length']
    future_pattern_length = context['future_pattern_length']
    pattern_length = past_pattern_length + future_pattern_length
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(
            initial_forget_gate_bias=1,
            hidden_dim=7,
            num_layers=1,
        ),
        lr = context['lr'],
        lambda1=1e-8,
        optimizer__weight_decay=1e-8,
        iterator_train__shuffle=True,
        early_stopping=EarlyStopping(patience=100),
        max_epochs=500,
        train_split=CVSplit(context['n_cv_splits']),
        optimizer=Adam,
    )
    fsd = FlightSeriesDataset(pattern_length, future_pattern_length, context['except_last_n'], stride=stride, generate_test_dataset=True)
    tsp.fit(fsd)

    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > context['mean_r2_score']

    netout = tsp.predict(fsd.test.x)

    idx = np.random.randint(0, len(fsd.test.x))

    y_true = fsd.test.y[idx, :, :]
    y_hat = netout[idx, :, :]
    r2s = r2_score(y_true, y_hat)
    print("Final R2 score: {}".format(r2s))
    assert r2s > context['final_r2_score']
