import pytest
import numpy as np
from pathlib import Path
import os

filepath = Path(os.path.split(__file__)[0])

result = {}
with \
    open(Path.joinpath(filepath, 'size_stride_1_x.npy'), 'rb') as f_1_x, \
    open(Path.joinpath(filepath, 'size_stride_1_y.npy'), 'rb') as f_1_y, \
    open(Path.joinpath(filepath, 'size_stride_auto_x.npy'), 'rb') as f_auto_x, \
    open(Path.joinpath(filepath, 'size_stride_auto_y.npy'), 'rb') as f_auto_y:
    result = {
        "auto": {
            'x': np.load(f_auto_x, allow_pickle = True),
            'y': np.load(f_auto_y, allow_pickle = True),
        },
        "1": {
            'x': np.load(f_1_x, allow_pickle = True),
            'y': np.load(f_1_y, allow_pickle = True),
        }
}

@pytest.fixture
def expected_stride_result():
    def _expected_stride_result(stride):
        return result[str(stride)]
    
    return _expected_stride_result