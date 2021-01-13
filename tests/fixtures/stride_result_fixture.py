import pytest

result = {
    "auto": {
        "X_test.shape": (0,),
        "y_test.shape": (0,),
        "x.shape": (4, 24, 2),
        "y.shape": (4, 24, 1),
    },
    "1": {
        "X_test.shape": (0,),
        "y_test.shape": (0,),
        "x.shape": (109, 24, 2),
        "y.shape": (109, 24, 1),
    }
}

@pytest.fixture
def expected_stride_result():
    def _expected_stride_result(stride):
        return result[str(stride)]
    
    return _expected_stride_result