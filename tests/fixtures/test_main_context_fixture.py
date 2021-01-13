import pytest

result = {
    "auto": {
        "mean_r2_score": 0,
        "final_r2_score": -1,
        "n_cv_splits": 2
    },
    "1": {
        "mean_r2_score": 0,
        "final_r2_score": -1,
        "n_cv_splits": 10
    }
}

@pytest.fixture
def test_main_context():
    def _test_main_context(stride, pattern_length):
        res = result[str(stride)]
        if stride == 'auto':
            res['except_last_n'] = 1
        elif stride == 1:
            res['except_last_n'] = pattern_length
        return res
    
    return _test_main_context