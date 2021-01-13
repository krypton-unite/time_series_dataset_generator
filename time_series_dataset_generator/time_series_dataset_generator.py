import numpy as np
from time_series_generator import TimeseriesGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
from time_series_dataset import TimeSeriesDataset
import sys

def make_predictor(values, features_labels):
    def _raw_make_predictor(features):
        #pylint: disable=too-many-function-args
        return np.dstack(features).astype(np.float32)
    def _make_features(values, features_labels):
        return [values[label] for label in features_labels]
    features = _make_features(values, features_labels)
    return _raw_make_predictor(features), [feature.Name for feature in features]

def make_time_series_dataset(input_df, pattern_length, n_to_predict, input_features_labels, output_features_labels, except_last_n, batch_size = sys.maxsize, augmentation=0, stride='auto', shuffle=False, overlap = 0):
    def _make_regression(cd, pattern_length, input_features_labels, output_features_labels, n_to_predict, batch_size, augmentation, stride, shuffle):
        def generate_timeseries(cd, pattern_length, n_to_predict, batch_size, augmentation, stride, shuffle):
            tg = TimeseriesGenerator(
                input_df,
                input_df,
                pattern_length - n_to_predict, # past_pattern_length
                length_output = n_to_predict,
                batch_size=batch_size,
                augmentation=augmentation,
                stride=pattern_length + augmentation if stride == 'auto' else stride,
                shuffle=shuffle,
                overlap=overlap
            )

            x, y = tg[0]
            
            labels = list(cd)
            input_values = {}
            output_values = {}
            for idx, a_label in enumerate(labels):
                input_values[a_label] = pd.DataFrame(x[:,:,idx])
                output_values[a_label] = pd.DataFrame(y[:,:,idx])
                input_values[a_label].Name = a_label
                output_values[a_label].Name = a_label
            return input_values, output_values

        input_values, output_values = generate_timeseries(cd, pattern_length, n_to_predict, batch_size, augmentation, stride, shuffle)
        
        _x, labels_x = make_predictor( input_values,  input_features_labels)
        _y, labels_y = make_predictor(output_values, output_features_labels)

        return _x, _y, {'x': labels_x, 'y': labels_y}
    _x, _y, labels = _make_regression(
        input_df,
        pattern_length,
        input_features_labels,
        output_features_labels,
        n_to_predict,
        batch_size,
        augmentation,
        stride,
        shuffle
    )

    if except_last_n == 0:
        X_train = _x
        y_train = _y
        X_test = np.array([])
        y_test = X_test
    else:    
        X_train, X_test, y_train, y_test = train_test_split(_x, _y, test_size=except_last_n, shuffle=False)

    tsd = TimeSeriesDataset(X_train, y_train, labels)

    tsd.X_test = X_test
    tsd.y_test = y_test

    return tsd