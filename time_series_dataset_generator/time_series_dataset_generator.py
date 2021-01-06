import numpy as np
from time_series_generator import TimeseriesGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
from time_series_dataset import TimeSeriesDataset

def make_predictor(values, features_labels):
    def _raw_make_predictor(features):
        #pylint: disable=too-many-function-args
        return np.dstack(features).astype(np.float32)
    def _make_features(values, features_labels):
        return [values[label] for label in features_labels]
    features = _make_features(values, features_labels)
    return _raw_make_predictor(features), [feature.Name for feature in features]

def make_time_series_dataset(input_df, pattern_length, n_to_predict, input_features_labels, output_features_labels, except_last_n, data_augmentation=None):
    def _make_regression(cd, pattern_length, input_features_labels, output_features_labels, n_to_predict, data_augmentation):
        def _generate_samples(cd, pattern_length, n_to_predict, data_augmentation):
            number_of_training_examples = len(cd) - pattern_length + 1 - 2*data_augmentation
            if number_of_training_examples < 1:
                number_of_training_examples = 1
            a = []
            b = []
            for training_example_index in range(number_of_training_examples):
                random_index_a = training_example_index + data_augmentation
                random_index_b = training_example_index + data_augmentation - np.random.randint(-data_augmentation, data_augmentation+1)
                a.append(cd[random_index_a:random_index_a+pattern_length].values[np.newaxis, ...])
                b.append(cd[random_index_b:random_index_b+pattern_length].values[np.newaxis, ...])
            a1 = np.concatenate(a)
            b1 = np.concatenate(b)

            labels = list(cd)
            input_values = {}
            output_values = {}
            middle_pattern_index = pattern_length - n_to_predict

            for idx, a_label in enumerate(labels):
                df_a = pd.DataFrame(a1[..., idx])
                df_b = pd.DataFrame(b1[..., idx])
                input_values[a_label] = df_a.iloc[:, :middle_pattern_index]
                output_values[a_label] = df_b.iloc[:, -middle_pattern_index:]
                input_values[a_label].Name = a_label
                output_values[a_label].Name = a_label
            return input_values, output_values

        def generate_timeseries(cd, pattern_length, n_to_predict):
            number_of_training_examples = len(cd) - pattern_length + 1
            if number_of_training_examples < 1:
                number_of_training_examples = 1
            data = cd.values[:number_of_training_examples, :]
            targets = [cd.values[idx+n_to_predict:idx+pattern_length, :] for idx in range(number_of_training_examples)]
            length = pattern_length - n_to_predict
            tg = TimeseriesGenerator(data, targets, length, batch_size=25)

            tg_len = len(tg)
            range_tg_len = range(tg_len)
            x = np.concatenate([(tg[idx])[0] for idx in range_tg_len])
            y = np.concatenate([(tg[idx])[1] for idx in range_tg_len])
            
            labels = list(cd)
            input_values = {}
            output_values = {}
            for idx, a_label in enumerate(labels):
                input_values[a_label] = pd.DataFrame(x[:,:,idx])
                output_values[a_label] = pd.DataFrame(y[:,:,idx])
                input_values[a_label].Name = a_label
                output_values[a_label].Name = a_label
            return input_values, output_values

        if data_augmentation:
            input_values, output_values = _generate_samples(cd, pattern_length, n_to_predict, data_augmentation)
        else:
            input_values, output_values = generate_timeseries(cd, pattern_length, n_to_predict)
        
        _x, labels_x = make_predictor( input_values,  input_features_labels)
        _y, labels_y = make_predictor(output_values, output_features_labels)

        return _x, _y, {'x': labels_x, 'y': labels_y}
    _x, _y, labels = _make_regression(
        input_df,
        pattern_length,
        input_features_labels,
        output_features_labels,
        n_to_predict,
        data_augmentation
    )

    if except_last_n == 0:
        X_train = _x
        y_train = _y
        X_test = []
        y_test = []
    else:    
        X_train, X_test, y_train, y_test = train_test_split(_x, _y, test_size=except_last_n, shuffle=False)

    tsd = TimeSeriesDataset(X_train, y_train, labels)

    tsd.X_test = X_test
    tsd.y_test = y_test

    return tsd