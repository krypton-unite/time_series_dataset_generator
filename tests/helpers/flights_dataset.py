import math
import numpy as np
import pandas as pd

from .flight_series_dataset import FlightSeriesDataset

def _make_predictor(features, number_of_training_examples):
    def _raw_make_predictor(features, *reshape_args):
        #pylint: disable=too-many-function-args
        return np.concatenate(features, axis=-1).reshape(
            list(reshape_args) + [len(features)]).astype(np.float32)
    #pylint: disable=too-many-function-args
    return _raw_make_predictor(features, number_of_training_examples, -1)

class FlightsDataset(FlightSeriesDataset):
    def __init__(self, pattern_length=144, n_to_predict=0, except_last_n=0):
        super().__init__(pattern_length, n_to_predict, except_last_n, augmentation=0, stride=1)

    # pylint: disable=arguments-differ
    def make_future_dataframe(self, number_of_months, include_history=True):
        """
        make_future_dataframe

        :param number_of_months: number of months to predict ahead
        :param include_history: optional, selects if training history is to be included or not
        :returns: future dataframe with the selected amount of months
        """
        def create_dataframe(name, data):
            return pd.DataFrame(data={name: data})

        def create_month_dataframe(data):
            return create_dataframe('month_number', data)

        def create_year_dataframe(data):
            return create_dataframe('year', data)

        month_number_df = create_month_dataframe(self.get_feature('x', 'month'))
        year_df = create_year_dataframe(self.get_feature('x', 'year'))
        last_month = month_number_df.values[-1][0]
        last_year = year_df.values[-1][0]
        if not include_history:
            month_number_df = create_month_dataframe([])
            year_df = create_year_dataframe([])
        for i in range(number_of_months):
            month_index = last_month+i
            new_months = [math.fmod(month_index, 12)+1]
            new_years = [last_year + math.floor(month_index / 12)]
            month_number_df = month_number_df.append(
                create_month_dataframe(new_months), ignore_index=True)
            year_df = year_df.append(
                create_year_dataframe(new_years), ignore_index=True)
        input_features = [month_number_df, year_df]
        return _make_predictor(input_features, 1)
