import calendar
import math

import pandas as pd
import seaborn as sns
from time_series_dataset import TimeSeriesDataset
from time_series_dataset_generator import (make_predictor,
                                           make_time_series_dataset)


class FlightSeriesDataset(TimeSeriesDataset):
    def __init__(self, pattern_length, n_to_predict, except_last_n, data_augmentation=None):
        flights = sns.load_dataset("flights")
        input_features_labels = ['month', 'year']
        output_features_labels = ['passengers']

        month = flights['month']
        months_3l = [month_name[0:3]for month_name in list(calendar.month_name)]
        month_number = [months_3l.index(_month)for _month in month]
        flights['month'] = month_number

        tsd = make_time_series_dataset(
            flights,
            pattern_length,
            n_to_predict,
            input_features_labels,
            output_features_labels,
            except_last_n,
            data_augmentation
        )
        self.wrap(tsd)
        self.month_number_df = month_number
        self.year_df = flights['year']

    def wrap(self, tsd):
        self.__dict__ = tsd.__dict__

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

        month_number_df = self.month_number_df
        year_df = self.year_df
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
        return make_predictor(input_features, 1)
