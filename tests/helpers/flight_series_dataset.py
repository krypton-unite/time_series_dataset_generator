import calendar
import seaborn as sns
from time_series_dataset import TimeSeriesDataset
from time_series_dataset_generator import make_time_series_dataset


class FlightSeriesDataset(TimeSeriesDataset):
    def __init__(self, pattern_length, n_to_predict, except_last_n, augmentation=0, stride='auto', generate_test_dataset = False):
        flights = sns.load_dataset("flights")
        input_features_labels = ['month', 'year']
        output_features_labels = ['passengers']

        month = flights['month']
        months_3l = [month_name[0:3]for month_name in list(calendar.month_name)]
        month_number = [months_3l.index(_month)for _month in month]
        flights['month'] = month_number

        past_pattern_length = pattern_length - n_to_predict
        tsd = make_time_series_dataset(
            flights,
            pattern_length,
            n_to_predict,
            input_features_labels,
            output_features_labels,
            except_last_n,
            augmentation = augmentation,
            stride = stride,
            overlap = past_pattern_length - n_to_predict,
            generate_test_dataset = generate_test_dataset
        )
        self.wrap(tsd)

    def wrap(self, tsd):
        self.__dict__ = tsd.__dict__
    
    # pylint: disable=arguments-differ
    def make_future_dataframe(self, number_of_months, include_history=True):
        pass