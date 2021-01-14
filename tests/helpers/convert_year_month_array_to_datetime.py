from datetime import datetime

def convert_year_month_array_to_datetime(year_month_array):
    def convert_singe_year_month_array_to_datetime(single_year_month_array):
        return datetime(year=single_year_month_array[1], month=single_year_month_array[0], day=15)
    year_month_array_size = year_month_array.size
    if year_month_array_size == 2:
        return convert_singe_year_month_array_to_datetime(year_month_array)
    return [convert_singe_year_month_array_to_datetime(year_month_array[idx, ...]) for idx in range(len(year_month_array))]