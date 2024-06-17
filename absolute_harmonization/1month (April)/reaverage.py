import pandas as pd
import numpy as np

""" data = pd.read_csv('files/Astana_PM2.5_2024_04.csv')
data['hour_index'] = None
hour_indices = data['hour_index'].values """

def group_by_hour_range(data: np.array, current_row: int = 0 , current_range: int = 1, range_length: int = 2, counter : int = 1) -> int:

    """ insert a new column of hour indices to a dataframe """

    try:
        assert 24%range_length == 0, 'range length should be divisor of 24'

        data[current_row] = current_range
        current_row  += 1
        
        if counter == range_length:
            current_range += 1
            counter = 1
            group_by_hour_range(data, current_row, current_range, range_length, counter)
        else:
            counter += 1
            group_by_hour_range(data, current_row, current_range, range_length, counter)
        
    except IndexError:
        return 

 
""" group_by_hour_range(hour_indices, range_length=3)
print(data[:20]) """