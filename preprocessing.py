import numpy as np
import pandas as pd

def preprocess(input, affordable_brands, scalar):
    """
    Prepare a single-row dataframe from raw input dict.
    Returns: pd.DataFrame suitable for model.predict
    """

    feature_dict = {
        'max_power': [0],
        'affordable_brand': [0],
        'engine': [0],
        'transmission_type': [0],
        'mileage': [0],
        'fuel_type_Diesel': [0],
        'fuel_type_Petrol': [0],
        'vehicle_age': [0],
        'seats_5.0': [0],
        'seats_4.0': [0]
    }

    columns_to_transform = ['engine', 'max_power', 'vehicle_age']

    # work on a shallow copy to avoid mutating caller's dict
    data = input.copy()

    # apply log1p to chosen columns (ensure scalar floats)
    for key in columns_to_transform:
        val = float(np.asarray(data[key]))
        data[key] = np.log1p(val)

    # scale numerical features using provided scalar dict
    for key in scalar.keys():
        val = float(np.asarray(data[key]))
        feature_dict[key][0] = (val - scalar[key]['mean']) / scalar[key]['scale']

    # categorical / binary features
    if data.get('brand') in affordable_brands:
        feature_dict['affordable_brand'][0] = 1 

    seats_val = int(data.get('seats', 0))
    if seats_val == 5:
        feature_dict['seats_5.0'][0] = 1
    elif seats_val == 4:
        feature_dict['seats_4.0'][0] = 1

    if data.get('fuel_type', '').lower() == 'petrol':
        feature_dict['fuel_type_Petrol'][0] = 1
    else:
        feature_dict['fuel_type_Diesel'][0] = 1

    if data.get('transmission_type', '').lower() == 'automatic':
        feature_dict['transmission_type'][0] = 1

    input_df = pd.DataFrame.from_dict(feature_dict)
    return input_df