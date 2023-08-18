import pandas as pd

########################################################################
## Function to get offsetted yearmo
########################################################################
def offset_yearmo(yearmo, offset):
    """
    Return offseted yearmo based on offset
    Parameters:
    yearmo (int): snapshot yearmo
    offset (int): offset to go back by no of months
    """
    yearmo_timestamp = pd.to_datetime(yearmo, format="%Y%m")
    required_yearmo_timestamp = yearmo_timestamp - pd.DateOffset(months=offset)
    required_yearmo = int(str(required_yearmo_timestamp.year) + str(required_yearmo_timestamp.month).zfill(2))
    return required_yearmo
