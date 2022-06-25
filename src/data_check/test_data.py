import numpy as np
import pandas as pd
import scipy.stats


def test_column_names(data):
    """
    Tested set should have the same columns as reference
    """
    expected_columns = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    actual_columns = data.columns.values

    assert set(expected_columns) == set(actual_columns)


def test_neighborhood_names(data):
    """
    Tested set must have the same unique values in `neighbourhood_group` column
    """
    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_similar_neighbourhood_distribution(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different from that of the reference dataset
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


def test_row_count(data):
    """
    Tested data size (row number) should be withing set limits
    """
    lower_limit = 15000
    upper_limit = 1000000
    data_row_count = data.shape[0]
    assert lower_limit < data_row_count < upper_limit


def test_price_range(data: pd.DataFrame, min_price, max_price):
    """
    Accepted price (`price` column), should be between set limits
    """
    assert data['price'].between(min_price, max_price).all()
