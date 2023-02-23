import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

import utils

class Trainer :
    """
        Computes the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'.
    """

    def __init__(self,
                data_link = "train.csv"):
        self.data_link = data_link
        self.pipe_dist = Pipeline([
            ('distance', DistanceTransformer()),
            ('scaler', StandardScaler())
        ])

        self.pipe_time = Pipeline([
            ('time_features_create', TimeFeaturesEncoder()),
            ('time_features_ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preproc_pipe = FeatureUnion([
            ('pipe_dist', self.dist_pipe),
            ('pipe_time', self.time_pipe)
        ])

        self.model = LinearRegression()
        self.pipe = Pipeline([
            ('preprocessor', self.preproc_pipe),
            ('regressor', self.model)
        ])


def clean_data(df, test=False):
    '''returns a DataFrame without outliers and missing values'''
    # A COMPLETER
    df = df.drop(["key"], axis=1)
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
        df = df[df.passenger_count < 8]
        df = df[df.passenger_count >= 0]
        df = df[df["pickup_latitude"].between(left=40, right=42)]
        df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
        df = df[df["dropoff_latitude"].between(left=40, right=42)]
        df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df

    def train(X_train, y_train, pipeline):
    '''returns a trained pipelined model'''
   # A COMPLETER
    pipeline.fit(X_train, y_train)
    return pipeline

    def evaluate(X_test, y_test, pipeline):
    '''returns the value of the RMSE'''
    # A COMPLETER
    y_pred = pipeline.predict(X_test)
    rmse = compute_rmse(y_pred, y_test)

    print(rmse)
    return rmse 

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
    

# create a TimeFeaturesEncoder
class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extracts the day of week (dow), the hour, the month and the year from a time column.
        Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'.
    """

    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name
       # A COMPLETER

    def extract_time_features(self, X):
        # A COMPLETER
        timezone_name = self.time_zone_name
        time_column = self.time_column
        df = X.copy()
        df.index = pd.to_datetime(df[time_column])
        df.index = df.index.tz_convert(timezone_name)
        df["dow"] = df.index.weekday
        df["hour"] = df.index.hour
        df["month"] = df.index.month
        df["year"] = df.index.year        
        return df
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.extract_time_features(X)[['dow', 'hour', 'month', 'year']].reset_index(drop=True)

from sklearn.base import BaseEstimator, TransformerMixin

class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
        Computes the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'.
    """

    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        # A COMPPLETER
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon
    
    def fit(self, X, y=None):
        return self
        # A COMPLETER 

    def transform(self, X, y=None):
    
        return pd.DataFrame(haversine_vectorized(X)).rename(columns={0: "course distance [km]"}).copy()