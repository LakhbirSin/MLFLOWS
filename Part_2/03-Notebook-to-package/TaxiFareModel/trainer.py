import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import set_config
set_config(display='diagram')

from encoders import DistanceTransformer, TimeFeaturesEncoder

class Trainer:
    def __init__(self):
        self.dist_pipe = Pipeline([
            ('distance', DistanceTransformer()),
            ('scaler', StandardScaler())
        ])

        self.time_pipe = Pipeline([
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