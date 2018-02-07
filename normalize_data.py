import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
import io
import json
import boto3
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, Imputer

class RemoveNA(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.created = True
    def fit(self, X, y=None):
        return self #Don't know what this is for right now
    def transform(self, X, y=None):
        X = X.dropna(axis=0, how='any')
        return X

class StandardScaleDF(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
    def fit(self, X, y=None):
        return self #Don't know what this is for right now
    def transform(self, X, y=None):
        X.loc[:,'Scaled'] = self.scaler.fit_transform(X['Value'].values.reshape(-1, 1))
        return X

class RemoveExtraHeaders(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pattern = '^Value'
    def fit(self, X, y=None):
        return self #Don't know what this is for right now
    def transform(self, X, y=None):
        for tag in X.columns:
            X.loc[:,'Value'] = X['Value'].apply(lambda x: self.convertState(x))
        return X
    def convertState(self, value):
        if not pd.isnull(value):
            result = re.match(self.pattern, str(value))
            if result:
                return np.nan
        return value

class RemoveSystemStates(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pattern = '^State: (\d+).+'
    def fit(self, X, y=None):
        return self #Don't know what this is for right now
    def transform(self, X, y=None):
        for tag in X.columns:
            X.loc[:, 'Value'] = X['Value'].apply(lambda x: self.convertState(x))
        return X
    def convertState(self, value):
        if not pd.isnull(value):
            result = re.match(self.pattern, str(value))
            if result:
                return np.nan
        return value

class SampleRawTimeSeries(BaseEstimator, TransformerMixin):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
    def fit(self, X, y=None):
        return self #Don't know what this is for right now
    def transform(self, X, y=None):
        X.loc[:, 'TimeStamp'] = pd.to_datetime(X['TimeStamp'])
        X.loc[:, 'Value'] = pd.to_numeric(X['Value'])
        resampled_data = X.resample(self.sample_rate, on='TimeStamp').mean()
        resampled_data.interpolate(method='linear', inplace=True)
        return resampled_data

class NewIndexDF(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.created = True
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.reset_index()  
    
class ChangeColtoIndexDF(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.set_index(self.column)


sample_pipeline = Pipeline([
    ('removeStates', RemoveSystemStates()),
    ('removeHeaders', RemoveExtraHeaders()),
    ('removeNA', RemoveNA()),
    ('sample', SampleRawTimeSeries(sample_rate='20T')),
    ('newIndex', NewIndexDF()),
])

scale_pipeline = Pipeline([
    ('changeIndex', ChangeColtoIndexDF('TimeStamp')),
    ('scaleDF', StandardScaleDF()),
])

def sample_data(filename):
    data = pd.read_csv(os.path.normpath(filename))
    data = sample_pipeline.fit_transform(data)
    return data

file = "C:\DataScience\predictive-maintenance\DataExtracts\PumpData\\18VE3305B2.csv"

sampledData = sample_data(file)

numPeriods = 2050

goodPeriods = [
    "1/13/15 00:00",
    "3/24/15 00:00",
    "7/14/16 23:00",
    "11/2/16 01:00",
    "4/19/17 00:00"
]

badPeriods = [
    "10/13/15 00:00",
    "11/16/15 00:00",
    "5/6/16 00:00"
]

def GenerateDataSet(periodStart, numPeriods, sampledData):
    indexNum = sampledData[sampledData['TimeStamp'] == periodStart].index[0]
    return scale_pipeline.fit_transform(sampledData.loc[indexNum:indexNum + numPeriods - 1])

goodData = {}
goodCounter = 0
badData = {}
badCounter = 0

for period in goodPeriods:
    goodData[period] = GenerateDataSet(period, numPeriods, sampledData)
    goodData[period].to_csv("C:\DataScience\predictive-maintenance\DataExtracts\PumpData\GoodNormalized" + str(goodCounter) + ".csv")
    goodCounter += 1
    
for period in badPeriods:
    badData[period] = GenerateDataSet(period, numPeriods, sampledData)
    badData[period].to_csv("C:\DataScience\predictive-maintenance\DataExtracts\PumpData\BadNormalized" + str(badCounter) + ".csv")
    badCounter += 1