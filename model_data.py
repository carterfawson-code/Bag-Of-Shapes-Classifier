import datetime
import pandas as pd
import numpy as np
import re
import itertools
import os
import io
import json
import boto3
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter

dataFiles = {
    'Good': [
        "C:\DataScience\predictive-maintenance\DataExtracts\PumpData\GoodNormalized0.csv",
        "C:\DataScience\predictive-maintenance\DataExtracts\PumpData\GoodNormalized1.csv",
        "C:\DataScience\predictive-maintenance\DataExtracts\PumpData\GoodNormalized2.csv",
        "C:\DataScience\predictive-maintenance\DataExtracts\PumpData\GoodNormalized3.csv",
        "C:\DataScience\predictive-maintenance\DataExtracts\PumpData\GoodNormalized4.csv"
    ],
    'Bad': [
        "C:\DataScience\predictive-maintenance\DataExtracts\PumpData\BadNormalized0.csv",
        "C:\DataScience\predictive-maintenance\DataExtracts\PumpData\BadNormalized1.csv",
        "C:\DataScience\predictive-maintenance\DataExtracts\PumpData\BadNormalized2.csv"
    ]
}

observationStringLength = 5
SAXcharacters = {
                    "a": 4.5, 
                    "b": 2.5,
                    "c": 1.5,
                    "d": .5,
                    "e": 0,
                    "f": -.75,
                    "g": -1.75,
                    "h": False
                }

def GenerateSparseDF(SAXcharacters, observationStringLength):
    possibilities = []
    for combination in itertools.product(SAXcharacters, repeat=observationStringLength):
        possibilities.append(''.join(map(str, combination)))
    sparseDF = pd.DataFrame(index=possibilities)
    return sparseDF

def chunkify(lst,n):
    return [ lst[i::n] for i in range(n) ]

def SAXconvert(lst, SAXcharacters):
    saxString = ''
    #This is custom SAX threshold logic. I need to figure out a way to make this configurable somehow.
    for i in lst:
        for char in SAXcharacters:
            tempFlag = True
            if SAXcharacters[char]:
                if i > SAXcharacters[char]:
                    saxString += char
                    tempFlag = False
                    break
        if tempFlag:
            saxString += char
    return saxString

def GetSAXCounts(data, observationStringLength, SAXcharacters):
    newLength = (len(data)-(len(data)%observationStringLength))
    data = data.loc[0:newLength]
    numChunks = int(len(data)/observationStringLength)
    chunks = chunkify(data['Scaled'], numChunks)
    SAXStringCounts = pd.DataFrame.from_dict(Counter([ SAXconvert(chunk, SAXcharacters) for chunk in chunks ]), orient='index')
    return SAXStringCounts

sparseSAXMatrix = GenerateSparseDF(SAXcharacters, observationStringLength)

for classifier in dataFiles:
    iterator = 0
    for file in dataFiles[classifier]:
        data = pd.read_csv(file)
        SAXdf = GetSAXCounts(data, observationStringLength, SAXcharacters)
        SAXdf.rename(index=str, columns={0:classifier + str(iterator)}, inplace=True)
        sparseSAXMatrix = sparseSAXMatrix.merge(SAXdf, how='left', left_index=True, right_index=True)
        iterator += 1

sparseSAXMatrix.fillna(0, inplace=True)

distances = euclidean_distances(sparseSAXMatrix.transpose(), sparseSAXMatrix.transpose())