import numpy as np
import pandas as pd
import itertools
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from collections import Counter

class BagOShapesClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.dummy = "dummy"

    #def check_X(self, X):
        #Is this in the correct format?
        #The inputs need to have been individually changed so that the mean is 0 and the std dev is 1

    def GenerateSparseDF(self):
        possibilities = []
        for combination in itertools.product(list(self.SAXparameters.keys()), repeat=self.SAXstringlength):
            possibilities.append(''.join(map(str, combination)))
        sparseDF = pd.DataFrame(index=possibilities)
        return sparseDF

    def chunkify(self, lst, n):
        return [ lst[i::n] for i in range(n) ]

    def SAXconvert(self, lst):
        saxString = ''
        #This is custom SAX threshold logic. I need to figure out a way to make this configurable somehow.
        for i in lst:
            for char in self.SAXparameters:
                tempFlag = True
                if self.SAXparameters[char]:
                    if i > self.SAXparameters[char]:
                        saxString += char
                        tempFlag = False
                        break
            if tempFlag:
                saxString += char
        return saxString

    def GetSAXCounts(self, data):
        newLength = (len(data)-(len(data)%self.SAXstringlength))
        data = data.loc[0:newLength]
        numChunks = int(len(data)/self.SAXstringlength)
        chunks = self.chunkify(data['Scaled'], numChunks)
        SAXStringCounts = pd.DataFrame.from_dict(Counter([ self.SAXconvert(chunk) for chunk in chunks ]), orient='index')
        return SAXStringCounts

    def coefCalculation(self):
        zeroDistances = euclidean_distances(self.sparseSAXMatrix.transpose(), self.sparseSAXMatrix.transpose())
        zeroDistances = np.concatenate(zeroDistances).tolist()
        self.distances = list(filter(lambda a: a != 0, zeroDistances))
        return np.mean(self.distances), np.std(self.distances)

    def fit(self, X_list, SAXparameters, SAXstringlength, std_devSensitivity):
        #X = self.check_X(X)
        self.SAXparameters = SAXparameters
        self.SAXstringlength = SAXstringlength
        self.sensitivity = std_devSensitivity
        self.Xlist_ = X_list
        self.sparseSAXMatrix = self.GenerateSparseDF()
        iterator = 0
        for timeSeries in self.Xlist_:
            SAXdf = self.GetSAXCounts(timeSeries)
            SAXdf.rename(index=str, columns={0:str(iterator)}, inplace=True)
            self.sparseSAXMatrix = self.sparseSAXMatrix.merge(SAXdf, how='left', left_index=True, right_index=True)
            iterator += 1

        self.sparseSAXMatrix.fillna(0, inplace=True) 
        self.avg_codistance, self.std_dev = self.coefCalculation()
        return self

    def predict(self, X):
        #I'm pretty sure that I have everything set up, now it's just deliver the prediction with the new input data!
        # Check is fit had been called
        #check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        #X = check_array(X)
        classification = True
        SAXdf = self.GetSAXCounts(X)
        SAXdf.rename(index=str, columns={0:'input'}, inplace=True)
        SAXtestdf = self.sparseSAXMatrix.merge(SAXdf, how='left', left_index=True, right_index=True)
        SAXtestdf.fillna(0, inplace=True)
        distances = euclidean_distances(SAXtestdf.transpose(), SAXtestdf.transpose())[-1].tolist()
        distances = list(filter(lambda a: a != 0, distances))
        avg = np.mean(distances)
        std_devs = np.abs(avg - self.avg_codistance)/self.std_dev
        if std_devs > self.sensitivity:
            classification = False

        #TODO: calculate confidence and use that for sensitivity instead of #of std devs
        
        return {
                'classification': classification,
                'confidence': np.nan
                }