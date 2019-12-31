from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn import pipeline
import os
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
import numpy as np
import pandas as pd
import sklearn.preprocessing as sp
from sklearn.utils import shuffle
import sklearn.metrics as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingRegressor
import neurolab as nl
import warnings
warnings.filterwarnings('ignore')

class MachineLearn(object):
    def __init__(self,city,edu,year):
        self.city=city
        self.edu=edu
        self.year=year
