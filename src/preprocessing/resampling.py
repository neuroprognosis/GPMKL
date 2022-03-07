import random
from random import randrange
import pandas as pd
from sklearn.model_selection import KFold


# code to split train and test dataset
def train_test_split(dataset, split=0.75):
     train = dataset.sample(frac=split, random_state=200)
     test = dataset.drop(train.index)
     return train, test
 
# split a dataset into K folds
def cross_validation_split(dataset_index, folds=10):
     train = []
     target = []
     kf = KFold(n_splits=folds, shuffle=False)
     for train_index, test_index in kf.split(dataset_index):
         train.append(train_index)
         target.append(test_index)
     return train, target

