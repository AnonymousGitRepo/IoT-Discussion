import pandas as pd
import timeit
import re
import keras
import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from numpy.random import seed
import pickle

seed(1)
tf.random.set_seed(2)

benchmark_dataset=pd.read_excel('BenchmarkUddinSO-ConsoliatedAspectSentiment.xlsx')
benchmark_dataset=benchmark_dataset.drop_duplicates(keep='first')
benchmark_data = benchmark_dataset['sent']
benchmark_label = benchmark_dataset['codes'] 
aspect = "Security"
def Labeling(row):
    if aspect in row:
        return 1
    return 0

label=benchmark_label.apply(Labeling)

skf=StratifiedKFold(n_splits=10,shuffle=True)
temp=skf.split(benchmark_data,label)
dataset={"X_Train":[],"Y_Train":[],"X_Test":[],"Y_Test":[]}
for train,test in temp:
    
    dataset["X_Train"].append(benchmark_data[train])
    dataset["Y_Train"].append(label[train])
    dataset["X_Test"].append(benchmark_data[test])
    dataset["Y_Test"].append(label[test])

pickle.dump(dataset, open("kfold_cross_validation_dataset_security_aspect.p","wb"))
