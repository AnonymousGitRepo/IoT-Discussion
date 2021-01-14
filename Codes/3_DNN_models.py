import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer,PorterStemmer
from tensorflow.keras.optimizers import Adam
from sklearn.manifold import TSNE
from nltk.corpus import stopwords 
import re
from numpy import array
import numpy as np
from sklearn.metrics import classification_report as cr 
from sklearn.metrics import f1_score,  recall_score, roc_auc_score, precision_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from keras import Input,Model,callbacks
from keras.layers import Dense,Dropout,Embedding,LSTM,Flatten,Dot,ReLU,LeakyReLU,LayerNormalization,GlobalAveragePooling1D,GlobalMaxPooling1D,Bidirectional,Concatenate,Reshape
import math 
import tensorflow as tf
import scipy.sparse as sp
from numpy.random import seed
from tensorflow.keras.preprocessing.sequence import pad_sequences,skipgrams
from tensorflow.keras.preprocessing.text import Tokenizer 

import pickle
seed(1)
tf.random.set_seed(2)

dataset  = pickle.load(open("kfold_cross_validation_dataset_security_aspect.p","rb"))

def DNN(x_train,x_test,y_train,y_test):
  input=Input(shape=len(x_train[0]))
  x=Dense(128,activation='relu')(input)
  output=Dense(1,activation='sigmoid')(x)
  model=Model([input],output)
  return model
def Model_Compilation(model):
  return model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])  
def Model_Training(model,x_train,x_test,y_train,y_test):
  cb = []
  reduce_lr_loss = callbacks.ReduceLROnPlateau(monitor='loss', factor=.2, patience=2, verbose=0, min_delta=1e-6, mode='min')
  cb.append(reduce_lr_loss)
  early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',restore_best_weights=True)
  cb.append(early_stop)
  history=model.fit(x_train,y_train,
                  batch_size=128,
                  epochs=20,validation_data=(x_test,y_test),
                  shuffle=True,
                  callbacks=cb,
                  verbose=0)
def Confusion_Matrix(model,x_train,x_test,y_train,y_test):
  y_pred=np.rint(model.predict(x_test))
  

  pre = precision_score(y_test, y_pred, average = None)
  re = recall_score(y_test, y_pred, average = None)
  f1_score_val=2*pre*re/(pre+re)
  return pre,re,f1_score_val


f1=[]
pre=[]
re=[]
for k in range(10):
  x_train,x_test,y_train,y_test=dataset2['X_Train'][k],dataset2['X_Test'][k],dataset2['Y_Train'][k],dataset2['Y_Test'][k]
  x_train,x_test,y_train,y_test=Train_Test_Data_vectorization(x_train,x_test,y_train,y_test)
  model=DNN(x_train,x_test,y_train,y_test)
  Model_Compilation(model)

  Model_Training(model,x_train,x_test,y_train,y_test)
  pre_val,re_val,f1_val=Confusion_Matrix(model,x_train,x_test,y_train,y_test)
  
  f1.append(f1_val)
  pre.append(pre_val)
  re.append(re_val)
  
print(sum(pre)[1]/10,sum(re)[1]/10,sum(f1)[1]/10,sum(auc)/10)
