import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from numpy import random
import re
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.model_selection import StratifiedKFold 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from keras.layers import LSTM,Conv1D,Dense,Input,Dropout,Bidirectional,Concatenate,MaxPool1D,Flatten,GRU,Attention
from keras import Model,callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score,precision_score,recall_score, roc_auc_score, accuracy_score, matthews_corrcoef
from pandas import DataFrame
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

opiner = pd.read_excel('Opiner_Samples.xlsx')
combined = pd.read_excel('Combined_Training_Samples.xlsx')
validation = pd.read_excel('Validation_Samples.xlsx')
embed_matrix=pd.read_csv('../input/glove6b/glove.6B.200d.txt',sep=" ", header=None,quoting=3)

vocab={}
def feature_extraction(val):
  vocab[val[0]]=np.array(val[1:])
  return val

embed_matrix.apply(feature_extraction,axis=1)




stopword=stopwords.words("english")
stopword.append("i'm")
stopword.append('could')
lemmatizer=WordNetLemmatizer()

def SentenceCleaner(sent):
#     print(sent)
    txt=""
    for word in re.split('[,()<>|}{~\]\[ ]',sent.lower()): #?
      
      tempCheck=word

      urls=r"(?i)((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
      word=re.sub(urls,"url",word)
      word=re.sub("url_url","url",word)
      word=re.sub(".*[=].*","",word)  #remove word with =

      #word=re.sub("i/o","io",word)
      word=re.sub("[:]"," ",word)
      #word=re.sub("[.][.]+"," ",word)
      #word=re.sub("[^ '\"]+[.][^ .'\"]+","",word)
      word=re.sub("[-/]"," ",word)
      #word=re.sub("[@][^ ]*","",word)  #remove word with @
      
      #word=re.sub(r'[uU][rR][lL]',"",word)  #remove word url
      word=re.sub(r'[0-9@"]+',"",word)
      #word=re.sub(r'[@_!#$%^&*()<>?\\|}{~:.;"+]+',"",word)

      word_list=re.split('[ ]+',word)
      
      for w in word_list:
        
        if len(w)>1:
          txt+=(w+" ")
          
    if txt=='':
      txt='nan'
    
    return txt

X_txt=opiner['Sentence'].apply(SentenceCleaner) # combined['Sentence'].apply(SentenceCleaner)
X_test = validation['sentence'].apply(SentenceCleaner)
#Input Feature
unfounded=set()
arr=[]
empty_array=np.zeros((200))
max_len=100
for d in (X_txt):
  temp=[]
  for word in text_to_word_sequence(d):
    if word in vocab:
      temp.append(vocab[word])
    else:
      unfounded.add(word)
    if len(temp)==max_len:
#       print(word)
      break
      
  while(len(temp)!=max_len):
    temp.append(empty_array)
  
  arr.append(np.array(temp))
X=np.array(arr)
X = np.asarray(X).astype('float32')
Y = opiner['IsAboutSecurity'] # combined['IsAboutSecurity']
Y_test = validation['IsAboutSecurity']
arr = []
for d in (X_test):
  temp=[]
  for word in text_to_word_sequence(d):
    if word in vocab:
      temp.append(vocab[word])
    else:
      unfounded.add(word)
    if len(temp)==max_len:
#       print(word)
      break
      
  while(len(temp)!=max_len):
    temp.append(empty_array)
  
  arr.append(np.array(temp))
X_test = np.asarray(X).astype('float32')
skf=StratifiedKFold(n_splits=10,shuffle=True,random_state = 42)
temp=skf.split(X,Y)
dictionary_k_folds={"Train":[],"Test":[]}
for train,test in temp:
  dictionary_k_folds["Train"].append(train)
  dictionary_k_folds["Test"].append(test)

def LSTMModel():

    input=Input((100,50))

    x=LSTM(64,input_shape=(100,50),return_sequences=True)(input)
    x= Dense(16)(x)
    x=Flatten()(x)
    x=Dense(32,activation='relu')(x)
    output=Dense(1,activation='sigmoid')(x)

    model=Model([input],output) 
    #model.summary()
    model.compile(optimizer=Adam(learning_rate=2e-3),loss='binary_crossentropy',metrics=['accuracy'])
    return model
def Bi_LSTMModel():

    input=Input((100,50))

    x=LSTM(64,input_shape=(100,50),return_sequences=True)
    y=LSTM(64,input_shape=(100,50),return_sequences=True, go_backwards=True)
    x = Bidirectional(x, backward_layer=y)(input)
    
    x= Dense(64)(x)
    x=Flatten()(x)
    x=Dense(32,activation='relu')(x)
    output=Dense(1,activation='sigmoid')(x)

    model=Model([input],output) 
    #model.summary()
    model.compile(optimizer=Adam(learning_rate=2e-3),loss='binary_crossentropy',metrics=['accuracy'])
    return model
pre = []
rec = []
f1 = []
mcc = []
auc = []
def training():
    x_train=X[dictionary_k_folds["Train"][current_k]]
    x_test=X[dictionary_k_folds["Test"][current_k]]
    y_train=Y[dictionary_k_folds["Train"][current_k]]
    y_test=Y[dictionary_k_folds["Test"][current_k]]

    model = Bi_LSTMModel() # LSTMModel()
    cb = []
    reduce_lr_loss = callbacks.ReduceLROnPlateau(monitor='loss', factor=.2, patience=3, min_delta=1e-6, mode='min')
    cb.append(reduce_lr_loss)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto',restore_best_weights=True)
    cb.append(early_stop)
    history=model.fit(x_train,y_train,
                  batch_size=64,
                  epochs=50,
                  shuffle=True,validation_data=(x_test,y_test),
                  callbacks=cb,
                  verbose=0
                  )


    y_pred=np.rint(model.predict(x_test))
    pre_v,re_v,f1_score_val, ac, mc=Confusion_Matrix(model,x_train,x_test,y_train,y_test)
    pre.append(pre_v[1])
    rec.append(re_v[1])
    f1.append(f1_score_val[1])
    mcc.append(mc)
    auc.append(ac)
def performance_validation():
    x_train=X
    x_test=X_tset
    y_train=Y
    y_test=Y_test
    model = Bi_LSTMModel() # LSTMModel()
    cb = []
    reduce_lr_loss = callbacks.ReduceLROnPlateau(monitor='loss', factor=.2, patience=3, min_delta=1e-6, mode='min')
    cb.append(reduce_lr_loss)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto',restore_best_weights=True)
    cb.append(early_stop)
    history=model.fit(x_train,y_train,
                  batch_size=64,
                  epochs=50,
                  shuffle=True,
                  callbacks=cb,
                  verbose=0
                  )


    y_pred=np.rint(model.predict(x_test))
    pre_v,re_v,f1_score_val, ac, mc=Confusion_Matrix(model,x_train,x_test,y_train,y_test)
    print(pre_v[1], re_v[1], f1_score_val[1], mc, ac)


for i in range(10):
    current_k = i
    training()

print(sum(pre)/10,sum(rec)/10,sum(f1)/10, sum(auc)/10, sum(mcc)/10)

print('Validation Performance')
performance_validation()
