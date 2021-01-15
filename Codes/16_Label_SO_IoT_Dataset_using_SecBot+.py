import pandas as pd
import timeit

import pickle
from tensorflow.keras.optimizers import Adam
import re
import keras
import numpy as np
from sklearn.metrics import classification_report as cr 
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from keras import Input,Model,callbacks
import tensorflow as tf
from numpy.random import seed

#from transformers import AutoTokenizer,AutoModelForSequenceClassification

from transformers import BertTokenizer, TFBertModel, BertConfig, TFBertForSequenceClassification

seed(1)
tf.random.set_seed(2)


unlabeled_dataset=pd.read_csv('SO_IoT_sentences.csv')
model_weight = pickle.load(open('SecBot+_weights.p','rb'))

from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

num_class = 2
model = TFRobertaForSequenceClassification.from_pretrained('roberta-base' , num_labels = num_class)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

#pretained weight
model.set_weights(model_weight)


def new_tokenize(dataset):
    input_ids = []
    attention_masks = []

    for sent in dataset:
        bert_inp = tokenizer .encode_plus(sent.lower(), add_special_tokens = True, max_length = 100, truncation = True, padding = 'max_length', return_attention_mask = True)
        input_ids.append(bert_inp['input_ids'])
        attention_masks.append(bert_inp['attention_mask'])

    train_input_ids = np.asarray(input_ids)
    train_attention_masks = np.array(attention_masks)

    return [train_input_ids,train_attention_masks]
    
    
import pandas

dataset = unlabeled_dataset['sentence']



new_data = []
for i in range(1+(len(dataset))//10000):
    sentences = dataset[10000*i:min(10000*(i+1),len(dataset))]
    tkn = new_tokenize(sentences)
    pred_val = model.predict(tkn)
    final_output = np.argmax(pred_val['logits'], axis = -1)
    new_data.extend(final_output)
    
    
    print(i)

df = unlabeled_dataset
df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])

df['SecBot+'] = new_data

df.insert(0, 'SentenceId', range(1,  len(df)+1))

df.to_csv('SecBot+_SO_IoT_Label.csv')
