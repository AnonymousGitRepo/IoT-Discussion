import pandas as pd
import timeit

from tensorflow.keras.optimizers import Adam
import re
import keras
import numpy as np
import keras.backend as K
from sklearn.metrics import classification_report as cr 
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from keras import Input,Model,callbacks
import tensorflow as tf
from numpy.random import seed
from transformers import BertTokenizer, TFBertModel, BertConfig, TFBertForSequenceClassification
import pickle

seed(1)
tf.random.set_seed(2)

dataset  = pickle.load(open("kfold_cross_validation_dataset_security_aspect.p","rb"))

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

from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

class BertModel:
    def __init__(self,label):
        self.num_class = label
        self.model = TFRobertaForSequenceClassification.from_pretrained('roberta-base' , num_labels = self.num_class)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.dataset = dataset



    def re_initialize(self):
        self.model = TFRobertaForSequenceClassification.from_pretrained('roberta-base' , num_labels = self.num_class)

    
    
    def tokenize(self, dataset):
        input_ids = []
        attention_masks = []

        for sent in dataset:
            bert_inp = self.tokenizer .encode_plus(sent.lower(), add_special_tokens = True, max_length = 100, truncation = True, padding = 'max_length', return_attention_mask = True)
            input_ids.append(bert_inp['input_ids'])
            attention_masks.append(bert_inp['attention_mask'])

        train_input_ids = np.asarray(input_ids)
        train_attention_masks = np.array(attention_masks)

        return [train_input_ids,train_attention_masks]
    
    def model_compilation(self):

        print('\nAlBert Model', self.model.summary())


        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-5, epsilon = 1e-08)

        self.model.compile(loss = loss, optimizer = optimizer, metrics = [metric])

    
    def run_model(self):
        
        def run_full_dataset(self):
        data = self.tokenize(benchmark_data)
        labels = label
        self.re_initialize()
        self.model_compilation()
        history = self.model.fit(data, labels, batch_size = 16, epochs = 3, callbacks = callbacks.ReduceLROnPlateau(monitor='loss', factor=.2, patience=3, verbose=0, min_delta=1e-6, mode='min'))
        return self.model


class_count = 2
bert = BertModel(class_count)
model = bert.run_model()
final_model_weights ={"Roberta Model weight(Roberta-base) full-dataset":model.get_weights()} 
pickle.dump(final_model_weights, open("security_aspect_classifier_model_weights_trained_on_full_dataset.p","wb"))
