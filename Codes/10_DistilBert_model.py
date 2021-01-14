
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


seed(1)
tf.random.set_seed(2)

dataset  = pickle.load(open("kfold_cross_validation_dataset_security_aspect.p","rb"))


"""#Roberta Net"""

from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, DistilBertConfig

class BertModel:
    def __init__(self,label):
        self.num_class = label
        self.model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels = self.num_class )
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.dataset = dataset



    def re_initialize(self):
        self.model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels = self.num_class)


    
    
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
        
        f1=[]
        recall = []
        precision = []
        auc = []
        for i in range(10):
            train = self.tokenize(self.dataset['X_Train'][i])
            test = self.tokenize(self.dataset['X_Test'][i])
            train_y = self.dataset['Y_Train'][i].to_numpy()
            test_y = self.dataset['Y_Test'][i].to_numpy()

            self.re_initialize()
            self.model_compilation()

            history = self.model.fit(train, train_y, batch_size = 16, epochs = 3, validation_data = (test,test_y), callbacks = callbacks.ReduceLROnPlateau(monitor='loss', factor=.2, patience=3, verbose=0, min_delta=1e-6, mode='min'))
            
            output = self.model.predict(test)

            final_output = np.argmax(output['logits'], axis = -1)
            

            pre = precision_score(test_y, final_output, average = None)
            re = recall_score(test_y, final_output, average = None)            
            f1_score_val=f1_score(test_y, final_output, average = None)
            
            f1.append(f1_score_val)
            precision.append(pre)
            recall.append(re)

            print(f1_score_val)

        #print(f1)

        return f1,precision,recall


class_count = 2
bert = BertModel(class_count)
f1,precision, recall = bert.run_model()
print(sum(precision[:][1])/10,sum(recall[:][1])/10,sum(f1[:][1])/10)
