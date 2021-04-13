from transformers import *
import numpy as np
import pandas as pd
import random as rd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report as cr 
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, matthews_corrcoef, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
from keras import Input,Model,callbacks


np.random.seed(10)
rd.seed(10)
tf.random.set_seed(10)
import scipy.sparse as sp

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from keras import Input,Model,callbacks
from keras.layers import Dense,Dropout,Embedding,LSTM,Flatten,Dot,ReLU,LeakyReLU,LayerNormalization,GlobalAveragePooling1D,GlobalMaxPooling1D,Bidirectional,Concatenate,Reshape
opiner = pd.read_excel('Opiner_Samples.xlsx')
combined = pd.read_excel('Combined_Training_Samples.xlsx')
validation = pd.read_excel('Validation_Samples.xlsx')

total = combined
skf=StratifiedKFold(n_splits=10,shuffle=True, random_state = 42)
temp=skf.split(total['Sentence'],total['IsAboutSecurity'])
dataset={"X_Train":[],"Y_Train":[],"X_Test":[],"Y_Test":[]}
for train,test in temp:
    print(train)
    dataset["X_Train"].append(total['Sentence'][train])
    dataset["Y_Train"].append(total['IsAboutSecurity'][train])
    dataset["X_Test"].append(total['Sentence'][test])
    dataset["Y_Test"].append(total['IsAboutSecurity'][test])


class BertModel:
    def __init__(self,label,aspect):
        self.num_class = label
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.model = TFAlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=2)
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         self.model = TFBertForSequenceClassification.from_pretrained('bert-base')
#         self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#         self.model = TFRobertaForSequenceClassification.from_pretrained('roberta-base')
#         self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
#         self.model = TFXLNetForSequenceClassification.from_pretrained('xlnet-base-cased')
#         self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
#         self.model = TFElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator')
#         self.model = TFFunnelForSequenceClassification.from_pretrained('funnel-transformer/small-base' , num_labels = self.num_class)
#         self.tokenizer = FunnelTokenizer.from_pretrained('funnel-transformer/small-base')
        self.dataset = dataset
        self.current_aspect = aspect
    



    def re_initialize(self):
#         self.model = TFBertForSequenceClassification.from_pretrained('bert-base')
#         self.model = TFRobertaForSequenceClassification.from_pretrained('roberta-base')
        self.model = TFAlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=2)
#         self.model = TFXLNetForSequenceClassification.from_pretrained('xlnet-base-cased')
#         self.model = TFElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator')
#         self.model = TFFunnelForSequenceClassification.from_pretrained('funnel-transformer/small-base' , num_labels = self.num_class)
        

    
    
    def tokenize(self, dataset):
        input_ids = []
        attention_masks = []
        for sent in dataset:
            sent = str(sent)
            bert_inp = self.tokenizer .encode_plus(sent.lower(), add_special_tokens = True, max_length = 100, truncation = True, padding = 'max_length', return_attention_mask = True)
            input_ids.append(bert_inp['input_ids'])
            attention_masks.append(bert_inp['attention_mask'])

        train_input_ids = np.asarray(input_ids)
        train_attention_masks = np.array(attention_masks)

        return [train_input_ids,train_attention_masks]
    
    def model_compilation(self):

        #print('\nAlBert Model', self.model.summary())


        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5, epsilon = 1e-08)

        self.model.compile(loss = loss, optimizer = optimizer, metrics = [metric])

    
    def run_model(self):
        
        f1=[]
        recall = []
        precision = []
        auc = []
        mcc = []
        df = []
        for i in range(10):
            train = self.tokenize(self.dataset['X_Train'][i].to_list())
            test = self.tokenize(self.dataset['X_Test'][i].to_list())
            train_y = self.dataset['Y_Train'][i].to_numpy()
            test_y = self.dataset['Y_Test'][i].to_numpy()
            
            

            self.re_initialize()
            self.model_compilation()

            history = self.model.fit(train, train_y, batch_size = 32, epochs = 3, validation_data = (test,test_y), callbacks = callbacks.ReduceLROnPlateau(monitor='loss', factor=.2, patience=3, verbose=0, min_delta=1e-6, mode='min'))
            
            output = self.model.predict(test)
            
            
            

#             final_output = np.argmax(output, axis = -1)[0]
#             print(output)
            final_output = np.argmax(output['logits'], axis = -1)
            f1_score_val=f1_score(test_y, final_output, average = None)
            

            pre = precision_score(test_y, final_output, average = None)
            re = recall_score(test_y, final_output, average = None)
            ac = accuracy_score(test_y, final_output)
            rc=roc_auc_score(test_y, final_output)
            mc = matthews_corrcoef(test_y, final_output)

            f1.append(f1_score_val)
            precision.append(pre)
            recall.append(re)
            auc.append(rc)
            mcc.append(mc)
            df.extend(final_output)

#             print(f1_score_val, ac)

        #print(f1)
        return f1,precision,recall,auc, mcc, df
    def run_iot_model(self):
        
        f1=[]
        recall = []
        precision = []
        auc = []
        mcc = []
        for i in range(1):
            train = self.tokenize(opiner['Sentence'].to_list())
            test = self.tokenize(validation['sentence'].to_list())
            train_y = opiner['IsAboutSecurity'].to_numpy()
            test_y = validation['IsAboutSecurity'].to_numpy()
            
            

            self.re_initialize()
            self.model_compilation()

            history = self.model.fit(train, train_y, batch_size = 32, epochs = 3, validation_data = (test,test_y), callbacks = callbacks.ReduceLROnPlateau(monitor='loss', factor=.2, patience=3, verbose=0, min_delta=1e-6, mode='min'))
            
            output = self.model.predict(test)
            
            
            

#             final_output = np.argmax(output, axis = -1)[0]
            final_output = np.argmax(output['logits'], axis = -1)
    
            pro_score = [ val[0]/(val[0]+val[1]) for val in output['logits']]
            
            f1_score_val=f1_score(test_y, final_output, average = None)
            

            pre = precision_score(test_y, final_output, average = None)
            re = recall_score(test_y, final_output, average = None)
            ac = accuracy_score(test_y, final_output)
            rc=roc_auc_score(test_y, final_output)
            mc = matthews_corrcoef(test_y, final_output)

            f1.append(f1_score_val)
            precision.append(pre)
            recall.append(re)
            auc.append(rc)
            mcc.append(mc)
#             print(f1_score_val)
            p_fpr, p_tpr, _ = roc_curve(test_y, pro_score)
            

        #print(f1)

        return f1,precision,recall,auc, mcc, p_fpr, p_tpr, output['logits']
class_count = 2
aspect = 10
bert = BertModel(class_count,aspect)
f1,precision, recall, auc, mcc, df = bert.run_model()
print(sum(precision)[1]/10, sum(recall)[1]/10, sum(f1)[1]/10, sum(auc)/10, sum(mcc)/10)
f1,precision, recall, auc, mcc, df = bert.run_iot_model()
print(sum(precision)[1], sum(recall)[1], sum(f1)[1], sum(auc), sum(mcc))
