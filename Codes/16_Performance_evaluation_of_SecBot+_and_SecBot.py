import pandas as pd
import pickle
from tensorflow.keras.optimizers import Adam
import keras
import numpy as np
from sklearn.metrics import classification_report as cr 
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from keras import Input,Model,callbacks
from numpy.random import seed
import tensorflow as tf
import pandas
from transformers import *
seed(1)
tf.random.set_seed(2)

sec_bot_model_weight = pickle.load(open('SecBot_weights.p','rb'))
sec_bot_plus_model_weight = pickle.load(open('SecBot+_weights.p','rb'))

random_dataset=pd.read_excel('Random_dataset.xlsx')
judgemental_dataset = pd.read_excel('Judgemental_dataset.xlsx')
overall_dataset = random_dataset.append(judgemental_dataset)


from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

num_class = 2
model = TFRobertaForSequenceClassification.from_pretrained('roberta-base' , num_labels = num_class)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

#pretained weight
model.set_weights(sec_bot_model_weight)


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
    


dataset = random_dataset['sentence']
label = random_dataset['IsAboutSecurity'].to_numpy()
tkn = new_tokenize(dataset)
pred_val = model.predict(tkn)
final_output = np.argmax(pred_val['logits'], axis = -1)
f1_score_val=f1_score(label, final_output, average = None)
pre = precision_score(label, final_output, average = None)
re = recall_score(label, final_output, average = None)
print(pre[1],re[1],f1_score_val[1])



dataset = judgemental_dataset['sentence']
label = judgemental_dataset['IsAboutSecurity'].to_numpy()
tkn = new_tokenize(dataset)
pred_val = model.predict(tkn)
final_output = np.argmax(pred_val['logits'], axis = -1)
f1_score_val=f1_score(label, final_output, average = None)
pre = precision_score(label, final_output, average = None)
re = recall_score(label, final_output, average = None)
print(pre[1],re[1],f1_score_val[1])

dataset = overall_dataset['sentence']
label = overall_dataset['IsAboutSecurity'].to_numpy()
tkn = new_tokenize(dataset)
pred_val = model.predict(tkn)
final_output = np.argmax(pred_val['logits'], axis = -1)
f1_score_val=f1_score(label, final_output, average = None)
pre = precision_score(label, final_output, average = None)
re = recall_score(label, final_output, average = None)
print(pre[1],re[1],f1_score_val[1])


######Sec_BOT+#########################
model.set_weights(sec_bot_plus_model_weight)

dataset = random_dataset['sentence']
label = random_dataset['IsAboutSecurity'].to_numpy()
tkn = new_tokenize(dataset)
pred_val = model.predict(tkn)
final_output = np.argmax(pred_val['logits'], axis = -1)
f1_score_val=f1_score(label, final_output, average = None)
pre = precision_score(label, final_output, average = None)
re = recall_score(label, final_output, average = None)
print(pre[1],re[1],f1_score_val[1])



dataset = judgemental_dataset['sentence']
label = judgemental_dataset['IsAboutSecurity'].to_numpy()
tkn = new_tokenize(dataset)
pred_val = model.predict(tkn)
final_output = np.argmax(pred_val['logits'], axis = -1)
f1_score_val=f1_score(label, final_output, average = None)
pre = precision_score(label, final_output, average = None)
re = recall_score(label, final_output, average = None)
print(pre[1],re[1],f1_score_val[1])

dataset = overall_dataset['sentence']
label = overall_dataset['IsAboutSecurity'].to_numpy()
tkn = new_tokenize(dataset)
pred_val = model.predict(tkn)
final_output = np.argmax(pred_val['logits'], axis = -1)
f1_score_val=f1_score(label, final_output, average = None)
pre = precision_score(label, final_output, average = None)
re = recall_score(label, final_output, average = None)
print(pre[1],re[1],f1_score_val[1])



