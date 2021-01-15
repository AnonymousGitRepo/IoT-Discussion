# Basic Info
This repository contains the implementation of IoT Security Discussions
# Requirements

*   Python
*   Numpy
*   Pandas
*   Keras
*   transformers
*   nltk
*   Tensorflow
*   scipy

# Materical Used
*   Glove Embedding Wikipedia 2014 + Gigaword- 200D (https://nlp.stanford.edu/projects/glove/)
*   Mallet (http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip)
*   BERT pretrained model and tokenizer( https://huggingface.co/models)

# Techniques/Algorithms Used
*   Natural Language Processings 
*   Deep Learning
*   SGD
*   Logistic Regression
*   LDA Mallet

# Code
You will find the codes of this project inside the "Codes" folder. The codes are organised in a sequential manner.

You need to download the datasets from corresponding source (please follow the 'Materials Used' section for sources) and keep them in the same folder with the codes before running. You will have to rename (or format) them as mentioned in the codes.

# Dataset Collection
*   BenchmarkUddinSO-ConsoliatedAspectSentiment: a dataset of 4,522 sentences from 1,338 StackOverflow posts created by Uddin and Khomh to develop the tool Opiner
*   StackOverflow IoT dataset: collected from IoT related 53K SO posts.
    List of all tags used in the IoT data collection:
    
    
** ***arduino***: 
arduino,
arduino-c++,
arduino-due,
arduino-esp8266,
arduino-every,
arduino-ide,
arduino-mkr1000,
arduino-ultra-sonic,
arduino-uno,
arduino-uno-wifi,
arduino-yun,
platformio


**  ***iot***: 
audiotoolbox,
audiotrack,
aws-iot,
aws-iot-analytics,
azure-iot-central,
azure-iot-edge,
azure-iot-hub,
azure-iot-hub-device-management,
azure-iot-sdk,
azure-iot-suite,
bosch-iot-suite,
eclipse-iot,
google-cloud-iot,
hypriot,
iot-context-mapping,
iot-devkit,
iot-driver-behavior,
iot-for-automotive,
iot-workbench,
iotivity,
microsoft-iot-central,
nb-iot,
rhiot,
riot,
riot-games-api,
riot.js,
riotjs,
watson-iot,
windows-10-iot-core,
windows-10-iot-enterprise,
windows-iot-core-10,
windowsiot,
wso2iot,
xamarin.iot


**  ***raspberry-pi***:
adafruit,
android-things,
attiny,
avrdude,
esp32,
esp8266,
firmata,
gpio,
hm-10,
home-automation,
intel-galileo,
johnny-five,
lora,
motordriver,
mpu6050,
nodemc,
omxplayer,
raspberry-pi,
raspberry-pi-zero,
raspberry-pi2,
raspberry-pi3,
raspberry-pi4,
raspbian,
serial-communication,
servo,
sim900,
teensy,
wiringpi,
xbee


# Code Replication
After tuning, the best hyperparameters for each model are provided in the codes. Please follow the following steps to replicate this repo- 
1. download BenchmarkUddinSO-ConsoliatedAspectSentiment.xls from Data folder
1. run 1_shallow_models.py from Codes folder to get performaces of Baseline-SVM and Logits
1. download kfold_cross_validation_dataset_security_aspect.p from Data folder(Recomended) **or** run 2_10_fold_cross_validation_dataset.py. A file named kfold_cross_validation_dataset_security_aspect.p will be created in your environment. This file may differ from provided one beacuse of random variable. 
1. run 3_DNN_models.py, 4_CNN_model.py, 5_LSTM_model.py, 6_Bi_LSTM_model.py to get performances of Deep Neural Network, CNN, LSTM and Bi-LSTM respectively.
1. run 7_BERT_model.py, 8_RoBERTa_model.py, 9_XLNet_model.py, 10_DistilBERT_model.py, 11_ALBERT_model.py, 12_FunnelBERT_model.py, 13_Electra_model.py to get performances of BERT, RoBERTa, XLNet, DistilBERT, ALBERT, FunnelBERT, and Electra respectively.
1. run 14_SecBot.py to get SecBot model described in our Paper. SecBot_weights.p file will be created in your environment after successful completion of this step. 
1. download Agreement_IoT_Training_samples.xlsx dataset. This file contains opinions of two coders either the sentence is security related or not e.g. 1 indicates security related and 0 indicates not. 
1. download IoT_Training_samples.xlsx dataset from Data folder. This dataset are the new 1000 training samples. 
1. run 15\_SecBot+.py to get Sec\_Bot+ model. SecBot+\_weights.p will be cretaed in your environment
1. download Agreement\_Random_dataset.xlsx and Agreement_Judgemental_dataset.xlsx dataset from Data folder. These files contain opinions of two coders either the sentence is security related or not e.g. 1 indicates security related and 0 indicates not.
1. download Random_dataset.xlsx(384 rows) and Judgemental_dataset.xlsx (600 rows).
1. run 16_Performance\_evaluation\_of\_SecBot\+\_and\_SecBot.py to get performances of SecBot and SecBot+
1. run 17_Label_SO_IoT_Dataset_using_SecBot+.py to label all sentences of SO_IoT_sentences.csv. A new file IoT_Security_dataset.csv will be created.
1. Make sure, you have downloaded Mallet from provided links and unzipped mallet-X(e.g. mallet2.0.8) to mallet-X/bin/mallet. run 18_Topic_Modeling.py and IoT_security_topics.xlsx file will be created. This file also present inside the Data folder. You can get 11 topics name from that file.
1. downlaod IoTPostInformation.csv, QuestionTags.pkl, and SO_IoT_sentences.rar file to create evolution charts and figures.

_N.B. Performances depends on random state. Random state may differ environment to environment and performance may also vary._
