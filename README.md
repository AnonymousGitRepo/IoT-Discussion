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

# Architecture Overview
*   Traditional Deep Learning Methods : We follow the standard architecture of traditional LSTM that is ubiquitously used in various research works for sentiment         analysis [[1]](#1) [[2]](#2) [[3]](#3) [[4]](#4). We use a single neuron with sigmoid activation for the output layer that means we classify an instance positively only if         output is greater or equals to .5. We follow standard grid search algorithms for fine-tuning hyperparameters. A brief overview of these architectures are given below-


    * ___LSTM___: We follow the architecture used by Alharbi at el.[[4]](#4). After fine tuning, our final LSTM model has an input layer followed by a lstm layer, a dense layer, flatten layer, again a dense layer and the output layer. For a detailed hyperparameter set, please follow 4_LSTM_model.py.
    * ___BiLSTM___: We implement a single layer BiLSTM model following the proposed architecture by Hameed at el. [[6]](#6) for sentiment detection of a single sentence. Architure contains a single bi-directional lstm layer followed by a pooling layer, a concatenation layer and the final output layer. For a detailed hyperparameter set, please follow 5_Bi_LSTM_model.py.
*   BERT based Advanced DL Models: BERT based model pre-trained on a large corpus of English data, over multiple gigantic datasets like Wikipedia, CC-News, OpenWebText, and etc,  in a self-supervised fashion with the Masked language modeling (MLM) objective. Taking a sentence, the model randomly masks 15% of the words in the input then runs the entire masked sentence through the model and has to predict the masked words. On the top of this embedding, we use pre-trained BERT sequence classifier that classifies sentence/ sentences according to a given number of classes. These pre-trained classifier has a high computation architecture that is pre-trained on GLUE dataset. We set the output layer with 2 neurons(Security & Non-security) and classified each sentence into the class having the maximum value among the classes. We then tune the parameters on the training dataset. There already exists a suggested hyperparameter set in huggingface for these transformers model. For example, suggested hyperparameters for BERT sequence classifiers are batch-size = 32, learning-rate = 3e-5, epochs = 3, max-sequence-length = 128. We applied grid search hyperparameter tuning here. For example, our batch-size tuning sets are  {16,32,64}, learning-rate are {1e-5, 3e-5} and epochs are {2,3,4}. A brief overview of used BERT-based models are given below-


     * ___BERT___: We use bert-base-uncased[[5]](#5) pre-trained model that is trained on BookCorpus, a dataset consisting of 11,038 unpublished books and English Wikipedia. Model has 110M parameters.
     * ___RoBERTa___: We use roberta-base[[5]](#5) model that is trained on a large collection of five datasets e.g. English Wikipedia, CC-News, OpenWebText, Stories, and BookCorpus, that combinedly equal to 160GB of texts. This model has around 125M parameters.
     * ___XLNet___: We use xlnet-base-cased model that is trained on English Wikipedia, and BookCorpus, and etc. This model has around 110M parameters.
     * ___DistilBERT___: We use distilbert-base-uncased[[7]](#7) model that is trained on same data as BERT. This model squeezes the layers which result in a smaller parameter than BERT of size 66M.
     * ___ALBERT___: We use albert-base-v1 model that is trained on same data as BERT. This model has only 11M parameters.

     
     


# Code
You will find the codes of this project inside the "Codes" folder. The codes are organised in a sequential manner.

You need to download the datasets from corresponding source (please follow the 'Materials Used' section for sources) and keep them in the same folder with the codes before running. You will have to rename (or format) them as mentioned in the codes.

# Dataset Collection
*   BenchmarkUddinSO-ConsoliatedAspectSentiment: a dataset of 4,522 sentences from 1,338 StackOverflow posts created by Uddin and Khomh to develop the tool Opiner
*   StackOverflow IoT dataset: collected from IoT related 53K SO posts.
    List of all tags used in the IoT data collection:
    
    
    * ___arduino___: arduino, arduino-c++, arduino-due, arduino-esp8266, arduino-every, arduino-ide, arduino-mkr1000, arduino-ultra-sonic, arduino-uno, arduino-uno-wifi, arduino-yun, platformio
    * ___iot___: audiotoolbox, audiotrack, aws-iot, aws-iot-analytics, azure-iot-central, azure-iot-edge, azure-iot-hub, azure-iot-hub-device-management, azure-iot-sdk, azure iot-suite, bosch-iot-suite, eclipse-iot, google-cloud-iot, hypriot, iot-context-mapping, iot-devkit, iot-driver-behavior, iot-for-automotive, iot-workbench, iotivity, microsoft iot-central, nb-iot, rhiot, riot, riot-games-api, riot.js, riotjs, watson-iot, windows-10-iot-core, windows-10-iot-enterprise, windows-iot-core-10, windowsiot, wso2iot, xamarin.iot
    * ___raspberry-pi___: adafruit, android-things, attiny, avrdude, esp32, esp8266, firmata, gpio, hm-10, home-automation, intel-galileo, johnny-five, lora, motordriver, mpu6050, nodemc, omxplayer, raspberry-pi, raspberry-pi-zero, raspberry-pi2, raspberry-pi3, raspberry-pi4, raspbian, serial-communication, servo, sim900, teensy, wiringpi, xbee


# Code Replication
After tuning, the best hyperparameters for each model are provided in the codes. Please follow the following steps to replicate this repo- 
1. download and install all requirements and materials stated above 
1. download all files from Data folder
1. IoT_Training_Samples_Agreement.xlsx, IoT_Validation_Samples_Agreement.xlsx file contain opinions of two coders either the sentence is security related or not e.g. 1 indicates security related and 0 indicates not. 
1. IoT_Training_samples.xlsx dataset contains new 1000 training samples that are collected from IoT dataset. 
1. Opiner_Samples.xlsx contains 4297 samples and Combined_Training_Samples.xlsx contains 5297 training samples. Validation_Samples.xlsx contains 984 samples that are used for testing. 
1. run 1_shallow_models.py from Codes folder to get performaces of Baseline-SVM and Logits
1. run 2_DLL_models.py to get performances of LSTM and Bi-LSTM respectively.
1. run 3_BERT_model.py, 4_Roberta_model.py, 5_Albert_model.py, 6_XLNet_model.py, 8_Distilbert_model.py to get performances of BERT, RoBERTa, ALBERT, XLNet, and DistilBERT respectively.
1. run 9_Label_SO_IoT_Dataset_using_SecBot+.py to label all sentences of SO_IoT_sentences.csv. A new file IoT_Security_dataset.csv will be created.
1. Make sure, you have downloaded Mallet from provided links and unzipped mallet-X(e.g. mallet2.0.8) to mallet-X/bin/mallet. run 10_Topic_Modeling.py and IoT_security_topics.xlsx file will be created. This file also present inside the Data folder. You can get 11 topics name from that file.
1. downlaod IoTPostInformation.csv, QuestionTags.pkl, and SO_IoT_sentences.rar file to create evolution charts and figures.

_N.B. Performance depends on random state. Random state may differ environment to environment and performance may also vary._

## References
<a id="1" >[1]</a>
N. C. Dang, M. N. Moreno-Garca, and F. De la Prieta, "Sentiment analysis based on deeplearning: A comparative study" Electronics, vol. 9, p. 483, Mar 2020.  

<a id="2" >[2]</a>
F. Abid,  M. Alam,  M. Yasir,  and C. Li,  "Sentiment analysis through recurrent variantslatterly on convolutional neural network of twitter" Future Generation Computer Systems,vol. 95, pp. 292–308, 2019.  

<a id="3" >[3]</a>
Q. T. Ain, M. Ali, A. Riaz, A. Noureen, M. Kamran, B. Hayat, and A. Rehman, "Sentimentanalysis using deep learning techniques:  A review" International Journal of AdvancedComputer Science and Applications, vol. 8, no. 6, 2017  

<a id="4" >[4]</a>
. S. M. Alharbi and E. de Doncker, "Twitter sentiment analysis with a deep neural net-work:  An enhanced approach using user behavioral information" Cognitive Systems Re-search, vol. 54, pp. 50–61, 2019.  

<a id="5" >[5]</a>
Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer,and V. Stoyanov,  "Roberta:  A robustly optimized BERT pretraining approach" CoRR,vol. abs/1907.11692, 2019.  

<a id="6">[6]</a>
Z. Hameed and B. Garcia-Zapirain, "Sentiment Classification Using a Single-Layered BiLSTM Model," in IEEE Access, vol. 8, pp. 73992-74001, 2020, doi: 10.1109/ACCESS.2020.2988550.  

<a id="7">[7]</a>
J. Devlin, M. Chang, K. Lee, and K. Toutanova, "BERT: pre-training of deep bidirectionaltransformers for language understanding" CoRR, vol. abs/1810.04805, 2018.  

<a id="8">[8]</a>
.  Sanh,  L.  Debut,  J.  Chaumond,  and  T.  Wolf,  “Distilbert,  a  distilled  version  of  bert:smaller, faster, cheaper and lighter,”ArXiv, vol. abs/1910.01108, 2019.
