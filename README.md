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

# Architectures
* Traditional Deep Learning Methods
-> We follow standard architecture of DNN, CNN, LSTM, Bi-LSTM that are ubiquotously used in various research works for sentiment analysis.   
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

_N.B. Performance depends on random state. Random state may differ environment to environment and performance may also vary._

@article{Dang_2020,
   title={Sentiment Analysis Based on Deep Learning: A Comparative Study},
   volume={9},
   ISSN={2079-9292},
   url={http://dx.doi.org/10.3390/electronics9030483},
   DOI={10.3390/electronics9030483},
   number={3},
   journal={Electronics},
   publisher={MDPI AG},
   author={Dang, Nhan Cach and Moreno-García, María N. and De la Prieta, Fernando},
   year={2020},
   month={Mar},
   pages={483}
}

@article{ALHARBI201950,
title = {Twitter sentiment analysis with a deep neural network: An enhanced approach using user behavioral information},
journal = {Cognitive Systems Research},
volume = {54},
pages = {50-61},
year = {2019},
issn = {1389-0417},
doi = {https://doi.org/10.1016/j.cogsys.2018.10.001},
url = {https://www.sciencedirect.com/science/article/pii/S1389041718300482},
author = {Ahmed Sulaiman M. Alharbi and Elise {de Doncker}},
keywords = {Opinion mining, Sentiment analysis, Social media, Deep learning, Natural language processing},
abstract = {Sentiment analysis on social media such as Twitter has become a very important and challenging task. Due to the characteristics of such data—tweet length, spelling errors, abbreviations, and special characters—the sentiment analysis task in such an environment requires a non-traditional approach. Moreover, social media sentiment analysis is a fundamental problem with many interesting applications. Most current social media sentiment classification methods judge the sentiment polarity primarily according to textual content and neglect other information on these platforms. In this paper, we propose a neural network model that also incorporates user behavioral information within a given document (tweet). The neural network used in this paper is a Convolutional Neural Network (CNN). The system is evaluated on two datasets provided by the SemEval-2016 Workshop. The proposed model outperforms current baseline models (including Naive Bayes and Support Vector Machines), which shows that going beyond the content of a document (tweet) is beneficial in sentiment classification, because it provides the classifier with a deep understanding of the task.}
}


@article{ABID2019292,
title = {Sentiment analysis through recurrent variants latterly on convolutional neural network of Twitter},
journal = {Future Generation Computer Systems},
volume = {95},
pages = {292-308},
year = {2019},
issn = {0167-739X},
doi = {https://doi.org/10.1016/j.future.2018.12.018},
url = {https://www.sciencedirect.com/science/article/pii/S0167739X18324944},
author = {Fazeel Abid and Muhammad Alam and Muhammad Yasir and Chen Li},
keywords = {Sentiment analysis, Word embeddings, Recurrent neural network (RNNs), Convolutional neural network (CNNs)},
abstract = {Sentiment analysis has been a hot area in the exploration field of language understanding, however, neural networks used in it are even lacking. Presently, the greater part of the work is proceeding on recognizing sentiments by concentrating on syntax and vocabulary. In addition, the task identified with natural language processing and for computing the exceptional and remarkable outcomes Recurrent neural networks (RNNs) and Convolutional neural networks (CNNs) have been utilized. Keeping in mind the end goal to capture the long-term dependencies CNNs, need to rely on assembling multiple layers. In this Paper for the improvement in understanding the sentiments, we constructed a joint architecture which places of RNN at first for capturing long-term dependencies with CNNs using global average pooling layer while on top a word embedding method using GloVe procured by unsupervised learning in the light of substantial twitter corpora to deal with this problem. Experimentations exhibit better execution when it is compared with the baseline model on the twitter’s corpora which tends to perform dependable results for the analysis of sentiment benchmarks by achieving 90.59% on Stanford Twitter Sentiment Corpus, 89.46% on Sentiment Strength Twitter Data and 88.72% on Health Care Reform Dataset respectively. Empirically, our work turned to be an efficient architecture with slight hyperparameter tuning which capable us to reduce the number of parameters with higher performance and not merely relying on convolutional multiple layers by constructing the RNN layer followed by convolutional layer to seizure long-term dependencies.}
}


@article{Ain2017,
title = {Sentiment Analysis Using Deep Learning Techniques: A Review},
journal = {International Journal of Advanced Computer Science and Applications},
doi = {10.14569/IJACSA.2017.080657},
url = {http://dx.doi.org/10.14569/IJACSA.2017.080657},
year = {2017},
publisher = {The Science and Information Organization},
volume = {8},
number = {6},
author = {Qurat Tul Ain and Mubashir Ali and Amna Riaz and Amna Noureen and Muhammad Kamran and Babar Hayat and A. Rehman}
}

