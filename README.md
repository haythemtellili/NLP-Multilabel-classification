# Multilabel-classification-NLP
This repository contains a multi-label classifier that takes as input a web url and send back a list of categories for the URL. 

# Getting started
Prerequisites: 

 1. Python 3
 2. Anaconda 
 
 # Usage
 1. Clone this repo: `git clone https://github.com/haythemtellili/Multilabel-classification-NLP.git`
 2. Create conda environment: `conda create -n adot python=3.8`
 3. Install the necessary packages: `pip3 install -r requirements.txt`
 ## Testing the Flask API
 In this step, we need to download the trained model with the label encoder and put them in a models directory.
 You can Donload them from this link.
 
 Start the service using the following command:
  ```bash
 python app.py
 ```
Example to test the Flask API from the terminal:
 ```bash
 curl -v http://127.0.0.1:9999/predict?url=https://dictionnaire.reverso.net/francais-arabe/
 ```
 ## Training
To train the model from scratch, we just need to train the following command:
  ```bash
 python train.py
 ```
During training, we evaluate the performance of the model, By the end of training we test the model using Testing Data. 
Result will be saved in performance.txt file.
 
 # Approach
 
 Our approach consist of:
 
 1. Preprocessing: In this step, We clean our data by removing duplicates, Filter categories that have fewer than <min_tag_freq> occurrences.
 2. Validation strategy: iterative train test split which maintains balanced representation . 
 3. Train `distilbert-base-multilingual-cased` model since the url can contain different languages.
 4. Validate the model after each epoch.
 5. Testing the model and finding the optimal threshold to be used for the REST API.

## Final result:
Metrics used are: F1 score, Precision, Recall and AUC.

| F1 score| Recall    |Precision  |
| :-----: | :-: | :-: |
| 0.4929 | 0.5111 | 0.5117 |
 
