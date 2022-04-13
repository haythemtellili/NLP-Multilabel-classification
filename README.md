# Multilabel-classification-NLP
This repository contains a multi-label classifier that takes as input a web url and send back a list of categories for the URL. 

# Getting started
Prerequisites: 

 1. Python 3
 2. Anaconda 
 
 # Usage
 1. Clone this repo: `git clone https://github.com/haythemtellili/Multilabel-classification-NLP.git`
 2. Create conda environment: `conda create -n adot python=3.8`
 3. Activate the environment: `conda activate adot`
 4. Install the necessary packages: `pip3 install -r requirements.txt`
 ## Testing the Flask API
 In this step, we need to download the trained model with the label encoder and put them in a models directory.\
 Both are available [here](https://drive.google.com/drive/folders/1gSifqnsZU_MybP5MFMAg_r0CNAVFxf2v?usp=sharing).
 
 Start the service using the following command:
  ```bash
 python app.py
 ```
Example to test the Flask API from the terminal:
 ```bash
 curl -v http://127.0.0.1:9999/predict?url=https://dictionnaire.reverso.net/francais-arabe/
 ```
 ## Training
To train the model, we just need to run the following command:
  ```bash
 python train.py
 ```
During training, we evaluate the performance of the model, By the end of training we test the model using Testing Data.\
Results will be saved in performance.txt file.
 
 # Approach
 
 Our approach consist of:
 
 1. Preprocessing: In this step, We clean our data by removing duplicates, Filter categories that have fewer than 3 occurrences.\
 (We tried to clean up the url by removing stop words, digits and extracting only the words, but that doesn't help because the bert-based models were pre-trained on real data containing urls so they learned how to handle them properly).
 3. Validation strategy: Iterative train test split which maintains balanced representation . 
 4. Train `distilbert-base-multilingual-cased` model since the url can contain different languages.
 5. Validate the model after each epoch.
 6. Testing the model and finding the best threshold for f1 to be used for the REST API.
## Improvements
1. The data is unbalanced, so we can handle this by using weighted loss, oversampling the data for categories that have less than <min_tag_freq> occurrences.
2. Extract information from the web page using URLs to help the model generalize better.
3. Experiment with other language models.
4. Add rules to improve the quality of results (kind of post processing ..).
6. Evaluate each class separately and determine the threshold per label.
7. Integrate MLflow to track different experiments.
8. Set up CI pipeline.
## Final result:
Metrics used are: F1 score, Precision, Recall and AUC.

| F1 score| Recall    |Precision  |AUC|
| :-----: | :-: | :-: |:-: 
| 0.47531835838544656 | 0.5008549121397524 | 0.49137788420214845 |0.9758651239487387
 
