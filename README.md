# README

## draft_config.json
- Config file for accessing PubMed Central (API key, login parameters)
- Called in download.py

## download.py
- Utility functions that download full text files from PubMed Central
    - Downloads search results into XML files
    - Parses the XML files into json files in a predetermined format, including information about:
        - Article ID
        - Title
        - Authors
        - Abstract
        - Article body
        - References
- **Output:**
    - Intermediate xml files in data/xml_el_key, which are then parsed further and converted into json
    - Final json files in data/json_el_key

## LDA.py
- Performs topic modeling using LDA on the Synthetic Bio and ELSI documents (abstracts)
    - Finds key terms that divide the document list into 5 topics 
    - Identifies 5 classes of papers - the topics become the class labels
- **Input:** all json files in SBKS/data_0523/json_sb4_n/all_data.csv
- **Output:** none, prints a list of top 5 topics

## comm_freq1.py
- Builds a word frequency list for the abstracts in the 5 topics
    - Iterates through abstracts and calculates frequency of topic words per document
- Since there are overlapping examples (documents which are in more than one class), performs data cleaning
    - Counts the document’s number of keyword “hits” for each topic
    - Topic with the most keyword “hits” in the document is assigned as the class label for the article
- **Output:**
    - abstracts.csv: a list of documents that have non-empty abstracts
    - t1.csv to t5.csv: sample lists of documents for classes 1 through 5 (topics 1 through 5) respectively
    - Additional subsets of t1.csv to t5.csv
        - e.g. sb1.csv: sample list of documents for class 1 that are only Synthetic Biology
        - e.g. elsi1.csv: sample list of documents for class 1 that are only ELSI
    - freq_all.csv: shows the frequency of keyword hits per document per word
        - Each row is a document
        - Each column is a word in a topic
        - Each cell shows the number of times a keyword appeared in a given document
        - So that we can count the frequency of each word in a topic
    - nump.csv: shows the sum of keyword hits for each topic (t1, t2, etc)
        - Each row is a document
        - Each column is a topic
        - Each cell shows the total number of keywords for a topic in a given document
        - So that we can compare relative overlap of topics
        
## dataf_shap4n.py
- Utility functions that create train/test/validation datasets
- **Code for the actual train/test/validation split seems to be missing**
    - But is apparently a 70/15/15 split
- **Input:**
    - json files for model pipeline
        - Synthetic Biology jsons in C:/Users/ddabl/OneDrive/Documents/Networks/SBKS/data_0523/json_sb4_n
        - ELSI jsons in C:/Users/ddabl/OneDrive/Documents/Networks/SBKS/data_0523/json_el_0608
    - csv files all located at C:/Users/ddabl/OneDrive/Documents/Networks/SBKS/data_0523/data_shap4n/
        - Lists of documents per topic/class
            - t1.csv to t5.csv (all)
            - sb1.csv to sb5.csv (only Synthetic Biology)
            - elsi1.csv to elsi5.csv (only ELSI)
        - freq_all.csv, which shows the frequency of keyword hits per document per word
- **Output:**
    - freq_all.csv converted using a minmax scaler and saved at freq_all_ss.csv
    - Features matrix of topic keyword hits for each ELSI topic
        - elsi1_feat_ss.csv to elsi5_feat_ss.csv
    - full csv file for model pipeline at C:/Users/ddabl/OneDrive/Documents/Networks/SBKS/data_0523/data_shap4n/all_data.csv
        - Contains:
        - document ID
        - PMID
        - title
        - abstract
        - features
        - label
        - type (Synthetic Biology or ELSI)
        - origin folder (e.g. sb1, elsi2)

## model_train.py
- Uses BERT model to train a classifier that separates papers into 5 classes
- Code written in PyTorch
- Model details:
    - Multiple layers
        - BioBert performs initial embedding based on word tokens
        - Fully connected layer 1 (FC1) performs a document embedding
        - Fully connected layer 2 (FC2) performs a class selection based on softmax
    - Combined loss function that uses
        - Contrastive loss for FC1 (to make the embeddings close in vector space)
        - Cross-entropy loss for class selection
        - Model pipelines created to implement contrastive loss functions, e.g. Siamese networks and Triplet Loss
- Model predicts the class that the Synthetic Biology article belongs to based solely on its abstract
    - Model then multiplies metadata on the Synthetic Biology article with a list of ELSI articles from the same class
    - Model finds the ELSI article with the highest correlation based on number of common keywords
- **Output:** 
    - Best model saved at sh4n_pmb_reg_pt1_drp.bin
    - Model accuracy/loss history saved at hist_4n_pmb_reg_drp.csv

## recommend.py
- Produces paper recommendations based on similarity of document embeddings
- **Input:**
    - Pre-trained model from sh4n_bb_reg_4_1.bin
    - List of all documents at all_data.csv
    - Train/test/validation data
        - train_data.csv, test_data.csv, val_data.csv 
    - Document embeddings
        - train_doc_emb.csv, test_doc_emb.csv, val_doc_emb.csv
- **Output:**
    - Document embeddings separated by ELSI and Synthetic Biology
        - ELSI only: elsi_emb.csv
        - Synthetic Biology only: sb_docs.csv
    - Predictions saved at shap4_rec_val.csv
