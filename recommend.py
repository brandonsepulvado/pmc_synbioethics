# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:45:00 2021
@author: ddabl
"""

import transformers
from transformers import BertModel, BertTokenizer, AdamW , get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModel
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from ast import literal_eval
import os
from numpy import dot
from numpy.linalg import norm

#settings
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

"""
Multi-layer model architecture:
- BioBert will perform initial embedding based on word tokens
- fully connected layer 1 (FC1) will perform a document embedding
- fully connected layer 2 (FC2) will perform a class selection based on softmax
The model contains a combined loss function that uses:
(1) contrastive loss for FC1 (to make the embeddings close in vector space) plus
(2) cross-entropy loss for class selection.

Model actions:
- The model predicts the class that the SB article belongs to based solely on its abstract
- Then it multiplies metadata on the SB article with a list of ELSI articles from the same class
- It finds the ELSI article with the highest correlation based on number of common keywords
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device) 


print(transformers.__version__) 

#PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
PRE_TRAINED_MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1'
#PRE_TRAINED_MODEL_NAME = 'allenai/scibert_scivocab_cased'
#PRE_TRAINED_MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'

tokenizer = transformers.BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
  
MAX_LEN = 512 #160

class SBKSDataset(Dataset):
  
  def __init__(self,abstracts,targets,tokenizer,max_len,titles,feats,typs,tok_lens,docs,pmids):
    self.title = titles
    self.feats = feats
    self.type = typs
    self.token_len = tok_lens
      
    self.abstracts = abstracts
    #self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.docs = docs
    self.pmids = pmids
  
  def __len__(self):
    return len(self.abstracts)
    #return len(self.reviews)
  
  def __getitem__(self, item):
    abstract = str(self.abstracts[item])
    #review = str(self.reviews[item])
    target = self.targets[item]
    title = self.title[item]
    feat = self.feats[item]
    typ = self.type[item]
    tok_len = self.token_len[item]
    doc = self.docs[item]
    pmid = self.pmids[item]

    encoding = self.tokenizer.encode_plus(
      abstract,
      #review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {     
      'abstract': abstract,
      
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long),
      'title': title,
      'feats': feat,
      'type': typ,
      'tok_len': tok_len,
      'doc': doc,
      'pmid': pmid
    }


#shapira
df_val = pd.read_csv('.../val_data.csv')
df_train = pd.read_csv('.../train_data.csv')
df_test = pd.read_csv('.../test_data.csv')

def create_data_loader(df, tokenizer, max_len, batch_size):
    
    ds = SBKSDataset(
       
    abstracts=df.abstr.to_numpy(),
    
    targets=df.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len,
    titles=df.title.to_numpy(),
    #feats =df.flist.to_numpy(),
    feats =df.feats.to_numpy(),
    typs = df.type.to_numpy(),
    tok_lens = df.token_len.to_numpy(),
    docs = df.doc.to_numpy(),
    pmids = df.pmid.to_numpy()
    )

    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0 #4
    )

BATCH_SIZE = 3 #8 #16

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)


class SBKSClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SBKSClassifier, self).__init__()
    #self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.bert = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    #self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    
    
    out = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask)
    
   
    return self.out(out[1]), out[1]




class_names = set(list(df_val.label.to_numpy()))
print(class_names) #{0, 1, 2, 3, 4}
class_names = list(class_names)
print(class_names) #[0, 1, 2, 3, 4]
print(type(class_names))

#############################################################################

model = SBKSClassifier(len(class_names))

saved_model = '.../sh4n_bb_reg_4_1.bin'
#saved_model = '.../sh4n_sci_reg_c.bin'
#saved_model = '.../sh4n_pmb_reg_ab.bin'
#print(saved_model)

model.load_state_dict(torch.load(saved_model))
model = model.to(device)


def get_predictions(model, data_loader):
  model = model.eval()
  abstr_texts = []
  predictions = []
  prediction_probs = []
  real_values = []
  #feat_list = []
  title_list=[]
  doc_list = []
  pmid_list=[]
  typ_list =[]
  tok_len =[]
  
  emb_count = 0 
  doc_list = []
  
  len_d = len(data_loader)*3 
  doc_emb = np.zeros((len_d,771)) 
  
  count = 0
  with torch.no_grad():
    for d in data_loader:
        
      texts = d['abstract'] 
      
      input_ids = d['input_ids'].to(device)
      
      attention_mask = d['attention_mask'].to(device)
      targets = d['targets'].to(device)
      
      
      outputs,out = model(
      input_ids=input_ids,
      attention_mask=attention_mask)
      #print('out ',out.shape, out) #torch.Size([3, 768]) 
      
      _, preds = torch.max(outputs, dim=1)
      abstr_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(outputs)
      real_values.extend(targets)
      
      title = d['title']  
      doc = d['doc']
      pmid = d['pmid']
      typ =d['type']
      tok =d['tok_len']
      
      title_list.extend(title)
      doc_list.extend(doc)
      pmid_list.extend(pmid)
      typ_list.extend(typ)
      tok_len.extend(tok)
          
      doc = d['doc'].detach().cpu().numpy()
      typ = d['type'].detach().cpu().numpy()
      preds = preds.detach().cpu().numpy()
      
      input_ids = input_ids.detach().cpu().numpy()
      
      #anchor.detach().cpu().numpy()
      
      """ applies if dataset == val set"""
      
      pooled = out.detach().cpu().numpy()
      
      bs = pooled.shape[0]
      #trip_size = int(bs / 3)
      #print(trip_size)
      #for i in range(trip_size):
      for i in range(bs):
          doc_emb[emb_count+i,0]= doc[i]
          doc_emb[emb_count+i,1]= typ[i]
          doc_emb[emb_count+i,2]= preds[i]
          doc_emb[emb_count+i,3:] = pooled[i]
      
      emb_count+=bs
      
      count+=1
      
  
  df = pd.DataFrame(doc_emb)
  savedf = '.../val_doc_emb_bb.csv'
  print(savedf)
  df.to_csv(savedf,index=False)
      
  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  
  return abstr_texts, predictions, prediction_probs, real_values,title_list,doc_list,pmid_list,typ_list,tok_len
  


y_abstr_texts, y_pred, y_pred_probs, y_test, title_list,doc_list,pmid_list,typ_list,tok_len= get_predictions(
  model, val_data_loader)

#####################################################################################

#generate a classification report
class_names =['0', '1', '2', '3', '4']

y_names = set(list(y_pred.numpy()))
print(y_names) #{0, 1, 2, 3, 4}

yt_names = set(list(y_test.numpy()))
print(yt_names) #{1, 2, 3, 4,5}

print(classification_report(y_test, y_pred, target_names=class_names,digits=4))
#########################################################################
#track the incorrect predictions so can analyze
yw_act=[]
yw_pred = []
yw_abstr=[]
yw_titl=[]
yw_doc=[]
yw_ntok=[]
yw_typ=[]

for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        yw_act.append(y_test[i].numpy())
        yw_pred.append(y_pred[i].numpy())
        yw_abstr.append(y_abstr_texts[i])
        yw_titl.append(title_list[i])
        yw_doc.append(doc_list[i].numpy())
        yw_ntok.append(tok_len[i].numpy())
        yw_typ.append(typ_list[i].numpy())
        
print(len(yw_act)) #173

dw = pd.DataFrame(columns=['doc', 'title','abstr',
        'actual',  'pred','ntok',  'type'])


for n in range(len(yw_act)):
    dw = dw.append({
            'doc': yw_doc[n], 
            'title': yw_titl[n],
            'abstr': yw_abstr[n],
            'actual': yw_act[n],
            'pred': yw_pred[n],
            'ntok': yw_ntok[n],
            'type': yw_typ[n]
            },
            ignore_index=True)
#was saving to 0504/0430 for just bert
dw.to_csv('.../sh4n_inc.csv',
                index=False)

############################################################
def show_confusion_matrix(confusion_matrix):
  fig = plt.figure()
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.title('Shapira Dataset Confusion Matrix')
  plt.ylabel('True class')
  plt.xlabel('Predicted class');
  savefig = '.../sh4n_confus_trip_bb.png'
  
  fig.savefig(savefig, dpi=fig.dpi)

  plt.show()

cm = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

show_confusion_matrix(df_cm)


##############################################################################
""" consolidate / build a single ESLI emb doc file from
 train, test, val files and sb emb doc from val file"""

def get_preds(model, data_loader,ds):
  model = model.eval()
  abstr_texts = []
  predictions = []
  prediction_probs = []
  real_values = []
  #feat_list = []
  title_list=[]
  doc_list = []
  pmid_list=[]
  typ_list =[]
  tok_len =[]
  
  emb_count = 0 #0 is for doc #
  doc_list = []
  
  len_d = len(data_loader)*3 
  doc_emb = np.zeros((len_d,771)) #771 vs 768 so that addl index col for doc #:
  
  count = 0
  with torch.no_grad():
    for d in data_loader:
        
      #if count > 1:
      #    break
      texts = d['abstract'] #mb not a tensor bec strings vs num
      #print(d["input_ids"].shape, d["input_ids"])
      
      input_ids = d['input_ids'].to(device)
      
      attention_mask = d['attention_mask'].to(device)
      targets = d['targets'].to(device)
      
      outputs,out = model(
      input_ids=input_ids,
      attention_mask=attention_mask)
      
      _, preds = torch.max(outputs, dim=1)
      abstr_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(outputs)
      real_values.extend(targets)
      
      title = d['title']  
      doc = d['doc']
      pmid = d['pmid']
      typ =d['type']
      tok =d['tok_len']
      
      title_list.extend(title)
      doc_list.extend(doc)
      pmid_list.extend(pmid)
      typ_list.extend(typ)
      tok_len.extend(tok)
          
      doc = d['doc'].detach().cpu().numpy()
      typ = d['type'].detach().cpu().numpy()
      preds = preds.detach().cpu().numpy()
      
      input_ids = input_ids.detach().cpu().numpy()
      
      pooled = out.detach().cpu().numpy()
      
      bs = pooled.shape[0]
      #trip_size = int(bs / 3)
      #print(trip_size)
      #for i in range(trip_size):
      for i in range(bs):
      
          if typ[i]==1:
              doc_emb[emb_count,0]= doc[i]
              doc_emb[emb_count,1]= typ[i]
              doc_emb[emb_count,2]= preds[i]
              doc_emb[emb_count,3:]= pooled[i]
              emb_count+=1
          #emb_count+=bs
      
      
      count+=1
  
  df = pd.DataFrame(doc_emb)
  if ds=='train':
      savedf = '.../train_doc_emb.csv'
  if ds=='test':
      savedf = '.../test_doc_emb.csv'
  
  print(savedf)
  df.to_csv(savedf,index=False)
      
  
#without feat_list
get_preds(model, train_data_loader,ds='train')

get_preds(model, test_data_loader,ds='test')

#######################################


trn = pd.read_csv('.../train_doc_emb.csv')
tst = pd.read_csv('.../test_doc_emb.csv')
#val file must prune to extract elsi only
val = pd.read_csv('.../val_doc_emb.csv')           

#get elsi only from val        

#get length of non-zero val doc
docs = val['0'].to_numpy()
doc_count=0
for i in range(len(docs)):
    if docs[i]>0:
        doc_count+=1
print('doc count: ',doc_count)

typs = val['1'].to_numpy()
typs
#Out[4]: array([0., 0., 0., ..., 0., 0., 0.])

#this allows to create empty array of size esli docs
num = 0
for t in range(len(typs)):
    if typs[t]==1:  #elsi article
        num+=1
print(num) #63 #elsi docs only

#elsi only embeddings#####################       
elsi_docs = np.zeros((num,771))        

embn = np.array(val)
embn.shape #(1731, 771)
elsi_docs.shape #(63, 771)

count=0
for t in range(len(typs)):
    if typs[t]==1:  #elsi article
        elsi_docs[count,:]= embn[t,:]
        count+=1

elsi_docs.shape#(63, 771)


#create train elsi only np file
trnn = np.array(trn)
typtrn = trn['1'].to_numpy()
typtrn_sum = int(np.sum(typtrn))
typtrn_sum 
trn_elsi = trnn[:typtrn_sum,:]

#create test only np file
tstn = np.array(tst)
typtst = tst['1'].to_numpy()
typtst_sum = int(np.sum(typtst))
typtst_sum 
tst_elsi = tstn[:typtst_sum,:]


elsi = np.vstack((elsi_docs, trn_elsi, tst_elsi))
elsi.shape #(906, 771)

df_el = pd.DataFrame(elsi)
df_el.to_csv('.../elsi_emb.csv',
                index=False)

#sb only doc embeddings#################

#this allows to create empty array of size sb docs
num = 0

#for t in range(len(typs)):
for t in range(doc_count):
    if typs[t]==0:  #sb article
        num+=1
print(num) 

#sb only embeddings       
sb_docs = np.zeros((num,771))
#sb_docs = np.zeros((doc_count,771))

count=0
for t in range(doc_count):

    if typs[t]==0:  #sb article
        sb_docs[count,:]= embn[t,:]
        count+=1

df_sb = pd.DataFrame(sb_docs)
df_sb.to_csv('.../sb_docs.csv',
              index=False)


#############################################################################

"""recommend an elsi paper based on nearest doc distance to sb doc embedding"""

df_el = pd.read_csv('.../elsi_emb.csv')

sb_el = pd.read_csv('.../sb_docs.csv')

elsi = np.array(df_el)
sb_docs = np.array(sb_el)

#for each sb doc embedding, find the most "similar" elsi doc from the same class,
#where similarity is measured by doc embedding distance
#because we search by class, we limit the search space######################

#organize the elsi docs by class based on id
labels = df_el['2'].to_numpy()
lab_set = set(labels)
#lab_set #{0, 1, 2, 3, 4}
elsi_lab_ind = {label: np.where(labels == label)#[0]
             for label in lab_set}
elsi_lab_ind


###################################################################

#get a list of all docs so can search for elsi metadata(abstract,title, etc)
#all_data = pd.read_csv('C:/Users/ddabl/OneDrive/Documents/Networks/SBKS/data/0504/0430/data_len1.csv')
all_data = pd.read_csv('.../all_data.csv')

ad = all_data.to_numpy()

doclist = all_data['doc']
doclist = list(doclist)
doclist,type(doclist)
len(doclist) #10109
len(y_pred) #1515
len(doc_list) #1515
len(sb_docs) #1385

sb_docs_docs = sb_el['0'].to_numpy()
sb_docs_docs

#tot = len(fl)

y_pred = y_pred.numpy()
y_pred = list(y_pred)

#print(doc_list)

doc_list = [int(t.numpy()) for t in doc_list]
#doc_list = [int(t) for t in doc_list]
doc_list
len(doc_list) #1515
pdd = pd.DataFrame(doc_list)
pdd.to_csv('.../pdd.csv',
           index=False)


n_sb=0
n_e=0
acc = 0

dpred = pd.DataFrame(columns=['sb_doc', 'sb_title','sb_abstr',
        'el_doc', 'el_title','el_abstr','el_pmid','pred_y','act_y'])

#for s in range(1):
for s in range(len(sb_docs)):
    #print('i ',i)
    sb_emb = sb_docs[s,3:] #sb doc embedding
    sb_id = int(sb_docs[s,0])
    
    sb_ind = doc_list.index(sb_id)
    
    
    sb_cl = y_pred[sb_ind]
    
    ids = elsi_lab_ind[sb_cl]
    E = elsi[ids]
    
    E_len = len(E)
    sb_el_dist=np.zeros((E_len))
    
    for i in range(E_len):
        e_emb = E[i,3:] #elsi embedding
        dist = np.subtract(sb_emb,e_emb) #.pow(2).sum() 
        #print('dist ',dist)
        dist = np.absolute(dist)
        dmean = np.mean(dist)
        #print(dmean)
        sb_el_dist[i] =dmean
    min_ind = sb_el_dist.argmin() #find min doc distance
    #print(min_ind)
    elsi_id =  E[min_ind,0]
    #print('elsi id ',elsi_id)#5017775.0
    
    sb_id = int(sb_id)
    elsi_id = int(elsi_id)
    
    yp = y_pred[sb_ind] #.numpy()
    ya = y_test[sb_ind].numpy()
    
    if yp == ya:
        acc+=1
    n_sb+=1
    
    doc_ind = doclist.index(elsi_id)
    
    
    
    dpred = dpred.append({
            #'sb_doc': doc_list[sb_ind].numpy(), 
            'sb_doc': doc_list[sb_ind], 
            'sb_title': title_list[sb_ind],
            'sb_abstr': y_abstr_texts[sb_ind],
            'el_doc': ad[doc_ind][0], 
            'el_title': ad[doc_ind][2],
            'el_abstr': ad[doc_ind][3],
            'el_pmid': ad[doc_ind][1],
            #'pred_y': y_pred[sb_ind], #.numpy(),
            #'act_y': y_test[sb_ind].numpy()
            'pred_y': yp, #.numpy(),
            'act_y': ya
            
            
            },
            ignore_index=True)

#was saving to 0504/0430 for just bert

dpred.to_csv('.../shap4_rec_val.csv',
                index=False)
     
    
print('Number of Syn Bio abstracts: %d' %(n_sb))

print('Accurate class predictions (recommendations): %d; %.2f percent' %(acc, acc/n_sb*100))


##############################################################################
##############################################################################   
    
    
   