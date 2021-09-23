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
import time

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
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device) 

print('transformer: ',transformers.__version__) 

#PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

#PRE_TRAINED_MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1'

PRE_TRAINED_MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'

#PRE_TRAINED_MODEL_NAME = 'allenai/scibert_scivocab_cased'

print(PRE_TRAINED_MODEL_NAME)

#PRE_TRAINED_MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
#PRE_TRAINED_MODEL_NAME = 'allenai/scibert_scivocab_uncased'

# Download vocabulary from huggingface.co and cache.
#tokenizer = transformers.BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


MAX_LEN = 512 


class SBKSDataset(Dataset):
  
  def __init__(self,abstracts,targets,tokenizer,max_len,titles,feats,typs,tok_lens,docs,pmids):
    self.title = titles
    self.feats = feats
    self.type = typs
    self.token_len = tok_lens
      
    self.abstracts = abstracts
    
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.docs = docs
    self.pmids = pmids
  
  def __len__(self):
    return len(self.abstracts)
    
  
  def __getitem__(self, item):
    abstract = str(self.abstracts[item])
    
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
      #'review_text': review,
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

#shapira 4n 70/15/15 - reg / reg
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

#example batch from the training data loader:
data = next(iter(train_data_loader))
print(data.keys())


print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)

#create a classifier that uses the BERT model:

class SBKSClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SBKSClassifier, self).__init__()
    #self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.bert = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.1)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    
    """  
    out, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask)
    """
    out = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask)
    
    #print('out0 ',out[0].shape) #'str' object has no attribute 'shape'
    #print('out ',out)
    #print('pooled ',out[1].shape)
    
    output = self.drop(out[1])
    #output = self.drop(pooled_output)
    return self.out(output)
    #return self.out(out[1])

class_names = set(list(df_val.label.to_numpy()))
print(class_names) #{0, 1, 2, 3, 4}
class_names = list(class_names)
print(class_names) #[0, 1, 2, 3, 4]
print(type(class_names))

model = SBKSClassifier(len(class_names))
model = model.to(device)
#print(model)
input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)
print(input_ids.shape) # batch size x seq length
print(attention_mask.shape) # batch size x seq length

EPOCHS = 10
print('epochs ',EPOCHS)
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

def train_epoch(model,data_loader,loss_fn,optimizer,device,scheduler,n_examples):
    
  model = model.train()
  losses = []
  correct_predictions = 0
  count = 0
  for d in data_loader:
      
    #if count > 5:
    #    break
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask)
    
       
    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)
    correct_predictions += torch.sum(preds == targets)
    
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    
    #losses.append(float(loss.item()))
    optimizer.zero_grad()
    
    count+=1
    
  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  count=0
  with torch.no_grad():
    for d in data_loader:
        
      #if count > 5:
      #    break
        
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
      count+=1
  return correct_predictions.double() / n_examples, np.mean(losses)


history = defaultdict(list)

best_accuracy = 0
t0=time.time()
for epoch in range(EPOCHS):
  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)
  t1=time.time()
  train_acc, train_loss = train_epoch(
    model,
    train_data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    len(df_train)
  )
  print(f'Train loss {train_loss} accuracy {train_acc}')
  val_acc, val_loss = eval_model(
    model,
    val_data_loader,
    loss_fn,
    device,
    len(df_val)
  )
  print(f'Val   loss {val_loss} accuracy {val_acc}')
  print()
  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)
  if val_acc > best_accuracy:
    torch.save(model.state_dict(), 'sh4n_pmb_reg_pt1_drp.bin')
    best_accuracy = val_acc
  t2 = time.time()
  print('epoch time: %.2f min' %((t2-t1)/60))
  print('total time: %.2f min' %((t2-t0)/60))
  print()

print(history)


pdhist = pd.DataFrame(history)
savef = '.../hist_4n_pmb_reg_drp.csv'
pdhist.to_csv(savef, index=False)


savefig = '.../hist_4n_pmb_reg_drp.png'
fig = plt.figure()
plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')
plt.title('Training history: PubMedBERT dropout percentage = .1')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1])
#plt.plot(range(10))
fig.savefig(savefig, dpi=fig.dpi)