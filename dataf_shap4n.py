# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:17:01 2021
@author: ddabl
"""
# ==============================================================================
# utilities to create train, val, test files
# ==============================================================================
# load modules

import json
import time
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

##############################################################################

t = ['t1','t2','t3','t4','t5']
s = ['sb1','sb2','sb3','sb4','sb5']
e = ['elsi1','elsi2','elsi3','elsi4','elsi5']

path = 'C:/Users/ddabl/OneDrive/Documents/Networks/SBKS/data_0523/data_shap4n/'

ttot=0
stot=0
etot=0

for i in range(5):
    tf = path + t[i] + '.csv'
    sf = path + s[i] + '.csv'
    ef = path + e[i] + '.csv'
    dt = pd.read_csv(tf)
    ds = pd.read_csv(sf)
    de = pd.read_csv(ef)
    print('topic %1d: %1d' %(i+1,len(dt)))
    print('syn bio only %1d: %1d' %(i+1,len(ds)))
    print('ELSI only %1d: %1d' %(i+1,len(de)))
    ttot +=len(dt)
    stot +=len(ds)
    etot +=len(de)
    print()
    
print('total docs = %1d; total syn bio only docs = %1d; total total ELSI only docs = %1d' %(ttot,stot,etot))

##############################################################################
"""convert key word frequency hits data with min max scaler"""

featpath = 'C:/Users/ddabl/OneDrive/Documents/Networks/SBKS/data_0523/data_shap4n/freq_all.csv'

#featpath = 'C:/Users/ddabl/OneDrive/Documents/Networks/SBKS/data/freq_all_2nd_1.csv'
ffile = pd.read_csv(featpath)
cols = list(ffile.columns)
cols
cols[0] = 'doc'
cols
ffile1 = ffile.to_numpy()
print(ffile1.shape) #(10617, 89)      (10617, 67)

shape2 = ffile1.shape[1]
shape2
#find number of rows with values / abstracts
count=0
for f in range(len(ffile1)):
#for f in range(25):
    sumr = np.sum(ffile1[f,1:])
    if sumr>0:
        count+=1
print(count)# 10180

ffile_1 = np.zeros((count,shape2))
ffile_1.shape

c=0
for f in range(len(ffile1)):
#for f in range(25):
    sumr = np.sum(ffile1[f,1:])
    if sumr>0:
        ffile_1[c,:]= ffile1[f,:]
        c+=1
        #print(c)# 10180

ffile_1
ffile2 = ffile_1[:,1:]
#print(ffile2.shape) #(10180, 66)


scaler = MinMaxScaler()
scaler.fit(ffile2)
x =scaler.transform(ffile2)
#print(x.shape,x)

col_add = ffile_1[:,0]
#print(col_add.shape, col_add) #(10180,)
col_add = col_add.reshape(len(col_add),1)
#print(col_add.shape, col_add) #(11815,1)


all_data = np.hstack((col_add,x))
#print(all_data.shape,all_data)

df_ad = pd.DataFrame(columns=cols, data=all_data)
df_ad.to_csv('C:/Users/ddabl/OneDrive/Documents/Networks/SBKS/data_0523/data_shap4n/freq_all_ss.csv',
             index=False)
    

##############################################################################
"""create features matrix of topic keyword hits for each ELSI topic"""

e = ['elsi1','elsi2','elsi3','elsi4','elsi5']
epath = 'C:/Users/ddabl/OneDrive/Documents/Networks/SBKS/data_0523/data_shap4n/'
#featpath = 'C:/Users/ddabl/OneDrive/Documents/Networks/SBKS/data_0523/data/freq_all_2nd_ss_1.csv'
featpath = 'C:/Users/ddabl/OneDrive/Documents/Networks/SBKS/data_0523/data_shap4n/freq_all_ss.csv'

ffile = pd.read_csv(featpath)
fcol = ffile.columns
fcol = ffile.columns.tolist()
fcol = fcol[1:]
#print(fcol)
#print(len(fcol))
#print(ffile.head())
ffile.set_index('doc')
    
ffile1 = ffile.to_numpy()
#print(ffile1.shape) #(875, 1)
ffile1 = np.squeeze(ffile1)
ffile1 = list(ffile1)

ffile.head()

ffile['doc'] = ffile['doc'].map(str)

ffile.head()

for ind in ffile.index:
    ffile['doc'][ind] = str(ffile['doc'][ind])

#print(type(ffile['doc'][1]))
ffile.head()

for ind in ffile.index:
        doc = ffile['doc'][ind]
        #print(doc)
#ffile.set_index('doc')
        
#for ind in ffile.index:
#    print(ffile.iloc[ind, 1:].values.tolist()  )  

#for ep in e[0]:
for ep in e:
    pth = epath + ep + '.csv'
    #print(pth)
    efile = pd.read_csv(pth)
    efile = efile.to_numpy()
    #print(efile.shape) #(875, 1)
    efile = np.squeeze(efile)
    efile = list(efile)
    efile = [int(x) for x in efile]
    efiles = [str(a) for a in efile]
    #print('e',type(efiles[0]),efiles[1])
    
    df = pd.DataFrame(index = efiles,columns=fcol)
    #c=0
    for ind in ffile.index:
        #print(ffile.iloc[ind])
        doc = ffile['doc'][ind]
        doc = doc.split(".")[0]
        #print('doc',doc,type(doc))
        #print(ffile[ind])
        #c+=1
        #if c >0:
        #    break
    
        if doc in efiles:
            x =ffile.iloc[ind, 1:].values.tolist()  
            #print(doc)
            count=0
            for i in fcol:
                df.loc[doc,i] = x[count] #ffile.loc[doc,i]
                count+=1
    #print(df.head() )
    
    savepth = epath + ep + '_feat_ss.csv'
    df.to_csv(savepth)#, index=False)
    
    

###########################################################################
"""create csv and json files with key data for model pipeline"""

#lists of docs per topic in csv files
t = ['t1','t2','t3','t4','t5']
s = ['sb1','sb2','sb3','sb4','sb5']
e = ['elsi1','elsi2','elsi3','elsi4','elsi5']

cvpth = 'C:/Users/ddabl/OneDrive/Documents/Networks/SBKS/data_0523/data_shap4n/'
jpth_e = 'C:/Users/ddabl/OneDrive/Documents/Networks/SBKS/data_0523/json_el_0608/'
jpth_s = 'C:/Users/ddabl/OneDrive/Documents/Networks/SBKS/data_0523/json_sb4_n/'
#jsonf = os.listdir(jpth)
#len(jsonf) #12611 which is more than 11,556 

#featpth = 'C:/Users/ddabl/OneDrive/Documents/Networks/SBKS/data/freq_all_2nd_ss_1.csv'
featpth = 'C:/Users/ddabl/OneDrive/Documents/Networks/SBKS/data_0523/data_shap4n/freq_all_ss.csv'
dfeat = pd.read_csv(featpth)
docnum = dfeat.iloc[:, 0].values.tolist() 
featlst = dfeat.iloc[:, 1:].values.tolist() 
#print('doc ',docnum)
#print('feat ',featlst)
#print(type(docnum))

#dict to be created
df_final = pd.DataFrame(columns=['doc','pmid', 'title','abstr',
        'feats','label','type','folder','body_list'])

count =0
for x in range(5):
    
    sf = cvpth + s[x] + '.csv'
    ef = cvpth + e[x] + '.csv'
    
    ds = pd.read_csv(sf)
    de = pd.read_csv(ef)
    
    #x =ffile.iloc[ind, 1:].values.tolist() 
    
    for i in range(2):
        
        if i == 0:
            folder = ds
            typed = 0 #syn bio
            file = s[x]
            #savef = cvpth + s[i]
            jpth = jpth_s
        else:
            folder = de
            typed = 1 #elsi
            file = e[x]
            #savef = cvpth + e[i]
            jpth = jpth_e
        
        for n in range(len(folder)):
            #print(int(folder.iloc[n,0]))
            ind = docnum.index(int(folder.iloc[n,0]))
            #print(ind)
            fold = featlst[ind]
            #print(fold)
            
            docpth = jpth + str(int(folder.iloc[n,0])) + '.json'
            #print(docpth)
        
            with open(docpth, 'r') as fp:
                jf = json.load(fp)
               
                df_final = df_final.append({
                    'doc':      int(folder.iloc[n,0]),
                    'pmid':     jf['article_id_pmid'],
                    'title':    jf['article_title'],
                    'abstr':    jf['abstract'],
                    'feats':    fold,
                    'label':    x+1,
                    'type':     typed,
                    'folder':   file,
                    #'body_list':jf['body_list']
            },
            ignore_index=True)
            count+=1
            if count % 100 ==0:
                print(count)
                
saved = cvpth + 'all_data.csv'
df_final.to_csv(saved,index=False)
                
# Code for the actual train/test/validation split seems to be missing, but is apparently a 70/15/15 split              
