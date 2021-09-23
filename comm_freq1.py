# -*- coding: utf-8 -*-
"""
@author: ddabl
"""
"""
purpose:  to build a word frequency list for the abstracts in the 5 topics

"""

import string
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import FreqDist
from nltk.stem import PorterStemmer
import pandas as pd
import os
import numpy as np



t1=['network','control','circuit','promoter', 'cellular','transcription',
 'mechanism','signal']

t2=['pathway', 'strain','metabol', 'plant','enzyme', 'biosynthe','coli',
 'compound','yeast','chemical']
 
t3=['technology','genome','tool','design','genetic','organism','process']
   
t4=['dna','rna','sequence','structure','binding','site','acid', 'domain',
 'assembly','peptide','virus','interaction',
 'amino','membrane']

t5=['patient', 'disease', 'treatment', 'cancer', 'clinical']



a = '.../jcomb'
print(len(os.listdir(a))) #10728 of sb w/o elsi; 12611 w/ elsi
an = os.listdir(a)
dir_n = [n.split(".")[0] for n in an]
dir_nf = [float(n) for n in dir_n]
print(dir_nf[0])


tlist = [t1,t2,t3,t4,t5]
tls = t1+t2+t3+t4+t5
print(len(tls))
x = list(set(tls))
print(len(x)) #44 terms

import collections
print([item for item, count in collections.Counter(tls).items() if count > 1])
#print(collections.Counter(t1))
#['expression', 'protein', 'engineering', 'acid']
#no overlap
    
#stem topic lists so similar format to abstract searches below

porter_stemmer=PorterStemmer()

for t in tlist:
    #filtered for stop words, punctuation, numbers
    # Python program to convert a list
    # to string using list comprehension

    text = ' '.join([str(elem) for elem in t])
  
    lower_case = text.lower()
    words = nltk.tokenize.word_tokenize(lower_case)
        
    # remove punctuation from each word 
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    #print('words1 ',words)
    #stem / lemmitze the words
    stemmed_words=[porter_stemmer.stem(word=word) for word in words]
    #print('stemmed ',stemmed_words)
    if t == t1:
        s1=stemmed_words
    if t == t2:
        s2=stemmed_words
    if t == t3:
        s3=stemmed_words
    if t == t4:
        s4=stemmed_words
    if t == t5:
        s5=stemmed_words

lens1 = len(s1)#8
lens2 = len(s2) #10
lens3 = len(s3) #7
lens4 = len(s4) #14
lens5 = len(s5) #5

print(s1,lens1)
print(s2,lens2)
print(s3,lens3)
print(s4,lens4)
print(s5,lens5)

scomb = s1+s2+s3+s4+s5
print(scomb,len(scomb)) #44
lencomb = len(scomb)
############################################################################

#directory where training data saved 
directory = '.../jcomb/'

#jlst = ['j_add','j1','j2','j3','j4','j5','je_add','je1','je2','je3','je4','je5']

dir_list = []

count=0
for n in an:
    file = directory + n
    temp_file = pd.read_json(file, lines=True)
    df_temp = pd.DataFrame(temp_file) 

    #iterate through dataframe 
    for row, col in df_temp.iterrows():
        text = str(col['abstract']) #to avoid attribute error re:nan
        if text != '':
            count+=1
            dir_list.append(n)
            
print('number of docs with abstract: ',count) #11,816
print(dir_list[0])   # 1072806.json

dflist = pd.DataFrame(dir_list)
dflist.to_csv('.../abstracts.csv',index=False)

"""
for j in jlst:
    diry = directory + j + '/'
    dlst = os.listdir(diry)
    dir_list+= dlst
    
print(len(dir_list)) #12611

print(dir_list[0]) #1131869.json
"""
dir_name = [n.split(".")[0] for n in dir_list]

#print(dir_name)
#[..., '8005139', '8009270', '8009411', '8010332', '8013358', '8023661', '8024369']

directory_save = '.../data/'

#iterate through the computer generated(cg), human generated fake news (hg),
#trusted datasets, identify most commonly used words and save them to a file
#identify frequency of topic words per document

#un = ['the','and','or']

#indic = ['doc1','doc2']

#pandas file wtih each document as an index and each column as a word in the
#topic so that we can count the frequency of each word in a topic
#df = pd.DataFrame(index = dir_name, columns=scomb)

#pandas file that tracks the sum of the word hits for each topic so that we
#can compare relative overlap of topics
#cols = ['t1','t2','t3','t4','t5']
#dfn = pd.DataFrame(index = dir_name, columns=cols)
#df.head()

#determine overlapping terms/topics
#so = s2+s3+s4+s5
#dfo = pd.DataFrame(index = dir_name, columns=so)

#lst = ['the','and', 'the','and', 'the','and']

#fd = nltk.FreqDist()
#for word, frequency in fd.items():
#    print(word, frequency)

#############################################################################

ablist = pd.read_csv('.../abstracts.csv')
ab = ablist.to_numpy()
print(ab.shape) #(875, 1)
ab = np.squeeze(ab)
ab = list(ab)
print(len(ab))
print(ab[0]) # (11816, 1)  11816 1072806.json


#############################

t1_lst=[]
t2_lst=[]
t3_lst=[]
t4_lst=[]
t5_lst=[]

#pandas file wtih each document as an index and each column as a word in the
#topic so that we can count the frequency of each word in a topic
df = pd.DataFrame(index = dir_name, columns=scomb)

#pandas file that tracks the sum of the word hits for each topic so that we
#can compare relative overlap of topics
ncols = ['t1','t2','t3','t4','t5']
dfn = pd.DataFrame(index = dir_name, columns=ncols)


count=0
for d in dir_list:
#for d in dir_list[:5]:
    #read file  
    file_address = directory + d #+ '/train_70.jsonl'
    #file_address = directory
    #print()
    #print(f'starting: {file_address}')
    
    #dir_name = [n.split(".")[0] for n in dir_list]
    doc_name = d.split(".")[0] 
    
    #save as dataframe
    temp_file = pd.read_json(file_address, lines=True)
    df_temp = pd.DataFrame(temp_file) 
   
    #num_docs = 0
    count+=1
    if count % 250 ==0:
        print(count)
    
    #create empty word list that will be appended with words from the docs
    #which will then be used to determine word frequency
    #word_corpus = []
    
    #stem the words
    porter_stemmer=PorterStemmer()
    
    #fd = nltk.FreqDist()
    #for word, frequency in fd.items():
    #    print(word, frequency)

    #iterate through dataframe 
    for row, col in df_temp.iterrows():
        text = str(col['abstract']) #to avoid attribute error re:nan
        #print(text)
        
        #most common words, filtered for stop words, punctuation, numbers
        lower_case = text.lower()
        words = nltk.tokenize.word_tokenize(lower_case)
        
        #track number of docs - should be equal to 70k
        #num_docs+=1
        
        # remove punctuation from each word 
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in words]
        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        #print('words1 ',words)
        #stem / lemmitze the words
        stemmed_words=[porter_stemmer.stem(word=word) for word in words]
        #print('stemmed ',stemmed_words)
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in stemmed_words if not w in stop_words]
        #print('words2 ',words)
        #append word corpus with cleaned words contained in each document
        #for w in words:
        #    word_corpus.append(w)
        #print('word corpus ',word_corpus)
        #if num_docs % 100 == 0:
        #    print(f'num docs: {num_docs}')
        #print(f'word corpus: {word_corpus}','\n')
        #calculate frequency of words in the corpus for the entire list of 
        #docs in the training set
    
    #fd = nltk.FreqDist(word_corpus)
    fd = nltk.FreqDist(words)
        #fd.plot()
    #num = len(word_corpus)
    #fdmc = fd.most_common(300)

    #print(f'fdmc: {fdmc}')
    
    #print(f'total number of documents: {num_docs}')
    #df_fdmc = pd.DataFrame(fdmc)
    
    #fd = nltk.FreqDist(lst)
    #for word, frequency in fd.items():
    #    print(word, frequency)

    fd_word = []
    freq = []
    for w, f in fd.items():
        fd_word.append(w)
        freq.append(f)
    
    #print(word)
    #print(freq)
    
    nfile = np.zeros((lencomb))
    #nfile.shape #(44,)

    for i in scomb:
        i_ind = scomb.index(i)
        if i in fd_word:
            ind = fd_word.index(i)
            df.loc[doc_name,i] = freq[ind]
            nfile[i_ind]=freq[ind]
        else:
            df.loc[doc_name,i] = 0
    
    #print(nfile)
    
    n2 = lens1 + lens2
    n3 = n2+ lens3
    n4 = n3 + lens4
    n5 = n4 + lens5
    
    tn1 = np.sum(nfile[:lens1])
    tn2 = np.sum(nfile[lens1:n2])
    tn3 = np.sum(nfile[n2:n3])
    tn4 = np.sum(nfile[n3:n4])
    tn5 = np.sum(nfile[n4:])
    
    #print(tn1,tn2,tn3,tn4,tn5)
    
    tnum = [tn1,tn2,tn3,tn4,tn5]
    tnum1 = np.array(tnum)
    tnum_sum = np.sum(tnum1)
    
    if tnum_sum > 0:
        
        dfn.loc[doc_name,'t1'] = tn1
        dfn.loc[doc_name,'t2'] = tn2
        dfn.loc[doc_name,'t3'] = tn3
        dfn.loc[doc_name,'t4'] = tn4
        dfn.loc[doc_name,'t5'] = tn5
    
        amax = np.argmax(tnum1)
        #print(amax)
    
        if amax == 0:
            t1_lst.append(d)
        if amax == 1:
            t2_lst.append(d)
        if amax == 2:
            t3_lst.append(d)
        if amax == 3:
            t4_lst.append(d)
        if amax == 4:
            t5_lst.append(d)
    
    
print('num docs in topic 1: ',len(t1_lst))
print('num docs in topic 2: ',len(t2_lst))
print('num docs in topic 3: ',len(t3_lst))
print('num docs in topic 4: ',len(t4_lst))
print('num docs in topic 5: ',len(t5_lst))
#print(t1_lst) #['1172234.json', '1172236.json']
print(len(t1_lst) + len(t2_lst) + len(t3_lst) + len(t4_lst) + len(t5_lst))
#11816 lst pass vs 11556 2nd pass which removed 260 of docs that no include
# key topic words

# shows the frequency of keyword hits per document per word
df.to_csv('.../freq_all.csv')#,index=False)
# shows the sum of keyword hits for each topic (t1, t2, etc)
dfn.to_csv('.../nump.csv')#,index=False)

# sample lists of documents for classes 1 through 5 (topics 1 through 5) respectively
df1 = pd.DataFrame(t1_lst)
df1.head()
df1.to_csv('.../t1.csv',
           index=False)
df2 = pd.DataFrame(t2_lst)
df2.to_csv('.../t2.csv',
           index=False)
df3 = pd.DataFrame(t3_lst)
df3.to_csv('.../t3.csv',
           index=False)
df4 = pd.DataFrame(t4_lst)
df4.to_csv('.../t4.csv',
           index=False)
df5 = pd.DataFrame(t5_lst)
df5.to_csv('.../t5.csv',
           index=False)

###########################################
#determine elsi articles

elsi = pd.read_csv('.../elsi.csv')
nel = elsi.to_numpy()
print(nel.shape) #(875, 1)
nel = np.squeeze(nel)
unel = list(nel)
print(len(unel))
print(len(unel)) #875
print(type(nel[0])) #<class 'numpy.float64'>

t1f = [float(n.split(".")[0]) for n in t1_lst]
t1f
inter = list(set(unel) & set(t1f))
print(len(inter)) #111
dft1f = pd.DataFrame(inter)
# elsi1.csv: sample list of documents for class 1 that are only ELSI
dft1f.to_csv('.../elsi1.csv',
           index=False)

t2f = [float(n.split(".")[0]) for n in t2_lst]
inter = list(set(unel) & set(t2f))
print(len(inter)) #128
dft2f = pd.DataFrame(inter)
dft2f.to_csv('.../elsi2.csv',
           index=False)

t3f = [float(n.split(".")[0]) for n in t3_lst]
inter = list(set(unel) & set(t3f))
print(len(inter)) #358
dft3f = pd.DataFrame(inter)
dft3f.to_csv('.../elsi3.csv',
           index=False)

t4f = [float(n.split(".")[0]) for n in t4_lst]
inter = list(set(unel) & set(t4f))
print(len(inter)) #90
dft4f = pd.DataFrame(inter)
dft4f.to_csv('.../elsi4.csv',
           index=False)

t5f = [float(n.split(".")[0]) for n in t5_lst]
inter = list(set(unel) & set(t5f))
print(len(inter)) #40
dft5f = pd.DataFrame(inter)
dft5f.to_csv('.../elsi5.csv',
           index=False)

"""
dft1_lst = dft1f.to_numpy()
print(dft1_lst.shape) #(875, 1)
dft1_lst = np.squeeze(dft1_lst)
dft1_lst = list(dft1_lst)
dft1_lst
"""

sb1 = []
for n in t1f:
    if n not in unel:
        sb1.append(n)
print(len(sb1))

sb2 = []
for n in t2f:
    if n not in unel:
        sb2.append(n)
print(len(sb2))

sb3 = []
for n in t3f:
    if n not in unel:
        sb3.append(n)
print(len(sb3))

sb4 = []
for n in t4f:
    if n not in unel:
        sb4.append(n)
print(len(sb4))

sb5 = []
for n in t5f:
    if n not in unel:
        sb5.append(n)
print(len(sb5))

# sb1.csv: sample list of documents for class 1 that are only Synthetic Biology
dsb1 = pd.DataFrame(sb1)
dsb1.to_csv('.../sb1.csv',
           index=False)

dsb2 = pd.DataFrame(sb2)
dsb2.to_csv('.../sb2.csv',
           index=False)

dsb3 = pd.DataFrame(sb3)
dsb3.to_csv('.../sb3.csv',
           index=False)

dsb4 = pd.DataFrame(sb4)
dsb4.to_csv('.../sb4.csv',
           index=False)

dsb5 = pd.DataFrame(sb5)
dsb5.to_csv('.../sb5.csv',
           index=False)

totalel = len(dft1f) + len(dft2f) + len(dft3f) + len(dft4f) + len(dft5f)
totalel #727

totalsb = len(dsb1) + len(dsb2) + len(dsb3) + len(dsb4) + len(dsb5)
totalsb #10829

total = totalel + totalsb
total #11556

#total inter = 727 elsi out of 875 - 148 not related to sb 5 topic classes


"""initial pass - there were 260 docs with no hits, so remove in 2nd pass
num docs in topic 1:  2991
num docs in topic 2:  2937
num docs in topic 3:  2386
num docs in topic 4:  2905
num docs in topic 5:  597

2nd pass with docs with no topic word hits removed
num docs in topic 1:  2731
num docs in topic 2:  2937
num docs in topic 3:  2386
num docs in topic 4:  2905
num docs in topic 5:  597
11556

length of sb after elsi removed
1 - 2620
2 - 2809
3 - 2028
4 - 2815
5 - 557



"""





##########################################################################
##########################################################################

 

##############################################################################
"""test to ensure that there is 874/875 overlap betw new elsi and old elsi"""
a = '.../jcomb'
print(len(os.listdir(a))) #10728 of sb w/o elsi
an = os.listdir(a)
dir_n = [n.split(".")[0] for n in an]
dir_n = [float(n) for n in dir_n]
print(dir_n[0])

elsi = pd.read_csv('.../elsi.csv')
nel = elsi.to_numpy()
print(nel.shape) #(875, 1)
nel = np.squeeze(nel)
unel = list(nel)
print(len(unel))
print(len(unel)) #875
print(type(nel[0])) #<class 'numpy.float64'>

inter = list(set(unel) & set(dir_n))
print(len(inter)) #0

#directory where training data saved 
directory = '.../data3/'

jelst = ['je_add','je1','je2','je3','je4','je5']

dire = []

for j in jelst:
    diry = directory + j + '/'
    dlst = os.listdir(diry)
    dire+= dlst
    
print(len(dire)) #1883

print(dire[0]) #1131869.json

dire1 = [n.split(".")[0] for n in dire]
dire1 = [float(n) for n in dire1]


inter = list(set(unel) & set(dire1))
print(len(inter)) #874

























