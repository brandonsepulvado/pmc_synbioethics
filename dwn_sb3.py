# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:17:01 2021
"""
# ==============================================================================
# utilities to download pubmed central full text files
# ==============================================================================
# load modules
import Bio
print(Bio.__version__) #1.78
from Bio import Entrez
import requests
import json
import time
import os
import pandas as pd
from requests.exceptions import HTTPError
import pubmed_parser as pp

#To search any of these databases, we use Bio.Entrez.esearch(). For example,
# let’s search in PubMed for publications that include Biopython in their title:
Entrez.email = 'name@norc.org'
handle = Entrez.esearch(db="pmc", term="synthetic biology[journal]", retmax="100" )
record = Entrez.read(handle)

print(record["Count"])
#71

#print(record["IdList"])
id_list = record["IdList"]
print(id_list)
print(id_list[0])


# pubmed config file
FILE_CONFIG = '.../draft_config.json'

# load pubmed config file
with open(FILE_CONFIG) as f:
    config = json.load(f)

# set parameters to put in pmc header
params = {
    "tool": config["tool"],
    "email": config["email"],
    "api_key": config["pubmed_api_key"]}

print(params)

# function that takes an id and returns pmc nxml response
def get_xml(id, params):
    URL = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={id}"
    try:
        response = requests.get(URL, params=params)
        # error structure from : https://realpython.com/python-requests/
        # If the response was successful, no Exception will be raised
        response.raise_for_status()
        return response
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')  # Python 3.6
    except Exception as err:
        print(f'Other error occurred: {err}')  # Python 3.6

# make calls for list of ids
xml_list = []
for i in id_list: #id list comes from biopython
    xml_file = get_xml(i, params=params)
    xml_list.append(xml_file)
    time.sleep(.334)

print(len(xml_list)) #71
print(xml_list)
#[<Response [200]>, <Response [200]>, <Response [200]>, <Response [200]>,
# <Response [200]>, <Response [200]>, <Response [200]>, <Response [200]>,....
# test it out
#test = get_xml(id="7772474", params=params)

# save as xml
dirpth = '.../pmc/xmls/'

count = 0
for i in xml_list:
    doc = id_list[count]
    end = '.xml'
    svpth = dirpth + doc + end

    with open(svpth, 'wb') as file:
        file.write(i.content)
    count+=1

##############################################################################
"""iterate throuth xml list, parse the xml files and save output as a json file"""

dirpth = '.../xmls/'
xmlf = os.listdir(dirpth)
print(len(xmlf),xmlf) #71
#['7121243.xml', '7445753.xml', '7445754.xml', '7445755.xml', '7445756.xml',
# '7445757.xml', '7445758.xml', '7445759.xml', '7445760.xml',...

#test
xmlf1 = ['7121243.xml']
"""
fsplit = xmlf[0].split('.')
fname = fsplit[0]
print(fname)
"""

#for x in xmlf1:
for x in xmlf:
    print(x)
    docpth = dirpth + x
    
    dict_out = pp.parse_pubmed_xml(docpth)
    #print(dict_out.keys())
    """dict_keys(['full_title', 'abstract', 'journal', 'pmid', 'pmc', 'doi',
        'publisher_id', 'author_list', 'affiliation_list', 'publication_year',
        'publication_date', 'subjects', 'coi_statement'])"""
    
    #parse full text
    ftxt = pp.parse_pubmed_paragraph(docpth, all_paragraph=True) #list, not dict
    
    f_text = []
    for i in range(len(ftxt)):
        info = {
        "section": ftxt[i]['section'],
        "text":    ftxt[i]['text']}
        f_text.append(info)
    
    """parse refere
    nces"""
    refs = pp.parse_pubmed_references(docpth) # return list of dictionary
    #print(len(refs)) #34 with true and 18 with false
    #print(refs[0]) #<class 'dict'>
    
    ref_list = []
    print(refs)
    #print(len(refs))
    print()
    if refs == None:
        ref_list = None
    else:
        for i in range(len(refs)):
            #print(i)
            ref_dict = {
              'ref_id':     refs[i]['ref_id'],
              'pmid_cited': refs[i]['pmid_cited'],
              'doi_cited':  refs[i]['doi_cited'],
              'ref_title':  refs[i]['article_title'],
              'ref_auth':   refs[i]['name'],
              'ref_year':   refs[i]['year'],
              'ref_jrnal':  refs[i]['journal']          
                }
            ref_list.append(ref_dict)
       
    save_dict = {
        'article_title':    dict_out['full_title'],
        'abstract':         dict_out['abstract'],
        'pub_name':         dict_out['journal'],
        'article_id_pmid':  dict_out['pmid'],
        
        'article_id_pmc':   dict_out['pmc'],
        'article_id_doi':   dict_out['doi'],
        'publisher_id':     dict_out['publisher_id'],
        'author_list':      dict_out['author_list'],
        
        'affiliation_list': dict_out['affiliation_list'],
        'issue_pub_year':   dict_out['publication_year'],
        'issue_pub_date':   dict_out['publication_date'],
        'keywords':         dict_out['subjects'],
        'body_list':        f_text,
        'ref_list':         ref_list
            }
    fsplit = x.split('.')
    #print(fsplit)
    fname = fsplit[0]
    #print(fname)
    jpath = '.../jsons1/' + fname + '.json'
    
    with open(jpath, 'w') as fp:
        json.dump(save_dict, fp)

jpck = '.../jsons1/'
jres = os.listdir(jpck)
print(len(jres)) #71

###############################################################################
"""no response data counts and store"""

jpth = '.../jsons1/'
jsonf = os.listdir(jpth)

df_final = pd.DataFrame(columns=['doc', 'title','abstr',
        'pub_name', 'pmid','pmcid','doi_id',
        'pub_id', 'auth_lst','affil_lst',
        'pub_year','pub_date','keywrd','section',   
        'text', 'refs', 'ref_id', 'ref_count','pmid_cite','doi_cite',  
        'ref_titl',  'ref_auth','ref_year',  'ref_jrnl'])

#test
j1 = ['7121243.json']
print(len(jsonf),jpth+j1[0]) #71

#for x in j1:
for x in jsonf:
    print(x)
    docpth = jpth + x
    
    with open(docpth, 'r') as fp:
        jf = json.load(fp)
        
    at=0
    ab=0
    pn=0
    aipd=0    
    aipc=0
    aid =0
    
    pi=0
    al=0
    af=0
    ipy=0
    ipd=0
    k=0
    fsec=0
    ftext=0
    ref=0
    ref_id=0
    ref_pm=0
    ref_d=0
    ref_t=0
    ref_a=0
    ref_y=0
    ref_j=0
    
    if jf['article_title'] == '' or jf['article_title']==None:
        at+=1  
    if jf['abstract'] == '' or jf['abstract']==None:
        ab+=1
    if jf['pub_name'] == '' or jf['pub_name']==None:
        pn+=1       
    if jf['article_id_pmid'] == '' or jf['article_id_pmid']==None:
        aipd+=1   
    if jf['article_id_pmc'] == '' or jf['article_id_pmc']==None:
        aipc+=1
    if jf['article_id_doi'] == '' or jf['article_id_doi']==None:
        aid+=1          
    if jf['publisher_id'] == '' or jf['publisher_id']==None:
        pi+=1 
    if jf['author_list'] == '' or jf['author_list']==None:
        al+=1
    if jf['affiliation_list'] == '' or jf['affiliation_list']==None:
        af+=1      
    if jf['issue_pub_year'] == '' or jf['issue_pub_year']==None:
        ipy+=1   
    if jf['issue_pub_date'] == '' or jf['issue_pub_date']==None:
        ipd+=1
    if jf['keywords'] == '' or jf['keywords']==None:
        k+=1      
        
    ftxt = jf['body_list']
    for i in range(len(ftxt)):
        if ftxt[i]['section'] == '' or ftxt[i]['section']==None:
            fsec+=1   
        if ftxt[i]['text'] == '' or ftxt[i]['text']==None:
            ftext+=1
        #print(i)
        #print(ftxt[i]['section'])
        #print(ftxt[i]['text'])
        #print()
    
    refs = jf['ref_list']
    ref_count = 0
    if refs == '' or refs==None:
            ref+=1   
    else:  
        ref_count = len(refs)
        for i in range(len(refs)):
            if refs[i]['ref_id'] == '' or refs[i]['ref_id']==None:
                ref_id+=1  
            if refs[i]['pmid_cited'] == '' or refs[i]['pmid_cited']==None:
                ref_pm+=1  
            if refs[i]['doi_cited'] == '' or refs[i]['doi_cited']==None:
                ref_d+=1  
            if refs[i]['ref_title'] == '' or refs[i]['ref_title']==None:
                ref_t+=1  
            if refs[i]['ref_auth'] == '' or refs[i]['ref_auth']==None:
                ref_a+=1  
            if refs[i]['ref_year'] == '' or refs[i]['ref_year']==None:
                ref_y+=1  
            if refs[i]['ref_jrnal'] == '' or refs[i]['ref_jrnal']==None:
                ref_j+=1  
               
    df_final = df_final.append({
            'doc': x, 
            'title': at,
            'abstr': ab,
            'pub_name': pn,
            'pmid': aipd,
            'pmcid': aipc,
            'doi_id': aid,
            'pub_id': pi,
            'auth_lst': al,
            'affil_lst': af,
            'pub_year': ipy,
            'pub_date': ipd,
            'keywrd': k,
            
            'section': fsec,   
            'text': ftext,
            'refs': ref,
            'ref_id': ref_id,
            'ref_count': ref_count,
            'pmid_cite': ref_pm,  
            'doi_cite': ref_d,  
            'ref_titl': ref_t,  
            'ref_auth': ref_a,
            'ref_year': ref_y,  
            'ref_jrnl': ref_j  
            },
            ignore_index=True)
    
df_final.to_csv('.../df_final.csv',
                index=False)












