# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:17:01 2021
@author: ddabl
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


##############################################################################

#To search any of these databases, we use Bio.Entrez.esearch(). 

Entrez.email = 'XXX'
#final
handle = Entrez.esearch(db="pmc", term="((“synthetic biology”[KWD] OR \
    “synthetic dna”[KWD] OR “synthetic genome”[KWD] OR \
    “synthetic nucleotide”[KWD] OR “synthetic promoter”[KWD] OR \
    “synthetic gene cluster”[KWD]) NOT (“photosynthe*”[KWD]))  \
    OR ( (“synthetic mammalian gene*”[KWD] AND “mammalian cell”[KWD]) \
    NOT (“photosynthe*”[KWD])) \
    OR (“synthetic gene”[KWD] NOT (“synthetic gener*”[KWD] OR \
    “photosynthe*”[KWD])) \
    OR ( (“artificial gene network”[KWD] OR (“artificial gene circuit*”[KWD] \
    AND “biological system”[KWD])) NOT “gener*”[KWD] ) \
    OR ( (“artificial cell”[KWD]) \
    NOT  (“cell* telephone”[KWD] OR “cell* phone”[KWD] OR “cell* culture”[KWD] \
    OR “logic cell*”[KWD] OR “fuel cell*”[KWD] OR “battery cell*”[KWD] OR \
    “load-cell*”[KWD] OR “geo-synthetic cell*”[KWD] OR “memory cell*”[KWD] \
    OR “cellular network”[KWD] OR “ram cell*”[KWD] OR “rom cell*”[KWD] \
    OR “maximum cell*”[KWD] OR “electrochemical cell*”[KWD] OR \
    “solar cell*”[KWD])) \
    OR ( (“synthetic cell”[KWD]) \
    NOT (“cell* telephone”[KWD] OR “cell* phone”[KWD] OR “cell* culture”[KWD] \
    OR “logic cell*”[KWD] OR “fuel cell*”[KWD] OR “battery cell*”[KWD] \
    OR “load-cell*”[KWD] OR “geo-synthetic cell*”[KWD] \
    OR “memory cell*”[KWD] OR “cellular network”[KWD] OR “ram cell*”[KWD] \
    OR “rom cell*”[KWD] \
    OR “maximum cell*”[KWD] OR “electrochemical cell*”[KWD] \
    OR “solar cell*”[KWD] OR “photosynthe*”[KWD]  )  ) \
    OR ( (“artificial nucleic acid*”[KWD] OR “artificial *nucleotide”[KWD])) \
    OR ( (“bio brick”[KWD] OR “biobrick”[KWD] OR “bio-brick”[KWD])) OR \
    \
    ((“synthetic biology”[ABST] OR \
    “synthetic dna”[ABST] OR “synthetic genome”[ABST] OR \
    “synthetic *nucleotide”[ABST] OR “synthetic promoter”[ABST] OR \
    “synthetic gene* cluster”[ABST]) NOT (“photosynthe*”[ABST]))  \
    OR ( (“synthetic mammalian gene*”[ABST] AND “mammalian cell”[ABST]) \
    NOT (“photosynthe*”[ABST])) \
    OR (“synthetic gene”[ABST] NOT (“synthetic gener*”[ABST] OR \
    “photosynthe*”[ABST])) \
    OR ( (“artificial gene network”[ABST] OR (“artificial gene circuit*”[ABST] \
    AND “biological system”[ABST])) NOT “gener*”[ABST] ) \
    OR ( (“artificial cell”[ABST]) \
    NOT  (“cell* telephone”[ABST] OR “cell* phone”[ABST] OR “cell* culture”[ABST] \
    OR “logic cell*”[ABST] OR “fuel cell*”[ABST] OR “battery cell*”[ABST] OR \
    “load-cell*”[ABST] OR “geo-synthetic cell*”[ABST] OR “memory cell*”[ABST] \
    OR “cellular network”[ABST] OR “ram cell*”[ABST] OR “rom cell*”[ABST] \
    OR “maximum cell*”[ABST] OR “electrochemical cell*”[ABST] OR \
    “solar cell*”[ABST])) \
    OR ( (“synthetic cell”[ABST]) \
    NOT (“cell* telephone”[ABST] OR “cell* phone”[ABST] OR “cell* culture”[ABST] \
    OR “logic cell*”[ABST] OR “fuel cell*”[ABST] OR “battery cell*”[ABST] \
    OR “load-cell*”[ABST] OR “geo-synthetic cell*”[ABST] \
    OR “memory cell*”[ABST] OR “cellular network”[ABST] OR “ram cell*”[ABST] \
    OR “rom cell*”[ABST] \
    OR “maximum cell*”[ABST] OR “electrochemical cell*”[ABST] \
    OR “solar cell*”[ABST] OR “photosynthe*”[ABST]  )  ) \
    OR ( (“artificial nucleic acid*”[ABST] OR “artificial *nucleotide”[ABST])) \
    OR ( (“bio brick”[ABST] OR “biobrick”[ABST] OR “bio-brick”[ABST])) \
      AND (ethical[ABST] OR ethics[ABST] OR bioethic*[ABST] OR \
         policy[ABST] OR governance[ABST] OR 'public perception'[ABST] OR\
         bio-risk[ABST] OR biosafety[ABST] or social issues[ABST] OR \
         social impact[ABST] or social implications[ABST] or societal impact[ABST] \
         or societal implications[ABST] or environmental impact[ABST] \
         or environmental issues[ABST])",
        retmax="20000" )


record = Entrez.read(handle)

print(record["Count"]) 

id_list = record["IdList"]
print(len(id_list)) #256


# pubmed config file

FILE_CONFIG = ".../draft_config.json"
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
count=0
for i in id_list: #id list comes from biopython
    xml_file = get_xml(i, params=params)
    xml_list.append(xml_file)
    time.sleep(.2) #.334
    count+=1
    if count % 250 ==0:
        print(count)

print(len(xml_list)) #16765
# save as xml

dirpth = ".../data/xml_el_key/"

count = 0
for i in xml_list:
    doc = id_list[count]
    end = '.xml'
    svpth = dirpth + doc + end

    with open(svpth, 'wb') as file:
        
        file.write(i.content)
    count+=1
    
    print(count)
print(count)



##############################################################################
"""iterate thru xml list, parse the xml files and save output as a json file"""

dirpth = ".../data/xml_el_key/"

xmlf = os.listdir(dirpth)
print(len(xmlf)) #9626

count = 0
#for x in xmlf1:
for x in xmlf:
    #print(x)
    docpth = dirpth + x
    
    dict_out = pp.parse_pubmed_xml(docpth)
        
    #parse full text
    ftxt = pp.parse_pubmed_paragraph(docpth, all_paragraph=True) #list, not dict
    
    f_text = []
    for i in range(len(ftxt)):
        info = {
        "section": ftxt[i]['section'],
        "text":    ftxt[i]['text']}
        f_text.append(info)
    
    """parse references"""
    refs = pp.parse_pubmed_references(docpth) # return list of dictionary
    
    
    ref_list = []
    
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
    
    fname = fsplit[0]
    
    jpath = '.../data/json_el_key/' + fname + '.json'
    
    with open(jpath, 'w') as fp:
        json.dump(save_dict, fp)
    count+=1
    if count % 250 ==0:
        print(count)
    
print(count)
jpck = ".../data/json_el_key/"

jres = os.listdir(jpck)
print(len(jres)) #71

###############################################################################

#################################################
jpth = ".../data/json_el_key/"

jsonf = os.listdir(jpth)


#for x in j1:
for x in jsonf:
    print(x)
    docpth = jpth + x
    
    with open(docpth, 'r') as fp:
        jf = json.load(fp)
        print(jf['abstract'])
        print()











