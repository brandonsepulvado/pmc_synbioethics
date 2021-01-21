# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 17:41:13 2020
@author: ddabl
"""
import os
import json
import xmltodict

xmlfile = ".../shapira.xml"

jsonfile = ".../big_jsonf.json"

with open(xmlfile) as xml_file:
     
    data_dict = xmltodict.parse(xml_file.read())
    xml_file.close()
     
    # generate the object using json.dumps()  
    # corresponding to json data   
    json_data = json.dumps(data_dict)
     
    # Write the json data to output  
    # json file
    with open(jsonfile, "w") as json_file:
        json_file.write(json_data)
        json_file.close()

# set path to json version of the file
#FILE_PATH = jsonfile #Path(r'path_to_json')

# load file from json
with open(jsonfile, 'r') as f:
    #test = json.load(f)
    articles = json.load(f)
    f.close()

print(type(articles)) #<class 'dict'>

#n = list(articles['pmc-articleset']['article']['body'])

#print(n[0]) #['pmc-articleset']['article'][0]

########################################################
for i in range(10):
    print(articles['pmc-articleset']['article'][i]['front'] \
            ['article-meta']['title-group']['article-title'])
    print()

numart =len(articles['pmc-articleset']['article']) 
print(numart) #7089
"""
articles['pmc-articleset']['article'][7057]['front'] \
            ['article-meta']['article-id'][1]

len1 = len(articles['pmc-articleset']['article'][7057]['front'] \
            ['article-meta']['article-id'])
print(len1)
"""

count=0
"""all article ids and ref ids are doi"""

for i in range(7057,numart):
    print('i ',i)
    print(articles['pmc-articleset']['article'][i]['front'] \
            ['article-meta']['title-group']['article-title'])
    
    extractfile = r'.../shapira/files/doc_' + \
        str(i) + '.json'
    
    len1 = len(articles['pmc-articleset']['article'][i]['front'] \
            ['article-meta']['article-id'])
    try:
        for j in range(len1):
            dict1 = articles['pmc-articleset']['article'][i]['front'] \
            ['article-meta']['article-id'][j]
            #print(dict1)
            if dict1['@pub-id-type'] =='doi':
                article_id = (dict1['#text'])
    except:
        article_id = 'None'
    try:
        num_auth = len(articles['pmc-articleset']['article'][i]['front'] \
            ['article-meta']['contrib-group']['contrib'])
        auth_last = []
        auth_first = []
        for n in range(num_auth):
            auth_last.append(articles['pmc-articleset']['article'][i] \
                ['front']['article-meta']['contrib-group']['contrib'][n] \
                ['name']['surname'])
            auth_first.append(articles['pmc-articleset']['article'][i] \
                ['front']['article-meta'] \
                ['contrib-group']['contrib'][n]['name']['given-names'])
          
    except:
        try:
            #print('except contrib')
            num_auth = len(articles['pmc-articleset']['article'][i]['front'] \
                           ['article-meta']['contrib-group'][0]['contrib'])
            #print(num_auth)
            auth_last = []
            auth_first = []
            for k in range(num_auth):
                #print(k)
                #print(articles['pmc-articleset']['article'][i]['front']['article-meta'] \
                #    ['contrib-group'][0]['contrib'][k]['name']['surname'])
                auth_last.append(articles['pmc-articleset']['article'][i] \
                    ['front']['article-meta'] \
                    ['contrib-group'][0]['contrib'][k]['name']['surname'])
                auth_first.append(articles['pmc-articleset']['article'][i] \
                    ['front']['article-meta'] \
                    ['contrib-group'][0]['contrib'][k]['name']['given-names'])
         
        except:
            num_auth = 0
            auth_last = ['None']
            auth_first = ['None']
    
    try:
        num_refs =len(articles['pmc-articleset']['article'][i]['back']['ref-list'] \
                      ['ref']) 
    except:
        num_refs = 0
    
    ref_titles = []
    ref_id_list = []
    
    if num_refs > 0:
        for m in range(num_refs):
            #print('m ',m)
            ref_id = []
          
            try:
                ref_titles.append(articles['pmc-articleset']['article'][i]['back'] \
                    ['ref-list']['ref'][m]['element-citation']['article-title']) 
            
            except:
                try:
                    ref_titles.append(articles['pmc-articleset']['article'][i]['back'] \
                    ['ref-list']['ref'][m]['mixed-citation']['article-title']) 
                except:
                    ref_titles.append('None')
    
            ref_id_num = []
        
            try:
                #print((articles['pmc-articleset']['article'][i]['back']['ref-list'] \
                # ['ref'][m]['element-citation']['pub-id']['#text'])) #3 authors
                ref_id_num = articles['pmc-articleset']['article'][i]['back']['ref-list'] \
                    ['ref'][m]['element-citation']['pub-id']['#text'] #3 authors
                ref_id_list.append(ref_id_num)
            except:
                try:
                    #print('keyerror with refs i: ',i,' m ',m)
                    #print(articles['pmc-articleset']['article'][i]['back']['ref-list'] \
                    # ['ref'][m]['mixed-citation']['pub-id'][0]['#text'])  #4
                    ref_id_num = articles['pmc-articleset']['article'][i]['back']['ref-list'] \
                      ['ref'][m]['mixed-citation']['pub-id'][0]['#text']  #4
                    ref_id_list.append(ref_id_num)
                
                except:
                    try:
                        # print('keyerror with refs2 i: ',i,' m ',m)
                        #print((articles['pmc-articleset']['article'][i]['back']['ref-list'] \
                        #7   ['ref'][m]['element-citation']['pub-id'][0]['#text'])) #3 authors
      
                        ref_id_num =articles['pmc-articleset']['article'][i]['back']['ref-list'] \
                            ['ref'][m]['element-citation']['pub-id'][0]['#text'] #3 authors
                        ref_id_list.append(ref_id_num)
                    except:
                        #print('keyerror with refs NONE i: ',i,' m ',m)
                        ref_id_list.append('None')
    
    try:
    #if num_refs > 0:
        body = articles['pmc-articleset']['article'][i]['body']
    except:            
    #else:
        body = 'None'
    try:
        pub_year = articles['pmc-articleset']['article'][i]['front'] \
                ['article-meta']['pub-date'][0]['year']  
    except:
        pub_year = 'None'
    
    with open(extractfile, 'w') as outfile:     
        data = {'title':articles['pmc-articleset']['article'][i]['front'] \
            ['article-meta']['title-group']['article-title'],
            
            'article_id': article_id,
            
            'abstract':articles['pmc-articleset']['article'][0]['front'] \
            ['article-meta']['abstract'],
            
            #'pub_year': articles['pmc-articleset']['article'][i]['front'] \
            #    ['article-meta']['pub-date'][0]['year'], 
            'pub_year': pub_year, 
            
            'num_auth': num_auth,
            
            'auth_last': auth_last,
            
            'auth_first': auth_first,
            
            'num_refs': num_refs,
            
            'ref_titles': ref_titles,
            
            'ref_ids': ref_id_list,
            
            'body': body,
                
            } 
            
        json.dump(data,outfile)
        outfile.close()

#same as extract file
dir = '.../shapira/files/' #same as extract file

ids = os.listdir(dir)
#print(ids)
json_fps = [os.path.join(dir, image_id) for image_id in ids]
#print(json_fps)

count=0
for i in json_fps[:1]:
    #print(i)
    print()
    with open(i, 'r') as f:
        #test = json.load(f)
        article = json.load(f)
        print(type(article['body'])) #<class 'dict'>
        print(article['body'])
        #if article['body'] == 'None':
        #    count+=1
        #print()
        #count+=1
        f.close()
print(count)
# num with no body = 89



















