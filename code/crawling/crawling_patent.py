# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:22:04 2022

@author: yebin
"""

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import requests
from datetime import datetime
import copy
import itertools
from collections import Counter

pat_num = pd.read_csv('../data/raw/gp-20102019.csv', encoding='utf8')
pat_nums = [i.split('-')[1] for i in pat_num['id']]

#%% uspto에서 특허 정보 수집
url_post = 'https://api.patentsview.org/patents/query'
post_len = 10000
#crawling elements
condition = '&f=["patent_number", "patent_date", "patent_type", "patent_title", "patent_abstract", "citedby_patent_number", "citedby_patent_date"]'

#%% crawling
pt_dict_list=[]
st = time.time()
for n, i in enumerate(tqdm(pat_nums)):
    q1 = '?q={"patent_number":'
    q2 = '}'
    q = q1 + '"' + i + '"' + q2
    
    t_data = q + condition
    t_post = requests.get(url_post + t_data)
    
    while '500' in str(t_post):
        t_post = requests.get(url_post + t_data)
    try:
        t_json = t_post.json()
    except:
        continue
        
    try:
        pt_dict_list.extend(t_json['patents'])
    except TypeError:
        pass
#%%filtering by patent type (we need utility type patent only!!)
util_patent = [i for i in pt_dict_list if i['patent_type'] == 'utility']
print("특허 수: ", len(util_patent))

first_patent_df = pd.DataFrame.from_dict(util_patent)

#%% FC 승인된지 5년이내 FC 수
def AAFC(util_patent):
    all_fp_list = []

    for p in tqdm(util_patent):
        p_year = int(p['patent_date'].split('-')[0]) # 타겟 특허의 승인연도
        fp_list = []

        for fp in p['citedby_patents']:
            try:
                fp_year = int(fp['citedby_patent_date'].split('-')[0]) # 타겟특허의 fc들의 승인연도
                # 타겟특허가 승인된지 5년 이내 forward citation 횟수
                if p_year+5 >= fp_year:
                    fp_list.append(fp['citedby_patent_number'])
                else:
                    pass
            # 오류 예외처리
            except AttributeError:
                pass
            except TypeError:
                pass

        all_fp_list.append(fp_list)
    # all_fp_list는 이중리스트의 형태이므로 각 원소리스트의 길이를 구하면서 flatten 진행
    all_fp_list_length = [len(p) for p in all_fp_list]
    return all_fp_list_length

aafc = AAFC(util_patent)

#%%
patent = first_patent_df.assign(AAFC = aafc)
patent2 = patent.drop_duplicates(['patent_abstract'])

patent2.to_excel('../data/raw/patent_1019.xlsx', encoding='utf8', index=False)







































