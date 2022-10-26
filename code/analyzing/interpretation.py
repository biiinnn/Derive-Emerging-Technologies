# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:16:39 2022

@author: yebin
"""

import pandas as pd
import numpy as np

vacant = pd.read_csv("../data/vacant/textvector/vacant_text.csv")

valid = ['vacant2', 'vacant3', 'vacant5', 'vacant7', 'vacant8', 'vacant9', 'vacant10', 'vacant11', 'vacant13', 'vacant15', 'vacant16', 'vacant18', 'vacant22', 'vacant25', 'vacant26', 'vacant28', 'vacant29', 'vacant33']

v_text = []
for i in range(len(vacant)):
    if vacant['vacant_number'][i] in valid:
        v_text.append(vacant['text'][i])

word_dict = []        
for i in v_text:
    key = i.split(' ')
    word_dict.append(key)

word_list = []        
for i in v_text:
    key = i.split(' ')
    word_list.extend(key)
    
    
word_list.sort()

from collections import Counter

cnt = Counter(word_list)

cnt_df = pd.DataFrame.from_dict(cnt, orient='index').reset_index()

cnt_df = cnt_df.rename(columns={'index': 'keyword', 0:'count'})

cnt_df['freq'] = cnt_df['count'] / 18

cnt_df.to_csv("../data/result/cnt_key.csv", index=False)