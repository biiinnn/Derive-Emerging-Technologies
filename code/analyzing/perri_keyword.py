# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:33:59 2022

@author: yebin
"""

#%% library import
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
#%% data
pe1 = pd.read_csv("../data/perri/perri1.csv")
pe2 = pd.read_csv("../data/perri/perri2.csv")
pe3 = pd.read_csv("../data/perri/perri3.csv")
pe4 = pd.read_csv("../data/perri/perri4.csv")
pe5 = pd.read_csv("../data/perri/perri5.csv")
pe6 = pd.read_csv("../data/perri/perri6.csv")
pe7 = pd.read_csv("../data/perri/perri7.csv")
pe8 = pd.read_csv("../data/perri/perri8.csv")
pe9 = pd.read_csv("../data/perri/perri9.csv")
pe10 = pd.read_csv("../data/perri/perri10.csv")

#%% va1
pat_num1 = pe1['patent_number']
aafc1 = pe1['AAFC']
pe1 = pe1.iloc[:,1:-1]
print(aafc1.describe())
sns.histplot(aafc1)
plt.title('Vacant1 Neighbor')
plt.show()

key1 = []
for i in range(len(pe1)):
    for j in range(pe1.shape[1]):
        if pe1.iloc[i][j] != 0:
            key1.append(pe1.columns[j])
            
cnt1 = Counter(key1)
cnt1_df = pd.DataFrame.from_dict(cnt1, orient='index').reset_index()
cnt1_df.columns = ['keyword', 'count']
cnt1_df['perc'] = cnt1_df['count'] / len(pe1)
#cnt1_df['aafc'] = aafc1
cnt1_df.to_csv("../data/vacant/va1.csv", index=False)

#%% va2
pat_num2 = pe2['patent_number']
aafc2 = pe2['AAFC']
pe2 = pe2.iloc[:,1:-1]
print(aafc2.describe())
sns.histplot(aafc2)
plt.title('Vacant2 Neighbor')
plt.show()

key2 = []
for i in range(len(pe2)):
    for j in range(pe2.shape[1]):
        if pe2.iloc[i][j] != 0:
            key2.append(pe2.columns[j])


cnt2 = Counter(key2)
cnt2_df = pd.DataFrame.from_dict(cnt2, orient='index').reset_index()
cnt2_df.columns = ['keyword', 'count']
cnt2_df['perc'] = cnt2_df['count'] / len(pe2)
#cnt2_df['aafc'] = aafc2
cnt2_df.to_csv("../data/vacant/va2.csv", index=False)

#%% va3
pat_num3 = pe3['patent_number']
aafc3 = pe3['AAFC']
pe3 = pe3.iloc[:,1:-1]
print(aafc3.describe())
sns.histplot(aafc3)
plt.title('Vacant3 Neighbor')
plt.show()

key3 = []
for i in range(len(pe3)):
    for j in range(pe3.shape[1]):
        if pe3.iloc[i][j] != 0:
            key3.append(pe3.columns[j])
            
cnt3 = Counter(key3)
cnt3_df = pd.DataFrame.from_dict(cnt3, orient='index').reset_index()
cnt3_df.columns = ['keyword', 'count']
cnt3_df['perc'] = cnt3_df['count'] / len(pe3)
#cnt3_df['aafc'] = aafc3
cnt3_df.to_csv("../data/vacant/va3.csv", index=False)

#%% va4
pat_num4 = pe4['patent_number']
aafc4 = pe4['AAFC']
pe4 = pe4.iloc[:,1:-1]
print(aafc4.describe())
sns.histplot(aafc4)
plt.title('Vacant4 Neighbor')
plt.show()

key4 = []
for i in range(len(pe4)):
    for j in range(pe4.shape[1]):
        if pe4.iloc[i][j] != 0:
            key4.append(pe4.columns[j])
            
cnt4 = Counter(key4)
cnt4_df = pd.DataFrame.from_dict(cnt4, orient='index').reset_index()
cnt4_df.columns = ['keyword', 'count']
cnt4_df['perc'] = cnt4_df['count'] / len(pe4)
#cnt4_df['aafc'] = aafc4
cnt4_df.to_csv("../data/vacant/va4.csv", index=False)

#%% va5
pat_num5 = pe5['patent_number']
aafc5 = pe5['AAFC']
pe5 = pe5.iloc[:,1:-1]
print(aafc5.describe())
sns.histplot(aafc5)
plt.title('Vacant5 Neighbor')
plt.show()

key5 = []
for i in range(len(pe5)):
    for j in range(pe5.shape[1]):
        if pe5.iloc[i][j] != 0:
            key5.append(pe5.columns[j])
            
cnt5 = Counter(key5)
cnt5_df = pd.DataFrame.from_dict(cnt5, orient='index').reset_index()
cnt5_df.columns = ['keyword', 'count']
cnt5_df['perc'] = cnt5_df['count'] / len(pe5)
#cnt5_df['aafc'] = aafc5
cnt5_df.to_csv("../data/vacant/va5.csv", index=False)

#%% va6
pat_num6 = pe6['patent_number']
aafc6 = pe6['AAFC']
pe6 = pe6.iloc[:,1:-1]
print(aafc6.describe())
sns.histplot(aafc6)
plt.title('Vacant6 Neighbor')
plt.show()

key6 = []
for i in range(len(pe6)):
    for j in range(pe6.shape[1]):
        if pe6.iloc[i][j] != 0:
            key6.append(pe6.columns[j])
            
cnt6 = Counter(key6)
cnt6_df = pd.DataFrame.from_dict(cnt6, orient='index').reset_index()
cnt6_df.columns = ['keyword', 'count']
cnt6_df['perc'] = cnt6_df['count'] / len(pe6)
#cnt6_df['aafc'] = aafc6
cnt6_df.to_csv("../data/vacant/va6.csv", index=False)

#%% va7
pat_num7 = pe7['patent_number']
aafc7 = pe7['AAFC']
pe7 = pe7.iloc[:,1:-1]
print(aafc7.describe())
sns.histplot(aafc7)
plt.title('Vacant7 Neighbor')
plt.show()

key7 = []
for i in range(len(pe7)):
    for j in range(pe7.shape[1]):
        if pe7.iloc[i][j] != 0:
            key7.append(pe7.columns[j])
            
cnt7 = Counter(key7)
cnt7_df = pd.DataFrame.from_dict(cnt7, orient='index').reset_index()
cnt7_df.columns = ['keyword', 'count']
cnt7_df['perc'] = cnt7_df['count'] / len(pe7)
#cnt7_df['aafc'] = aafc7
cnt7_df.to_csv("../data/vacant/va7.csv", index=False)

#%% va8
pat_num8 = pe8['patent_number']
aafc8 = pe8['AAFC']
pe8 = pe8.iloc[:,1:-1]
print(aafc8.describe())
sns.histplot(aafc8)
plt.title('Vacant8 Neighbor')
plt.show()

key8 = []
for i in range(len(pe8)):
    for j in range(pe8.shape[1]):
        if pe8.iloc[i][j] != 0:
            key8.append(pe8.columns[j])
            
cnt8 = Counter(key8)
cnt8_df = pd.DataFrame.from_dict(cnt8, orient='index').reset_index()
cnt8_df.columns = ['keyword', 'count']
cnt8_df['perc'] = cnt8_df['count'] / len(pe8)
#cnt8_df['aafc'] = aafc8
cnt8_df.to_csv("../data/vacant/va8.csv", index=False)

#%% va9
pat_num9 = pe9['patent_number']
aafc9 = pe9['AAFC']
pe9 = pe9.iloc[:,1:-1]
print(aafc9.describe())
sns.histplot(aafc9, bins=50)
plt.title('Vacant9 Neighbor')
plt.show()

key9 = []
for i in range(len(pe9)):
    for j in range(pe9.shape[1]):
        if pe9.iloc[i][j] != 0:
            key9.append(pe9.columns[j])
            
cnt9 = Counter(key9)
cnt9_df = pd.DataFrame.from_dict(cnt9, orient='index').reset_index()
cnt9_df.columns = ['keyword', 'count']
cnt9_df['perc'] = cnt9_df['count'] / len(pe9)
#cnt9_df['aafc'] = aafc9
cnt9_df.to_csv("../data/vacant/va9.csv", index=False)

#%% va10
pat_num10 = pe10['patent_number']
aafc10 = pe10['AAFC']
pe10 = pe10.iloc[:,1:-1]
print(aafc10.describe())
sns.histplot(aafc10)
plt.title('Vacant10 Neighbor')
plt.show()

key10 = []
for i in range(len(pe10)):
    for j in range(pe10.shape[1]):
        if pe10.iloc[i][j] != 0:
            key10.append(pe10.columns[j])
            
cnt10 = Counter(key10)
cnt10_df = pd.DataFrame.from_dict(cnt10, orient='index').reset_index()
cnt10_df.columns = ['keyword', 'count']
cnt10_df['perc'] = cnt10_df['count'] / len(pe10)
#cnt10_df['aafc'] = aafc10
cnt10_df.to_csv("../data/vacant/va10.csv", index=False)