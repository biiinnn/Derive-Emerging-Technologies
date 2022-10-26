# -*- coding: utf-8 -*-
"""
Created on Thu May 26 20:55:22 2022

@author: yebin
"""

#%% library import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% data
patent = pd.read_csv("../data/textvector/gtm_transformed_tfidf.csv")
tfidf = pd.read_csv("../data/textvector/tfidf.csv")

patent['AAFC'] = tfidf['AAFC']

#%% VA1
# x1: 0.866667, x2: 1
# x1: 0.866667, x2: 0.866667
# x1: 1, x2: 0.733333

va1_num = []
for i in range(len(patent)):
    if round(patent['x1'][i], 4)==0.8667 and round(patent['x2'][i],4)==1.0:
        va1_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==0.8667 and round(patent['x2'][i],4)==0.8667:
        va1_num.append(patent['patent_number'][i])           
    if round(patent['x1'][i], 4)==1 and round(patent['x2'][i],4)==0.7333:
        va1_num.append(patent['patent_number'][i])        

va1_num = pd.DataFrame(va1_num, columns=['patent_number'])
merge1 = pd.merge(va1_num, tfidf, on='patent_number')

#%% VA2
# x1: 0.0667, x2: 1
# x1: -0.0667, x2: 0.8667
# x1: 0.2, x2: 0.8667
# x1: 0.0667, x2: 0.7333

va2_num = []
for i in range(len(patent)):
    if round(patent['x1'][i], 4)==0.0667 and round(patent['x2'][i],4)==1.0:
        va2_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==-0.0667 and round(patent['x2'][i],4)==0.8667:
        va2_num.append(patent['patent_number'][i])        
    if round(patent['x1'][i], 4)==0.2 and round(patent['x2'][i],4)==0.8667:
        va2_num.append(patent['patent_number'][i])        
    if round(patent['x1'][i], 4)==0.0667 and round(patent['x2'][i],4)==0.7333:
        va2_num.append(patent['patent_number'][i])        

va2_num = pd.DataFrame(va2_num, columns=['patent_number'])
merge2 = pd.merge(va2_num, tfidf, on='patent_number')

#%% VA3
# x1: -0.3333, x2: 0.8667
# x1: -0.4667, x2: 0.7333
# x1: -0.3333, x2: 0.6
# x1: -0.2, x2: 0.4667
# x1: -0.0667, x2: 0.6
# x1: -0.2, x2: 0.7333

va3_num = []
for i in range(len(patent)):
    if round(patent['x1'][i], 4)==-0.3333 and round(patent['x2'][i],4)==0.8667:
        va3_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==-0.4667 and round(patent['x2'][i],4)==0.7333:
        va3_num.append(patent['patent_number'][i])        
    if round(patent['x1'][i], 4)==-0.3333 and round(patent['x2'][i],4)==0.6:
        va3_num.append(patent['patent_number'][i])        
    if round(patent['x1'][i], 4)==-0.2 and round(patent['x2'][i],4)==0.4667:
        va3_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==-0.0667 and round(patent['x2'][i],4)==0.6:
        va3_num.append(patent['patent_number'][i])    
    if round(patent['x1'][i], 4)==-0.2 and round(patent['x2'][i],4)==0.7333:
        va3_num.append(patent['patent_number'][i])        

va3_num = pd.DataFrame(va3_num, columns=['patent_number'])
merge3 = pd.merge(va3_num, tfidf, on='patent_number')

#%% VA4
# x1: -1.0, x2: 0.3333
# x1: -0.8667, x2: 0.2
# x1: -1.0, x2: 0.0667

va4_num = []
for i in range(len(patent)):
    if round(patent['x1'][i], 4)==-1.0 and round(patent['x2'][i],4)==0.3333:
        va4_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==-0.8667 and round(patent['x2'][i],4)==0.2:
        va4_num.append(patent['patent_number'][i])        
    if round(patent['x1'][i], 4)==-1.0 and round(patent['x2'][i],4)==0.0667:
        va4_num.append(patent['patent_number'][i])             

va4_num = pd.DataFrame(va4_num, columns=['patent_number'])
merge4 = pd.merge(va4_num, tfidf, on='patent_number')

#%% VA5
# x1: -0.3333, x2: 0.3333
# x1: -0.4667, x2: 0.2
# x1: -0.6, x2: 0.0667
# x1: -0.4667, x2: -0.0667
# x1: -0.6, x2: -0.2
# x1: -0.4667, x2: -0.3333
# x1: -0.3333, x2: -0.2
# x1: -0.2, x2: -0.0667
# x1: -0.2, x2: -0.2
# x1: -0.2, x2: -0.3333
# x1: -0.0667, x2: -0.4667
# x1: 0.0667, x2: -0.3333
# x1: 0.2, x2: -0.2
# x1: 0.2, x2: -0.0667
# x1: 0.0667, x2: 0.0667
# x1: -0.0667, x2: 0.0667
# x1: -0.2, x2: 0.2
    
va5_num = []
for i in range(len(patent)):
    if round(patent['x1'][i], 4)==-0.3333 and round(patent['x2'][i],4)==0.3333:
        va5_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==-0.4667 and round(patent['x2'][i],4)==0.2:
        va5_num.append(patent['patent_number'][i])        
    if round(patent['x1'][i], 4)==-0.6 and round(patent['x2'][i],4)==0.0667:
        va5_num.append(patent['patent_number'][i])             
    if round(patent['x1'][i], 4)==-0.4667 and round(patent['x2'][i],4)==-0.0667:
        va5_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==-0.6 and round(patent['x2'][i],4)==-0.2:
        va5_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==-0.4667 and round(patent['x2'][i],4)==-0.3333:
        va5_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==-0.3333 and round(patent['x2'][i],4)==-0.2:
        va5_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==-0.2 and round(patent['x2'][i],4)==-0.0667:
        va5_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==-0.2 and round(patent['x2'][i],4)==-0.2:
        va5_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==-0.2 and round(patent['x2'][i],4)==-0.3333:
        va5_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==-0.0667 and round(patent['x2'][i],4)==-0.4667:
        va5_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==0.0667 and round(patent['x2'][i],4)==-0.3333:
        va5_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==0.2 and round(patent['x2'][i],4)==-0.2:
        va5_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==0.2 and round(patent['x2'][i],4)==-0.0667:
        va5_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==0.0667 and round(patent['x2'][i],4)==0.0667:
        va5_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==-0.0667 and round(patent['x2'][i],4)==0.0667:
        va5_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==0.2 and round(patent['x2'][i],4)==0.2:
        va5_num.append(patent['patent_number'][i])
va5_num = pd.DataFrame(va5_num, columns=['patent_number'])
merge5 = pd.merge(va5_num, tfidf, on='patent_number')

#%% VA6
# x1: -1.0, x2: 0.0667
# x1: -0.8667, x2: -0.0667
# x1: -0.8667, x2: -0.2
# x1: -1.0, x2: -0.3333

va6_num = []
for i in range(len(patent)):
    if round(patent['x1'][i], 4)==-1.0 and round(patent['x2'][i],4)==0.0667:
        va6_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==-0.8667 and round(patent['x2'][i],4)==-0.0667:
        va6_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==-0.8667 and round(patent['x2'][i],4)==-0.2:
        va6_num.append(patent['patent_number'][i])        
    if round(patent['x1'][i], 4)==-1.0 and round(patent['x2'][i],4)==-0.3333:
        va6_num.append(patent['patent_number'][i])             

va6_num = pd.DataFrame(va6_num, columns=['patent_number'])
merge6 = pd.merge(va6_num, tfidf, on='patent_number')

#%% VA7
# x1: 0.2, x2: -0.2
# x1: 0.0667, x2: -0.3333
# x1: -0.0667, x2: -0.4667
# x1: -0.2, x2: -0.6
# x1: -0.2, x2: -0.7333
# x1: -0.0667, x2: -0.8667
# x1: 0.0667, x2: -0.7333
# x1: 0.2, x2: -0.6
# x1: 0.3333, x2: -0.6
# x1: 0.4667, x2: -0.4667
# x1: 0.3333, x2: -0.3333

va7_num = []
for i in range(len(patent)):
    if round(patent['x1'][i], 4)==0.2 and round(patent['x2'][i],4)==-0.2:
        va7_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==0.0667 and round(patent['x2'][i],4)==-0.3333:
        va7_num.append(patent['patent_number'][i])        
    if round(patent['x1'][i], 4)==-0.0667 and round(patent['x2'][i],4)==-0.4667:
        va7_num.append(patent['patent_number'][i])             
    if round(patent['x1'][i], 4)==-0.2 and round(patent['x2'][i],4)==-0.6:
        va7_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==-0.2 and round(patent['x2'][i],4)==-0.7333:
        va7_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==-0.0667 and round(patent['x2'][i],4)==-0.8667:
        va7_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==0.0667 and round(patent['x2'][i],4)==-0.7333:
        va7_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==0.2 and round(patent['x2'][i],4)==-0.6:
        va7_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==0.3333 and round(patent['x2'][i],4)==-0.6:
        va7_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==0.4667 and round(patent['x2'][i],4)==-0.4667:
        va7_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==0.3333 and round(patent['x2'][i],4)==-0.3333:
        va7_num.append(patent['patent_number'][i])

va7_num = pd.DataFrame(va7_num, columns=['patent_number'])
merge7 = pd.merge(va7_num, tfidf, on='patent_number')

#%% VA8
# x1: 0.6, x2: -0.2
# x1: 0.4667, x2: -0.3333
# x1: 0.4667, x2: -0.4667
# x1: 0.6, x2: -0.6
# x1: 0.7333, x2: -0.7333
# x1: 0.8667, x2: -0.6
# x1: 1.0, x2: -0.4667
# x1: 0.8667, x2: -0.3333
# x1: 0.7333, x2: -0.2

va8_num = []
for i in range(len(patent)):
    if round(patent['x1'][i], 4)==0.6 and round(patent['x2'][i],4)==-0.2:
        va8_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==0.4667 and round(patent['x2'][i],4)==-0.3333:
        va8_num.append(patent['patent_number'][i])        
    if round(patent['x1'][i], 4)==0.4667 and round(patent['x2'][i],4)==-0.4667:
        va8_num.append(patent['patent_number'][i])             
    if round(patent['x1'][i], 4)==0.6 and round(patent['x2'][i],4)==-0.6:
        va8_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==0.7333 and round(patent['x2'][i],4)==-0.7333:
        va8_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==0.8667 and round(patent['x2'][i],4)==-0.6:
        va8_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==1.0 and round(patent['x2'][i],4)==-0.4667:
        va8_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==0.8667 and round(patent['x2'][i],4)==-0.3333:
        va8_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==0.7333 and round(patent['x2'][i],4)==-0.2:
        va8_num.append(patent['patent_number'][i])

va8_num = pd.DataFrame(va8_num, columns=['patent_number'])
merge8 = pd.merge(va8_num, tfidf, on='patent_number')

#%% VA9
# x1: 0.4667, x2: -0.6
# x1: 0.3333, x2: -0.7333
# x1: 0.4667, x2: -0.8667
# x1: 0.6, x2: -0.8667
# x1: 0.7333, x2: -0.7333
# x1: 0.6, x2: -0.6

va9_num = []
for i in range(len(patent)):
    if round(patent['x1'][i], 4)==0.4667 and round(patent['x2'][i],4)==-0.6:
        va9_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==0.3333 and round(patent['x2'][i],4)==-0.7333:
        va9_num.append(patent['patent_number'][i])        
    if round(patent['x1'][i], 4)==0.4667 and round(patent['x2'][i],4)==-0.8667:
        va9_num.append(patent['patent_number'][i])        
    if round(patent['x1'][i], 4)==0.6 and round(patent['x2'][i],4)==-0.8667:
        va9_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==0.7333 and round(patent['x2'][i],4)==-0.7333:
        va9_num.append(patent['patent_number'][i])    
    if round(patent['x1'][i], 4)==0.6 and round(patent['x2'][i],4)==-0.6:
        va9_num.append(patent['patent_number'][i])        

va9_num = pd.DataFrame(va9_num, columns=['patent_number'])
merge9 = pd.merge(va9_num, tfidf, on='patent_number')


#%% VA10
# x1: -0.0667, x2: -0.8667
# x1: -0.2, x2: -1.0
# x1: 0.0667, x2: -1.0

va10_num = []
for i in range(len(patent)):
    if round(patent['x1'][i], 4)==-0.0667 and round(patent['x2'][i],4)==-0.8667:
        va10_num.append(patent['patent_number'][i])
    if round(patent['x1'][i], 4)==-0.2 and round(patent['x2'][i],4)==-1.0:
        va10_num.append(patent['patent_number'][i])        
    if round(patent['x1'][i], 4)==0.0667 and round(patent['x2'][i],4)==-1.0:
        va10_num.append(patent['patent_number'][i])             

va10_num = pd.DataFrame(va10_num, columns=['patent_number'])
merge10 = pd.merge(va10_num, tfidf, on='patent_number')
#%%
merge1.to_csv("../data/perri/perri1.csv", index=False)
merge2.to_csv("../data/perri/perri2.csv", index=False)
merge3.to_csv("../data/perri/perri3.csv", index=False)
merge4.to_csv("../data/perri/perri4.csv", index=False)
merge5.to_csv("../data/perri/perri5.csv", index=False)
merge6.to_csv("../data/perri/perri6.csv", index=False)
merge7.to_csv("../data/perri/perri7.csv", index=False)
merge8.to_csv("../data/perri/perri8.csv", index=False)
merge9.to_csv("../data/perri/perri9.csv", index=False)
merge10.to_csv("../data/perri/perri10.csv", index=False)










